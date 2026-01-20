use tokio::sync::watch;
use tokio::task::JoinHandle;

use redesmyn_logging::tracing;
use redesmyn_protocol::daemon::{ControlPlaneHelloAck, DaemonFrame, DaemonMessage};
use redesmyn_protocol::{ProtocolEnvelope, ProtocolVersion};
use redesmyn_transport::in_proc::InProcEndpoint;
use redesmyn_transport::{DaemonConnection, TransportError};

/// A minimal control-plane↔daemon adapter for embedded daemon mode.
///
/// For now this keeps the in-proc daemon connection drained so heartbeats don't
/// backpressure the daemon, and performs a basic T-11 handshake to prove the
/// wiring works. Higher-level routing belongs in the control-plane domain.
#[derive(Debug)]
pub struct DaemonLinkHandle {
    shutdown_tx: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl DaemonLinkHandle {
    #[must_use]
    pub fn start(runtime: &tokio::runtime::Runtime, conn: InProcEndpoint) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let task = runtime.spawn(run_daemon_link(conn, shutdown_rx));
        Self { shutdown_tx, task }
    }

    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);

        if let Err(err) = self.task.await {
            tracing::debug!(error = %err, "daemon link task join error");
        }
    }
}

async fn run_daemon_link(mut conn: InProcEndpoint, mut shutdown_rx: watch::Receiver<bool>) {
    let span = tracing::info_span!("control_plane.daemon_link");
    let _enter = span.enter();

    let frame = tokio::select! {
        _ = shutdown_rx.changed() => return,
        frame = conn.recv() => frame,
    };
    let frame = match frame {
        Ok(frame) => frame,
        Err(TransportError::ChannelClosed) => return,
        Err(err) => {
            tracing::warn!(error = %err, "daemon link transport error");
            return;
        }
    };

    let peer_version = frame.envelope.protocol_version();
    let DaemonMessage::DaemonHello(_) = frame.message else {
        tracing::warn!(?frame.message, "unexpected daemon message during handshake");
        return;
    };

    let accepted = match ProtocolVersion::CURRENT.negotiate(peer_version) {
        Ok(accepted) => accepted,
        Err(err) => {
            tracing::warn!(error = %err.message, "failed to negotiate daemon protocol version");
            return;
        }
    };

    let mut envelope = ProtocolEnvelope::new();
    envelope.protocol_major = accepted.major;
    envelope.protocol_minor = accepted.minor;
    envelope.correlation_id = Some(frame.envelope.msg_id);

    let ack = DaemonFrame::new(
        envelope,
        DaemonMessage::ControlPlaneHelloAck(ControlPlaneHelloAck {
            accepted_protocol: accepted,
            capabilities: vec!["stub".to_string()],
        }),
    );

    if let Err(err) = conn.send(ack).await {
        if !matches!(err, TransportError::ChannelClosed) {
            tracing::warn!(error = %err, "failed to send handshake ack to daemon");
        }
        return;
    }

    loop {
        if *shutdown_rx.borrow() {
            return;
        }

        tokio::select! {
            _ = shutdown_rx.changed() => {}
            frame = conn.recv() => {
                match frame {
                    Ok(_frame) => {}
                    Err(TransportError::ChannelClosed) => return,
                    Err(err) => {
                        tracing::warn!(error = %err, "daemon link transport error");
                        return;
                    }
                }
            }
        }
    }
}
