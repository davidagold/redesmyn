use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

use redesmyn_logging::tracing;
use redesmyn_protocol::daemon::{ControlPlaneHelloAck, DaemonFrame, DaemonHello, DaemonMessage};
use redesmyn_protocol::{ProtocolEnvelope, ProtocolVersion};
use redesmyn_transport::in_proc::InProcEndpoint;
use redesmyn_transport::{DaemonConnection, TransportError};

use crate::ControlPlane;

/// A minimal control-plane↔daemon adapter for embedded daemon mode.
///
#[derive(Debug)]
pub struct DaemonLinkHandle {
    shutdown_tx: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl DaemonLinkHandle {
    #[must_use]
    pub fn start(
        handle: &tokio::runtime::Handle,
        control_plane: ControlPlane,
        conn: InProcEndpoint,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let task = handle.spawn(run_daemon_link(control_plane, conn, shutdown_rx));
        Self { shutdown_tx, task }
    }

    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);

        if let Err(err) = self.task.await {
            tracing::debug!(error = %err, "daemon link task join error");
        }
    }
}

async fn run_daemon_link(
    control_plane: ControlPlane,
    mut conn: InProcEndpoint,
    mut shutdown_rx: watch::Receiver<bool>,
) {
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
    let DaemonMessage::DaemonHello(hello) = frame.message else {
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

    let (outbound_tx, mut outbound_rx) = mpsc::channel::<DaemonFrame>(64);
    register_daemon(&control_plane, &hello, accepted, outbound_tx).await;

    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        tokio::select! {
            _ = shutdown_rx.changed() => {}
            outbound = outbound_rx.recv() => {
                let Some(frame) = outbound else {
                    break;
                };

                if let Err(err) = conn.send(frame).await {
                    if !matches!(err, TransportError::ChannelClosed) {
                        tracing::warn!(error = %err, "daemon link send error");
                    }
                    break;
                }
            }
            frame = conn.recv() => {
                match frame {
                    Ok(frame) => {
                        handle_inbound_frame(&control_plane, hello.host_instance_id, frame).await;
                    }
                    Err(TransportError::ChannelClosed) => break,
                    Err(err) => {
                        tracing::warn!(error = %err, "daemon link recv error");
                        break;
                    }
                }
            }
        }
    }

    control_plane
        .daemons()
        .unregister_connection(hello.host_instance_id)
        .await;
}

async fn register_daemon(
    control_plane: &ControlPlane,
    hello: &DaemonHello,
    accepted: ProtocolVersion,
    outbound_tx: mpsc::Sender<DaemonFrame>,
) {
    control_plane
        .daemons()
        .register_connection(hello.host_id, hello.host_instance_id, accepted, outbound_tx)
        .await;
}

async fn handle_inbound_frame(
    control_plane: &ControlPlane,
    host_instance_id: redesmyn_ids::HostInstanceId,
    frame: DaemonFrame,
) {
    match frame.message {
        DaemonMessage::CommandUpdate(update) => {
            if let Err(err) = control_plane
                .apply_daemon_command_update(host_instance_id, update)
                .await
            {
                tracing::warn!(error = %err, "failed to apply daemon command update");
            }
        }
        DaemonMessage::SessionEventBatch(batch) => {
            let span = tracing::debug_span!(
                "control_plane.daemon_link.session_event_batch",
                host_instance_id = %host_instance_id,
                event_count = batch.events.len(),
            );
            let _enter = span.enter();

            if let Err(err) = control_plane
                .apply_daemon_session_event_batch(host_instance_id, batch)
                .await
            {
                tracing::warn!(error = %err, "failed to apply daemon session event batch");
            }
        }
        DaemonMessage::SessionLiveEventBatch(batch) => {
            let span = tracing::debug_span!(
                "control_plane.daemon_link.session_live_event_batch",
                host_instance_id = %host_instance_id,
                event_count = batch.events.len(),
            );
            let _enter = span.enter();

            if let Err(err) = control_plane
                .apply_daemon_session_live_event_batch(host_instance_id, batch)
                .await
            {
                tracing::warn!(error = %err, "failed to apply daemon session live event batch");
            }
        }
        DaemonMessage::RepoAttach(attach) => {
            control_plane
                .daemons()
                .attach_repo(host_instance_id, attach.repo_scope)
                .await;
        }
        DaemonMessage::RepoDetach(detach) => {
            control_plane
                .daemons()
                .detach_repo(host_instance_id, detach.repo_scope)
                .await;
        }
        _ => {}
    }
}
