use tokio::sync::watch;
use tokio::task::JoinHandle;

use redesmyn_logging::tracing;
use redesmyn_protocol::daemon::{
    CommandAck, DaemonCommand, DaemonFrame, DaemonMessage, DispatchCommand, HelloRequest, HelloResponse,
    MessageEnvelope,
};
use redesmyn_transport::DaemonConnection;
use redesmyn_transport::in_proc::InProcEndpoint;
use redesmyn_transport::TransportError;

#[derive(Debug)]
pub struct DaemonLinkHandle {
    shutdown_tx: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl DaemonLinkHandle {
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

pub async fn run_daemon_link(mut conn: InProcEndpoint, mut shutdown_rx: watch::Receiver<bool>) {
    let span = tracing::info_span!("desktop.daemon_link");
    let _enter = span.enter();

    let hello = DaemonFrame::new(
        MessageEnvelope::new(redesmyn_ids::MsgId::new()),
        DaemonMessage::HelloRequest(HelloRequest {
            client_name: "redesmyn-desktop".to_string(),
        }),
    );
    if let Err(err) = conn.send(hello).await {
        if !matches!(err, TransportError::ChannelClosed) {
            tracing::warn!(error = %err, "failed to send hello to daemon");
        }
        return;
    }

    let command_id = redesmyn_ids::CommandId::new();
    let noop = DaemonFrame::new(
        MessageEnvelope {
            command_id: Some(command_id),
            ..MessageEnvelope::new(redesmyn_ids::MsgId::new())
        },
        DaemonMessage::DispatchCommand(DispatchCommand {
            command: DaemonCommand::Noop,
        }),
    );
    if let Err(err) = conn.send(noop).await {
        if !matches!(err, TransportError::ChannelClosed) {
            tracing::warn!(error = %err, "failed to send noop command to daemon");
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
                    Ok(frame) => handle_daemon_frame(frame).await,
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

async fn handle_daemon_frame(frame: DaemonFrame) {
    match frame.message {
        DaemonMessage::HelloResponse(HelloResponse { daemon_name }) => {
            tracing::info!(daemon_name = %daemon_name, "daemon handshake complete");
        }
        DaemonMessage::CommandAck(CommandAck { status }) => {
            tracing::info!(?status, "daemon command ack");
        }
        DaemonMessage::Heartbeat(_) => {}
        DaemonMessage::Event(_) => {}
        DaemonMessage::HelloRequest(_) | DaemonMessage::DispatchCommand(_) => {
            tracing::debug!(?frame.message, "unexpected daemon message");
        }
    }
}
