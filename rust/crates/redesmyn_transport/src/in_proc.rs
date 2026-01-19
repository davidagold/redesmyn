use tokio::sync::mpsc;

use redesmyn_protocol::daemon::DaemonFrame;

use crate::{BoxFuture, DaemonConnection, TransportError};

#[derive(Debug)]
pub struct InProcEndpoint {
    tx: mpsc::Sender<DaemonFrame>,
    rx: mpsc::Receiver<DaemonFrame>,
}

impl InProcEndpoint {
    #[must_use]
    pub fn pair(buffer: usize) -> (Self, Self) {
        let (cp_to_daemon_tx, cp_to_daemon_rx) = mpsc::channel(buffer);
        let (daemon_to_cp_tx, daemon_to_cp_rx) = mpsc::channel(buffer);

        let control_plane = Self {
            tx: cp_to_daemon_tx,
            rx: daemon_to_cp_rx,
        };
        let daemon = Self {
            tx: daemon_to_cp_tx,
            rx: cp_to_daemon_rx,
        };

        (control_plane, daemon)
    }

    pub async fn send_frame(&self, frame: DaemonFrame) -> Result<(), TransportError> {
        let span = crate::span_for_envelope("daemon.in_proc.send", &frame.envelope);
        let _enter = span.enter();

        self.tx
            .send(frame)
            .await
            .map_err(|_| TransportError::ChannelClosed)
    }

    pub async fn recv_frame(&mut self) -> Result<DaemonFrame, TransportError> {
        let frame = self.rx.recv().await.ok_or(TransportError::ChannelClosed)?;

        let span = crate::span_for_envelope("daemon.in_proc.recv", &frame.envelope);
        let _enter = span.enter();
        Ok(frame)
    }
}

impl DaemonConnection for InProcEndpoint {
    fn send(&mut self, frame: DaemonFrame) -> BoxFuture<'_, Result<(), TransportError>> {
        Box::pin(async move { self.send_frame(frame).await })
    }

    fn recv(&mut self) -> BoxFuture<'_, Result<DaemonFrame, TransportError>> {
        Box::pin(async move { self.recv_frame().await })
    }
}
