use tokio::sync::mpsc;

use redesmyn_protocol::client::ClientFrame;

use crate::BoxFuture;
use crate::client::{ClientConnection, ClientTransportError};

#[derive(Debug)]
pub struct InProcEndpoint {
    tx: mpsc::Sender<ClientFrame>,
    rx: mpsc::Receiver<ClientFrame>,
}

impl InProcEndpoint {
    #[must_use]
    pub fn pair(buffer: usize) -> (Self, Self) {
        let (a_tx, a_rx) = mpsc::channel(buffer);
        let (b_tx, b_rx) = mpsc::channel(buffer);

        let a = Self { tx: a_tx, rx: b_rx };
        let b = Self { tx: b_tx, rx: a_rx };
        (a, b)
    }

    pub async fn send_frame(&self, frame: ClientFrame) -> Result<(), ClientTransportError> {
        let span = crate::client::span_for_frame("client.in_proc.send", &frame);
        let _enter = span.enter();

        self.tx
            .send(frame)
            .await
            .map_err(|_| ClientTransportError::ChannelClosed)
    }

    pub async fn recv_frame(&mut self) -> Result<ClientFrame, ClientTransportError> {
        let frame = self
            .rx
            .recv()
            .await
            .ok_or(ClientTransportError::ChannelClosed)?;

        let span = crate::client::span_for_frame("client.in_proc.recv", &frame);
        let _enter = span.enter();
        Ok(frame)
    }
}

impl ClientConnection for InProcEndpoint {
    fn send(&mut self, frame: ClientFrame) -> BoxFuture<'_, Result<(), ClientTransportError>> {
        Box::pin(async move { self.send_frame(frame).await })
    }

    fn recv(&mut self) -> BoxFuture<'_, Result<ClientFrame, ClientTransportError>> {
        Box::pin(async move { self.recv_frame().await })
    }
}
