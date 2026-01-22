use std::sync::Arc;

use redesmyn_ids::RequestId;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, ListEpicsRequest, Request, RequestPayload, ResponseResult,
    StatusRequest, StatusResponse,
};
use redesmyn_transport::client::{ClientConnection, ClientTransportError};
use tokio::runtime::Handle;
use tokio::sync::Mutex;

#[derive(Clone, Debug)]
pub struct ControlPlaneClient {
    tokio: Handle,
    conn: Arc<Mutex<redesmyn_transport::client::in_proc::InProcEndpoint>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ControlPlaneClientError {
    #[error(transparent)]
    Transport(#[from] ClientTransportError),
    #[error("control plane returned an error: {message}")]
    Server { message: String },
    #[error("control plane sent an unexpected message")]
    UnexpectedMessage,
}

impl ControlPlaneClient {
    pub fn new(tokio: Handle, conn: redesmyn_transport::client::in_proc::InProcEndpoint) -> Self {
        Self {
            tokio,
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    pub fn tokio(&self) -> &Handle {
        &self.tokio
    }

    pub async fn status(&self) -> Result<StatusResponse, ControlPlaneClientError> {
        match self
            .request(RequestPayload::Status(StatusRequest {}))
            .await?
        {
            ResponseResult::Status(status) => Ok(status),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn list_epics(
        &self,
    ) -> Result<Vec<redesmyn_protocol::client::EpicSummary>, ControlPlaneClientError> {
        match self
            .request(RequestPayload::ListEpics(ListEpicsRequest {}))
            .await?
        {
            ResponseResult::ListEpics(resp) => Ok(resp.epics),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    async fn request(
        &self,
        payload: RequestPayload,
    ) -> Result<ResponseResult, ControlPlaneClientError> {
        let request_id = RequestId::new();
        let frame = ClientFrame::new(
            redesmyn_protocol::ProtocolEnvelope::new(),
            ClientMessage::Request(Request {
                request_id,
                payload,
            }),
        );

        let mut conn = self.conn.lock().await;
        conn.send(frame).await?;

        loop {
            let frame = conn.recv().await?;
            match frame.message {
                ClientMessage::Response(resp) if resp.request_id == request_id => {
                    return Ok(resp.result);
                }
                ClientMessage::Response(_) => continue,
                ClientMessage::Event(_)
                | ClientMessage::Subscribe(_)
                | ClientMessage::Unsubscribe(_) => {
                    return Err(ControlPlaneClientError::UnexpectedMessage);
                }
                ClientMessage::Request(_) => {
                    return Err(ControlPlaneClientError::UnexpectedMessage);
                }
            }
        }
    }
}
