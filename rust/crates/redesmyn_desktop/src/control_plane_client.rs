use std::sync::Arc;

use redesmyn_ids::{EpicId, RequestId, SessionId};
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, CloseChatSessionRequest, CreateChatSessionRequest,
    CreateChatSessionResponse, GetEpicGraphRequest, ListEpicsRequest, PinChatSessionToEpicRequest,
    Request, RequestPayload, ResponseResult, StatusRequest, StatusResponse,
    UnpinChatSessionFromEpicRequest,
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

    pub async fn get_epic_graph(
        &self,
        epic_slug: String,
    ) -> Result<redesmyn_protocol::client::EpicGraph, ControlPlaneClientError> {
        match self
            .request(RequestPayload::GetEpicGraph(GetEpicGraphRequest {
                epic_slug,
            }))
            .await?
        {
            ResponseResult::GetEpicGraph(resp) => Ok(resp.graph),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn create_chat_session(
        &self,
        title: Option<String>,
    ) -> Result<CreateChatSessionResponse, ControlPlaneClientError> {
        match self
            .request(RequestPayload::CreateChatSession(
                CreateChatSessionRequest { title },
            ))
            .await?
        {
            ResponseResult::CreateChatSession(resp) => Ok(resp),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn close_chat_session(
        &self,
        session_id: SessionId,
    ) -> Result<(), ControlPlaneClientError> {
        match self
            .request(RequestPayload::CloseChatSession(CloseChatSessionRequest {
                session_id,
            }))
            .await?
        {
            ResponseResult::CloseChatSession(_) => Ok(()),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn pin_chat_session_to_epic(
        &self,
        epic_id: EpicId,
        session_id: SessionId,
    ) -> Result<(), ControlPlaneClientError> {
        match self
            .request(RequestPayload::PinChatSessionToEpic(
                PinChatSessionToEpicRequest {
                    epic_id,
                    session_id,
                },
            ))
            .await?
        {
            ResponseResult::PinChatSessionToEpic(_) => Ok(()),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn unpin_chat_session_from_epic(
        &self,
        epic_id: EpicId,
    ) -> Result<(), ControlPlaneClientError> {
        match self
            .request(RequestPayload::UnpinChatSessionFromEpic(
                UnpinChatSessionFromEpicRequest { epic_id },
            ))
            .await?
        {
            ResponseResult::UnpinChatSessionFromEpic(_) => Ok(()),
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
