use redesmyn_client_api::Client;
use redesmyn_ids::{EpicId, SessionId};
use redesmyn_protocol::client::{
    CloseChatSessionRequest, CreateChatSessionRequest, CreateChatSessionResponse,
    GetEpicGraphRequest, GetEpicPinnedChatSessionRequest, ListChatSessionsRequest,
    ListEpicsRequest, PinChatSessionToEpicRequest, RequestPayload, ResponseResult, StatusRequest,
    StatusResponse, UnpinChatSessionFromEpicRequest,
};
use redesmyn_protocol::{ProtocolEnvelope, RepoScope};
use redesmyn_transport::client::in_proc::InProcEndpoint;
use tokio::runtime::Handle;

#[derive(Clone, Debug)]
pub struct ControlPlaneClient {
    tokio: Handle,
    client: Client,
}

#[derive(Debug, thiserror::Error)]
pub enum ControlPlaneClientError {
    #[error("control plane client error: {message}")]
    Client { message: String },
    #[error("control plane returned an error: {message}")]
    Server { message: String },
    #[error("control plane sent an unexpected message")]
    UnexpectedMessage,
}

impl ControlPlaneClient {
    pub fn new(tokio: Handle, conn: InProcEndpoint) -> Self {
        let (client, task) = Client::connect(conn, 64);
        tokio.spawn(task.run());

        Self { tokio, client }
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

    pub async fn get_epic_pinned_chat_session(
        &self,
        scope: RepoScope,
        epic_id: EpicId,
    ) -> Result<Option<SessionId>, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::GetEpicPinnedChatSession(GetEpicPinnedChatSessionRequest {
                    epic_id,
                }),
            )
            .await?
        {
            ResponseResult::GetEpicPinnedChatSession(resp) => Ok(resp.session_id),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn create_chat_session(
        &self,
        scope: RepoScope,
        title: Option<String>,
    ) -> Result<CreateChatSessionResponse, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::CreateChatSession(CreateChatSessionRequest { title }),
            )
            .await?
        {
            ResponseResult::CreateChatSession(resp) => Ok(resp),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn list_chat_sessions(
        &self,
        scope: RepoScope,
        include_closed: bool,
        limit: u32,
    ) -> Result<Vec<redesmyn_protocol::client::AgentSessionSummary>, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::ListChatSessions(ListChatSessionsRequest {
                    include_closed,
                    limit,
                }),
            )
            .await?
        {
            ResponseResult::ListChatSessions(resp) => Ok(resp.sessions),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn close_chat_session(
        &self,
        scope: RepoScope,
        session_id: SessionId,
    ) -> Result<(), ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::CloseChatSession(CloseChatSessionRequest { session_id }),
            )
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
        scope: RepoScope,
        epic_id: EpicId,
        session_id: SessionId,
    ) -> Result<(), ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::PinChatSessionToEpic(PinChatSessionToEpicRequest {
                    epic_id,
                    session_id,
                }),
            )
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
        scope: RepoScope,
        epic_id: EpicId,
    ) -> Result<(), ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::UnpinChatSessionFromEpic(UnpinChatSessionFromEpicRequest {
                    epic_id,
                }),
            )
            .await?
        {
            ResponseResult::UnpinChatSessionFromEpic(_) => Ok(()),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    async fn request_scoped(
        &self,
        scope: RepoScope,
        payload: RequestPayload,
    ) -> Result<ResponseResult, ControlPlaneClientError> {
        let envelope = ProtocolEnvelope::new().with_scope(scope.into());
        self.request_with_envelope(envelope, payload).await
    }

    async fn request(
        &self,
        payload: RequestPayload,
    ) -> Result<ResponseResult, ControlPlaneClientError> {
        self.request_with_envelope(ProtocolEnvelope::new(), payload)
            .await
    }

    async fn request_with_envelope(
        &self,
        envelope: ProtocolEnvelope,
        payload: RequestPayload,
    ) -> Result<ResponseResult, ControlPlaneClientError> {
        self.client
            .request_with_envelope(envelope, payload)
            .await
            .map_err(|err| ControlPlaneClientError::Client {
                message: err.message,
            })
    }
}
