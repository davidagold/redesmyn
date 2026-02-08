use redesmyn_client_api::Client;
use redesmyn_ids::{EpicId, EventId, SessionId, SubscriptionId};
use redesmyn_protocol::client::{
    AgentKind, ArchiveChatSessionRequest, CommandState, CommandSummary, CreateChatSessionRequest,
    CreateChatSessionResponse, EventLogFilter, GetEpicGraphRequest,
    GetEpicPinnedChatSessionRequest, ListAgentModelsRequest, ListChatSessionsRequest,
    ListEpicsRequest, PinChatSessionToEpicRequest, RegenerateChatSessionTitleRequest,
    RegenerateChatSessionTitleResponse, RequestPayload, ResponseResult, SessionModelOption,
    SessionModelSelection, SetSessionCodexApprovalPolicyRequest,
    SetSessionCodexApprovalPolicyResponse, SetSessionCodexSandboxPolicyRequest,
    SetSessionCodexSandboxPolicyResponse, SetSessionModelRequest, SetSessionModelResponse,
    StatusRequest, StatusResponse, SubscriptionEvent, SubscriptionFilter,
    UnpinChatSessionFromEpicRequest, WaitForCommandRequest,
};
use redesmyn_protocol::{CodexApprovalPolicy, CodexSandboxPolicy, ProtocolEnvelope, RepoScope};
use redesmyn_transport::client::in_proc::InProcEndpoint;
use tokio::runtime::Handle;
use tokio::sync::mpsc;

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
        epic_id: Option<EpicId>,
        title: Option<String>,
    ) -> Result<CreateChatSessionResponse, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::CreateChatSession(CreateChatSessionRequest { title, epic_id }),
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
        epic_id: Option<EpicId>,
        include_archived: bool,
        limit: u32,
    ) -> Result<Vec<redesmyn_protocol::client::AgentSessionSummary>, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::ListChatSessions(ListChatSessionsRequest {
                    include_archived,
                    limit,
                    epic_id,
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

    pub async fn archive_chat_session(
        &self,
        scope: RepoScope,
        session_id: SessionId,
    ) -> Result<(), ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::ArchiveChatSession(ArchiveChatSessionRequest { session_id }),
            )
            .await?
        {
            ResponseResult::ArchiveChatSession(_) => Ok(()),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn regenerate_chat_session_title(
        &self,
        scope: RepoScope,
        session_id: SessionId,
    ) -> Result<RegenerateChatSessionTitleResponse, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::RegenerateChatSessionTitle(RegenerateChatSessionTitleRequest {
                    session_id,
                }),
            )
            .await?
        {
            ResponseResult::RegenerateChatSessionTitle(resp) => Ok(resp),
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

    pub async fn list_agent_models(
        &self,
        scope: RepoScope,
        agent_kind: AgentKind,
    ) -> Result<Vec<SessionModelOption>, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::ListAgentModels(ListAgentModelsRequest { agent_kind }),
            )
            .await?
        {
            ResponseResult::ListAgentModels(resp) => Ok(resp.options),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn set_session_model(
        &self,
        scope: RepoScope,
        session_id: SessionId,
        selection: SessionModelSelection,
    ) -> Result<SetSessionModelResponse, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::SetSessionModel(SetSessionModelRequest {
                    session_id,
                    selection,
                }),
            )
            .await?
        {
            ResponseResult::SetSessionModel(resp) => Ok(resp),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn set_session_codex_approval_policy(
        &self,
        scope: RepoScope,
        session_id: SessionId,
        approval_policy: Option<CodexApprovalPolicy>,
    ) -> Result<SetSessionCodexApprovalPolicyResponse, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::SetSessionCodexApprovalPolicy(SetSessionCodexApprovalPolicyRequest {
                    session_id,
                    approval_policy,
                }),
            )
            .await?
        {
            ResponseResult::SetSessionCodexApprovalPolicy(resp) => Ok(resp),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn set_session_codex_sandbox_policy(
        &self,
        scope: RepoScope,
        session_id: SessionId,
        sandbox_policy: Option<CodexSandboxPolicy>,
    ) -> Result<SetSessionCodexSandboxPolicyResponse, ControlPlaneClientError> {
        match self
            .request_scoped(
                scope,
                RequestPayload::SetSessionCodexSandboxPolicy(SetSessionCodexSandboxPolicyRequest {
                    session_id,
                    sandbox_policy,
                }),
            )
            .await?
        {
            ResponseResult::SetSessionCodexSandboxPolicy(resp) => Ok(resp),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn wait_for_command(
        &self,
        command_id: redesmyn_ids::CommandId,
        timeout_ms: u64,
    ) -> Result<CommandSummary, ControlPlaneClientError> {
        match self
            .request(RequestPayload::WaitForCommand(WaitForCommandRequest {
                command_id,
                terminal_states: vec![
                    CommandState::Succeeded,
                    CommandState::Failed,
                    CommandState::Canceled,
                ],
                timeout_ms,
            }))
            .await?
        {
            ResponseResult::WaitForCommand(resp) => Ok(resp.command),
            ResponseResult::Error(err) => Err(ControlPlaneClientError::Server {
                message: err.message,
            }),
            _ => Err(ControlPlaneClientError::UnexpectedMessage),
        }
    }

    pub async fn subscribe_repo_event_log(
        &self,
        scope: RepoScope,
        after_event_id: Option<EventId>,
    ) -> Result<(SubscriptionId, mpsc::Receiver<SubscriptionEvent>), ControlPlaneClientError> {
        let envelope = ProtocolEnvelope::new().with_scope(scope.into());
        self.client
            .subscribe(
                envelope,
                SubscriptionFilter::EventLog(EventLogFilter { after_event_id }),
            )
            .await
            .map_err(|err| ControlPlaneClientError::Client {
                message: err.message,
            })
    }

    pub async fn unsubscribe(
        &self,
        subscription_id: SubscriptionId,
    ) -> Result<(), ControlPlaneClientError> {
        self.client
            .unsubscribe(subscription_id)
            .await
            .map_err(|err| ControlPlaneClientError::Client {
                message: err.message,
            })
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
