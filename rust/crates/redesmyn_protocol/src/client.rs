//! Protocol types for the client ↔ control plane API boundary (T-12).
//!
//! This module defines the canonical typed message model used by:
//! - out-of-proc clients over UDS/TCP (framed + Protobuf by default), and
//! - embedded in-proc clients (typed messages; optional codec loopback in tests).

use redesmyn_ids::{
    CommandId, CommandUpdateId, EpicId, EventId, HostId, HostInstanceId, RepoId, RequestId,
    SessionEventId, SessionId, SubscriptionId, TaskId, WorkspaceId,
};

use crate::session::ImageAttachment;
use crate::{
    CodexApprovalPolicy, CodexSandboxPolicy, ErrorEnvelope, PermissionDecision, PermissionsMode,
    ProtocolEnvelope, ProtocolVersion, Scope, SessionEvent, SessionLiveEvent, Timestamp,
};

/// A single client ↔ control plane protocol frame.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClientFrame {
    pub envelope: ProtocolEnvelope,
    pub message: ClientMessage,
}

impl ClientFrame {
    #[must_use]
    pub fn new(envelope: ProtocolEnvelope, message: ClientMessage) -> Self {
        Self { envelope, message }
    }
}

/// Client API method identifier (initial minimal set; additive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientMethod {
    Health,
    Status,
    ListEpics,
    GetEpicGraph,
    GetSessionEvents,
    GetLatestTaskSession,
    GetEpicPinnedChatSession,
    SendSessionMessage,
    SetSessionPermissionsMode,
    SetSessionCodexApprovalPolicy,
    SetSessionCodexSandboxPolicy,
    ListAgentModels,
    ListSessionModels,
    SetSessionModel,
    RegenerateChatSessionTitle,
    RespondPermissionRequest,
    StartAgent,
    StopAgent,
    RestartAgent,
    SendTaskAgentMessage,
    AttachAgentSession,
    CreateCommand,
    GetCommand,
    WaitForCommand,
    WaitForEvent,
    WaitForIdle,
    ListTaskSessions,
    CreateChatSession,
    ArchiveChatSession,
    ListChatSessions,
    PinChatSessionToEpic,
    UnpinChatSessionFromEpic,
}

/// A request issued by a client.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Request {
    pub request_id: RequestId,
    pub payload: RequestPayload,
}

impl Request {
    #[must_use]
    pub const fn method(&self) -> ClientMethod {
        self.payload.method()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum RequestPayload {
    Health(HealthRequest),
    Status(StatusRequest),
    ListEpics(ListEpicsRequest),
    GetEpicGraph(GetEpicGraphRequest),
    GetSessionEvents(GetSessionEventsRequest),
    GetLatestTaskSession(GetLatestTaskSessionRequest),
    GetEpicPinnedChatSession(GetEpicPinnedChatSessionRequest),
    SendSessionMessage(SendSessionMessageRequest),
    SetSessionPermissionsMode(SetSessionPermissionsModeRequest),
    SetSessionCodexApprovalPolicy(SetSessionCodexApprovalPolicyRequest),
    SetSessionCodexSandboxPolicy(SetSessionCodexSandboxPolicyRequest),
    ListAgentModels(ListAgentModelsRequest),
    ListSessionModels(ListSessionModelsRequest),
    SetSessionModel(SetSessionModelRequest),
    RespondPermissionRequest(RespondPermissionRequestRequest),
    StartAgent(StartAgentRequest),
    StopAgent(StopAgentRequest),
    RestartAgent(RestartAgentRequest),
    SendTaskAgentMessage(SendTaskAgentMessageRequest),
    AttachAgentSession(AttachAgentSessionRequest),
    CreateCommand(CreateCommandRequest),
    GetCommand(GetCommandRequest),
    WaitForCommand(WaitForCommandRequest),
    WaitForEvent(WaitForEventRequest),
    WaitForIdle(WaitForIdleRequest),
    ListTaskSessions(ListTaskSessionsRequest),
    CreateChatSession(CreateChatSessionRequest),
    ArchiveChatSession(ArchiveChatSessionRequest),
    ListChatSessions(ListChatSessionsRequest),
    RegenerateChatSessionTitle(RegenerateChatSessionTitleRequest),
    PinChatSessionToEpic(PinChatSessionToEpicRequest),
    UnpinChatSessionFromEpic(UnpinChatSessionFromEpicRequest),
}

impl RequestPayload {
    #[must_use]
    pub const fn method(&self) -> ClientMethod {
        match self {
            Self::Health(_) => ClientMethod::Health,
            Self::Status(_) => ClientMethod::Status,
            Self::ListEpics(_) => ClientMethod::ListEpics,
            Self::GetEpicGraph(_) => ClientMethod::GetEpicGraph,
            Self::GetSessionEvents(_) => ClientMethod::GetSessionEvents,
            Self::GetLatestTaskSession(_) => ClientMethod::GetLatestTaskSession,
            Self::GetEpicPinnedChatSession(_) => ClientMethod::GetEpicPinnedChatSession,
            Self::SendSessionMessage(_) => ClientMethod::SendSessionMessage,
            Self::SetSessionPermissionsMode(_) => ClientMethod::SetSessionPermissionsMode,
            Self::SetSessionCodexApprovalPolicy(_) => ClientMethod::SetSessionCodexApprovalPolicy,
            Self::SetSessionCodexSandboxPolicy(_) => ClientMethod::SetSessionCodexSandboxPolicy,
            Self::ListAgentModels(_) => ClientMethod::ListAgentModels,
            Self::ListSessionModels(_) => ClientMethod::ListSessionModels,
            Self::SetSessionModel(_) => ClientMethod::SetSessionModel,
            Self::RespondPermissionRequest(_) => ClientMethod::RespondPermissionRequest,
            Self::StartAgent(_) => ClientMethod::StartAgent,
            Self::StopAgent(_) => ClientMethod::StopAgent,
            Self::RestartAgent(_) => ClientMethod::RestartAgent,
            Self::SendTaskAgentMessage(_) => ClientMethod::SendTaskAgentMessage,
            Self::AttachAgentSession(_) => ClientMethod::AttachAgentSession,
            Self::CreateCommand(_) => ClientMethod::CreateCommand,
            Self::GetCommand(_) => ClientMethod::GetCommand,
            Self::WaitForCommand(_) => ClientMethod::WaitForCommand,
            Self::WaitForEvent(_) => ClientMethod::WaitForEvent,
            Self::WaitForIdle(_) => ClientMethod::WaitForIdle,
            Self::ListTaskSessions(_) => ClientMethod::ListTaskSessions,
            Self::CreateChatSession(_) => ClientMethod::CreateChatSession,
            Self::ArchiveChatSession(_) => ClientMethod::ArchiveChatSession,
            Self::ListChatSessions(_) => ClientMethod::ListChatSessions,
            Self::RegenerateChatSessionTitle(_) => ClientMethod::RegenerateChatSessionTitle,
            Self::PinChatSessionToEpic(_) => ClientMethod::PinChatSessionToEpic,
            Self::UnpinChatSessionFromEpic(_) => ClientMethod::UnpinChatSessionFromEpic,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Ok,
    Error,
}

/// A response issued by the control plane.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Response {
    pub request_id: RequestId,
    pub result: ResponseResult,
}

impl Response {
    #[must_use]
    pub const fn status(&self) -> ResponseStatus {
        match &self.result {
            ResponseResult::Error(_) => ResponseStatus::Error,
            _ => ResponseStatus::Ok,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ResponseResult {
    Health(HealthResponse),
    Status(StatusResponse),
    ListEpics(ListEpicsResponse),
    GetEpicGraph(GetEpicGraphResponse),
    GetSessionEvents(GetSessionEventsResponse),
    GetLatestTaskSession(GetLatestTaskSessionResponse),
    GetEpicPinnedChatSession(GetEpicPinnedChatSessionResponse),
    SendSessionMessage(SendSessionMessageResponse),
    SetSessionPermissionsMode(SetSessionPermissionsModeResponse),
    SetSessionCodexApprovalPolicy(SetSessionCodexApprovalPolicyResponse),
    SetSessionCodexSandboxPolicy(SetSessionCodexSandboxPolicyResponse),
    ListAgentModels(ListAgentModelsResponse),
    ListSessionModels(ListSessionModelsResponse),
    SetSessionModel(SetSessionModelResponse),
    RespondPermissionRequest(RespondPermissionRequestResponse),
    StartAgent(StartAgentResponse),
    StopAgent(StopAgentResponse),
    RestartAgent(RestartAgentResponse),
    SendTaskAgentMessage(SendTaskAgentMessageResponse),
    AttachAgentSession(AttachAgentSessionResponse),
    CreateCommand(CreateCommandResponse),
    GetCommand(GetCommandResponse),
    WaitForCommand(WaitForCommandResponse),
    WaitForEvent(WaitForEventResponse),
    WaitForIdle(WaitForIdleResponse),
    ListTaskSessions(ListTaskSessionsResponse),
    CreateChatSession(CreateChatSessionResponse),
    ArchiveChatSession(ArchiveChatSessionResponse),
    ListChatSessions(ListChatSessionsResponse),
    RegenerateChatSessionTitle(RegenerateChatSessionTitleResponse),
    PinChatSessionToEpic(PinChatSessionToEpicResponse),
    UnpinChatSessionFromEpic(UnpinChatSessionFromEpicResponse),
    Error(ErrorEnvelope),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HealthRequest {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StatusRequest {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StatusResponse {
    pub accepted_protocol: ProtocolVersion,
    pub server_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListEpicsRequest {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EpicSummary {
    pub slug: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epic_id: Option<EpicId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListEpicsResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub epics: Vec<EpicSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetEpicGraphRequest {
    pub epic_slug: String,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Unknown,
    Todo,
    InProgress,
    Blocked,
    Done,
}

impl Default for TaskState {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum MergeReadiness {
    Unknown,
    Ready,
    Blocked,
}

impl Default for MergeReadiness {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DirectorModeLifecycle {
    Unknown,
    Inactive,
    Active,
    Paused,
    ResumeRequired,
    Error,
}

impl Default for DirectorModeLifecycle {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DirectorActivationIntent {
    RunInCurrentSession,
    RunInNewSession,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DirectorModeSummary {
    #[serde(default)]
    pub lifecycle: DirectorModeLifecycle,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub director_session_id: Option<SessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_intent: Option<DirectorActivationIntent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_required_at: Option<Timestamp>,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DirectorMergeAuthorityPolicySource {
    GlobalDefault,
    EpicOverride,
    #[serde(other)]
    Unknown,
}

impl Default for DirectorMergeAuthorityPolicySource {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DirectorMergeAuthorityPolicy {
    pub yolo_merge: bool,
    #[serde(default)]
    pub source: DirectorMergeAuthorityPolicySource,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EpicTaskNode {
    pub task_slug: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_task_id: Option<TaskId>,
    #[serde(default)]
    pub state: TaskState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_name: Option<String>,
    #[serde(default)]
    pub merge_readiness: MergeReadiness,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EpicTaskEdge {
    pub from_task_slug: String,
    pub to_task_slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_task_id: Option<TaskId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandState {
    Unknown,
    Queued,
    Accepted,
    Running,
    Blocked,
    Resumable,
    Succeeded,
    Failed,
    Canceled,
}

impl Default for CommandState {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommandUpdateSummary {
    pub update_id: CommandUpdateId,
    pub created_at: Timestamp,
    pub state: CommandState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_current: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_total: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommandSummary {
    pub command_id: CommandId,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub kind: String,
    pub state: CommandState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_update: Option<CommandUpdateSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DaemonPresenceSummary {
    pub host_instance_id: HostInstanceId,
    pub host_id: HostId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    pub connected_at: Timestamp,
    pub last_heartbeat_at: Timestamp,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disconnected_at: Option<Timestamp>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionSummary {
    pub session_id: SessionId,
    pub session_event_id: SessionEventId,
    pub task_id: TaskId,
    pub last_event_at: Timestamp,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_preview: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EpicGraph {
    pub epic_slug: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<EpicTaskNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<EpicTaskEdge>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epic_id: Option<EpicId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epic_title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<WorkspaceId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<RepoId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_slug: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_title: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command_summaries: Vec<CommandSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub daemon_presences: Vec<DaemonPresenceSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub session_summaries: Vec<SessionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub director_mode: Option<DirectorModeSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_authority_policy: Option<DirectorMergeAuthorityPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub as_of_event_id: Option<EventId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetEpicGraphResponse {
    pub graph: EpicGraph,
}

/// Stable cursor for session event pagination/subscriptions.
///
/// Ordering: `(created_at, session_event_id)` tuple.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct SessionEventCursor {
    pub created_at: Timestamp,
    pub session_event_id: SessionEventId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventKindFilter {
    SessionStarted,
    SessionEnded,
    TurnStarted,
    TurnCompleted,
    UserMessage,
    AssistantMessage,
    AssistantReasoning,
    ToolInvocation,
    ToolResult,
    StatusUpdate,
    TaskAgentMessageSent,
    ArtifactEmitted,
    PermissionsModeChanged,
    PermissionRequested,
    PermissionDecided,
    CodexApprovalPolicyChanged,
    CodexSandboxPolicyChanged,
    SessionModelChanged,
    /// A kind not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetSessionEventsRequest {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<SessionEventCursor>,
    pub limit: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kinds: Vec<SessionEventKindFilter>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetSessionEventsResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<SessionEvent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<SessionEventCursor>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetLatestTaskSessionRequest {
    pub task_id: TaskId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetLatestTaskSessionResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetEpicPinnedChatSessionRequest {
    pub epic_id: EpicId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetEpicPinnedChatSessionResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMessageConflictAction {
    Fail,
    InterruptTurn,
    StopSessionAndStartNew,
}

impl Default for AgentMessageConflictAction {
    fn default() -> Self {
        Self::Fail
    }
}

pub const SEND_SESSION_MESSAGE_MAX_IMAGE_ATTACHMENTS: usize = 8;
pub const SEND_SESSION_MESSAGE_MAX_IMAGE_ATTACHMENT_BYTES: u64 = 8 * 1024 * 1024;
pub const SEND_SESSION_MESSAGE_MAX_IMAGE_TOTAL_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SendSessionMessageRequest {
    pub session_id: SessionId,
    pub message: String,
    #[serde(default)]
    pub on_conflict: AgentMessageConflictAction,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_attachments: Vec<ImageAttachment>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SendSessionMessageResponse {
    pub event: SessionEvent,
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionPermissionsModeRequest {
    pub session_id: SessionId,
    pub mode: PermissionsMode,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionPermissionsModeResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionCodexApprovalPolicyRequest {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<CodexApprovalPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionCodexApprovalPolicyResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionCodexSandboxPolicyRequest {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_policy: Option<CodexSandboxPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionCodexSandboxPolicyResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionModelOption {
    pub model_id: String,
    pub display_name: String,
    pub description: String,
    pub is_default: bool,
    pub provider_model: String,
    pub default_reasoning_effort: ModelReasoningEffort,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_reasoning_efforts: Vec<ModelReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionModelSelection {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ModelReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListSessionModelsRequest {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListAgentModelsRequest {
    pub agent_kind: AgentKind,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListAgentModelsResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<SessionModelOption>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListSessionModelsResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<SessionModelOption>,
    pub selection: SessionModelSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionModelRequest {
    pub session_id: SessionId,
    pub selection: SessionModelSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionModelResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RespondPermissionRequestRequest {
    pub session_id: SessionId,
    pub request_id: String,
    pub decision: PermissionDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RespondPermissionRequestResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<CommandSummary>,
}

// Agent orchestration (T-41).

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskAgentMessageDelivery {
    StructuredStarted,
    StructuredResumed,
    InteractiveStarted,
    InteractiveSent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskAgentMessageConversationContinuity {
    Kept,
    Broken,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StartAgentRequest {
    pub task_id: TaskId,
    pub agent_kind: AgentKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
    #[serde(default)]
    pub on_conflict: AgentMessageConflictAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_model_selection: Option<SessionModelSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_approval_policy: Option<CodexApprovalPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_sandbox_policy: Option<CodexSandboxPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StartAgentResponse {
    pub command: CommandSummary,
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StopAgentRequest {
    pub task_id: TaskId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StopAgentResponse {
    pub command: CommandSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ended_session_ids: Vec<SessionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RestartAgentRequest {
    pub task_id: TaskId,
    pub agent_kind: AgentKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_model_selection: Option<SessionModelSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_approval_policy: Option<CodexApprovalPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_sandbox_policy: Option<CodexSandboxPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RestartAgentResponse {
    pub command: CommandSummary,
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SendTaskAgentMessageRequest {
    pub task_id: TaskId,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    #[serde(default)]
    pub on_conflict: AgentMessageConflictAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupt: Option<bool>,
    pub agent_kind: AgentKind,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SendTaskAgentMessageResponse {
    pub command: CommandSummary,
    pub session_id: SessionId,
    pub delivery: TaskAgentMessageDelivery,
    pub conversation_continuity: TaskAgentMessageConversationContinuity,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AttachAgentSessionRequest {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AttachAgentSessionResponse {
    pub command: CommandSummary,
}
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CreateCommandRequest {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CreateCommandResponse {
    pub command: CommandSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetCommandRequest {
    pub command_id: CommandId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetCommandResponse {
    pub command: CommandSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForCommandRequest {
    pub command_id: CommandId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub terminal_states: Vec<CommandState>,
    #[serde(default)]
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForCommandResponse {
    pub command: CommandSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EventWaitFilter {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub event_type_prefix: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_event_id: Option<EventId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForEventRequest {
    pub filter: EventWaitFilter,
    #[serde(default)]
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForEventResponse {
    pub event_log: EventLogEvent,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForIdleRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    #[serde(default)]
    pub timeout_ms: u64,
    #[serde(default)]
    pub quiescence_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForIdleResponse {}

// Session/query surfaces (T-40).

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    Codex,
    ClaudeCode,
    Shell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSessionScopeKind {
    Task,
    Chat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentSessionStatus {
    Running,
    Blocked,
    Stopped,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AgentSessionSummary {
    pub session_id: SessionId,
    pub scope_kind: AgentSessionScopeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epic_id: Option<EpicId>,
    pub agent_kind: AgentKind,
    pub status: AgentSessionStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived_at: Option<Timestamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<Timestamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<Timestamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<Timestamp>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListTaskSessionsRequest {
    pub task_id: TaskId,
    pub limit: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListTaskSessionsResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sessions: Vec<AgentSessionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_session_id: Option<SessionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CreateChatSessionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epic_id: Option<EpicId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CreateChatSessionResponse {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArchiveChatSessionRequest {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArchiveChatSessionResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListChatSessionsRequest {
    #[serde(default)]
    pub include_archived: bool,
    pub limit: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epic_id: Option<EpicId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListChatSessionsResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sessions: Vec<AgentSessionSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegenerateChatSessionTitleRequest {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RegenerateChatSessionTitleResponse {
    pub title: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PinChatSessionToEpicRequest {
    pub epic_id: EpicId,
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PinChatSessionToEpicResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnpinChatSessionFromEpicRequest {
    pub epic_id: EpicId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnpinChatSessionFromEpicResponse {}

/// Subscription topics for server-pushed streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionTopic {
    EventLog,
    SessionEvents,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Subscribe {
    pub subscription_id: SubscriptionId,
    pub filter: SubscriptionFilter,
}

impl Subscribe {
    #[must_use]
    pub const fn topic(&self) -> SubscriptionTopic {
        self.filter.topic()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SubscriptionFilter {
    EventLog(EventLogFilter),
    SessionEvents(SessionEventsFilter),
}

impl SubscriptionFilter {
    #[must_use]
    pub const fn topic(&self) -> SubscriptionTopic {
        match self {
            Self::EventLog(_) => SubscriptionTopic::EventLog,
            Self::SessionEvents(_) => SubscriptionTopic::SessionEvents,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Unsubscribe {
    pub subscription_id: SubscriptionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Subscribed {
    pub topic: SubscriptionTopic,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EventLogFilter {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_event_id: Option<EventId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionEventsFilter {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<SessionEventCursor>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EventLogEvent {
    pub event_id: EventId,
    pub occurred_at: Timestamp,
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

/// Server-pushed event for an active subscription.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Event {
    pub subscription_id: SubscriptionId,
    pub event: SubscriptionEvent,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum SubscriptionEvent {
    Subscribed(Subscribed),
    EventLog(EventLogEvent),
    SessionEvent(SessionEvent),
    SessionLiveEvent(SessionLiveEvent),
    Error(ErrorEnvelope),
}

/// Client ↔ control plane protocol messages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ClientMessage {
    Request(Request),
    Response(Response),
    Subscribe(Subscribe),
    Event(Event),
    Unsubscribe(Unsubscribe),
}
