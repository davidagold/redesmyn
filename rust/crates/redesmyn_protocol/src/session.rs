//! Structured session events for durable session viewer UX (T-14).
//!
//! Session semantics: session == conversation; turns are events.

use redesmyn_ids::{SessionEventId, SessionId, TaskId};

use crate::artifacts::ArtifactRef;
use crate::{ErrorEnvelope, Timestamp};

/// Session scope for querying: task-scoped sessions and user-managed chats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionScope {
    Task {
        task_id: TaskId,
    },
    /// User-managed chat session (pinned-to-epic relationships live elsewhere).
    Chat,
    /// A scope kind not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// Session interface mode (structured vs interactive/tmux).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterfaceMode {
    Interactive,
    Structured,
    /// A mode not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// Best-effort external session handle for structured agents.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExternalSessionRef {
    None,
    CodexThread {
        thread_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
    },
    #[serde(alias = "codex_conversation")]
    CodexSession {
        #[serde(alias = "conversation_id")]
        session_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
    },
    ClaudeSession {
        session_id: String,
    },
    /// Placeholder for future providers / ref types.
    Unknown {
        unknown_type: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        json_payload: Vec<u8>,
    },
}

/// Forward-compatible placeholder for session event union fallbacks (T-14).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnknownSessionEvent {
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionStarted {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionEnded {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TurnStarted {
    pub interface_mode: InterfaceMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_session_ref: Option<ExternalSessionRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_offset_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TurnCompleted {
    pub interface_mode: InterfaceMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_session_ref: Option<ExternalSessionRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorEnvelope>,
}

/// User message text is bounded; use `full_text_artifact` for large content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UserMessage {
    pub text: String,
    pub preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_text_artifact: Option<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_attachments: Vec<ImageAttachment>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImageAttachment {
    pub artifact: ArtifactRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Assistant message text is bounded; use `full_text_artifact` for large content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantMessage {
    pub text: String,
    pub preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_text_artifact: Option<ArtifactRef>,
}

/// Assistant reasoning is bounded; use artifacts for large content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantReasoningText {
    pub text: String,
    pub preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_text_artifact: Option<ArtifactRef>,
}

/// Structured assistant reasoning (summary + optional raw content).
///
/// Notes:
/// - `item_id` is a provider-scoped identifier (e.g. Codex app-server item id) used to correlate
///   streaming deltas with the durable event.
/// - `signature` is provider-specific and may be used to attest to the reasoning content (e.g.
///   Claude extended thinking).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantReasoning {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    pub summary: AssistantReasoningText,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<AssistantReasoningText>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// Tool invocation parameters are bounded; use artifacts for large inputs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolInvocation {
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub input_preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_artifact: Option<ArtifactRef>,
}

/// Tool results are bounded; use artifacts for large outputs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub output_preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_artifact: Option<ArtifactRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorEnvelope>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnState {
    Running,
    Blocked,
    Completed,
    /// A state not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// Status updates are compact; do not embed full logs or diffs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StatusUpdate {
    pub turn_state: TurnState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocking: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_percent: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskAgentMessageAgentKind {
    Codex,
    ClaudeCode,
    Shell,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskAgentMessageDelivery {
    StructuredStarted,
    StructuredResumed,
    InteractiveStarted,
    InteractiveSent,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskAgentMessageConversationContinuity {
    Kept,
    Broken,
    #[serde(other)]
    Unknown,
}

/// Durable metadata for task-agent message sends (intent + delivery context).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TaskAgentMessageSent {
    pub intent: String,
    pub message: String,
    pub message_preview: String,
    pub agent_kind: TaskAgentMessageAgentKind,
    pub delivery: TaskAgentMessageDelivery,
    pub conversation_continuity: TaskAgentMessageConversationContinuity,
}

/// Codex app-server approval policy (provider-native).
///
/// Mirrors the Codex `AskForApproval` enum (`"untrusted"`, `"on-request"`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CodexApprovalPolicy {
    #[serde(rename = "untrusted")]
    UnlessTrusted,
    OnFailure,
    OnRequest,
    Never,
    /// A policy not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// A change in the session's effective Codex approval policy override.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CodexApprovalPolicyChanged {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<CodexApprovalPolicy>,
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "camelCase")]
pub enum CodexNetworkAccess {
    #[default]
    Restricted,
    Enabled,
    /// A value not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

fn is_default_network_access(value: &CodexNetworkAccess) -> bool {
    *value == CodexNetworkAccess::Restricted
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Codex app-server sandbox policy (provider-native).
///
/// Mirrors the Codex `SandboxPolicy` union from the app-server protocol.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum CodexSandboxPolicy {
    DangerFullAccess,
    ReadOnly,
    #[serde(rename_all = "camelCase")]
    ExternalSandbox {
        #[serde(default, skip_serializing_if = "is_default_network_access")]
        network_access: CodexNetworkAccess,
    },
    #[serde(rename_all = "camelCase")]
    WorkspaceWrite {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        writable_roots: Vec<String>,
        #[serde(default, skip_serializing_if = "is_false")]
        network_access: bool,
        #[serde(default, skip_serializing_if = "is_false")]
        exclude_tmpdir_env_var: bool,
        #[serde(default, skip_serializing_if = "is_false")]
        exclude_slash_tmp: bool,
    },
    /// A policy not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// A change in the session's effective Codex sandbox policy override.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CodexSandboxPolicyChanged {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_policy: Option<CodexSandboxPolicy>,
}

/// Reasoning effort associated with a session model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionModelReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    /// A value not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// A change in the session's effective model + reasoning selection.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionModelChanged {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<SessionModelReasoningEffort>,
}

/// How the session handles permission/approval requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionsMode {
    /// Surface requests to the user and wait for a decision.
    Ask,
    /// Automatically approve requests.
    AutoApprove,
    /// Automatically deny requests.
    Deny,
    /// A mode not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// A change in the session's effective permissions mode.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PermissionsModeChanged {
    pub mode: PermissionsMode,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommandExecutionPermissionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileChangePermissionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum PermissionRequest {
    CommandExecution(CommandExecutionPermissionRequest),
    FileChange(FileChangePermissionRequest),
    /// Placeholder for future providers / request shapes.
    Unknown {
        unknown_kind: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        json_payload: Vec<u8>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Approve,
    Deny,
    /// A decision not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecisionBy {
    User,
    ModeAutoApprove,
    ModeAutoDeny,
    Timeout,
    /// A value not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PermissionRequested {
    /// A stable identifier for correlating request/decision events.
    pub request_id: String,
    /// A compact, user-facing summary of what is being approved.
    pub summary: String,
    pub request: PermissionRequest,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PermissionDecided {
    pub request_id: String,
    pub decision: PermissionDecision,
    pub decided_by: PermissionDecisionBy,
}

/// Tie an out-of-band artifact to a session and (optionally) a turn.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactEmitted {
    pub artifact: ArtifactRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Structured session event union.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SessionEventKind {
    SessionStarted(SessionStarted),
    SessionEnded(SessionEnded),
    TurnStarted(TurnStarted),
    TurnCompleted(TurnCompleted),
    UserMessage(UserMessage),
    AssistantMessage(AssistantMessage),
    AssistantReasoning(AssistantReasoning),
    ToolInvocation(ToolInvocation),
    ToolResult(ToolResult),
    StatusUpdate(StatusUpdate),
    TaskAgentMessageSent(TaskAgentMessageSent),
    PermissionsModeChanged(PermissionsModeChanged),
    CodexApprovalPolicyChanged(CodexApprovalPolicyChanged),
    CodexSandboxPolicyChanged(CodexSandboxPolicyChanged),
    SessionModelChanged(SessionModelChanged),
    PermissionRequested(PermissionRequested),
    PermissionDecided(PermissionDecided),
    ArtifactEmitted(ArtifactEmitted),
    Unknown(UnknownSessionEvent),
}

/// Durable session event record.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionEvent {
    pub session_event_id: SessionEventId,
    pub created_at: Timestamp,
    pub scope: SessionScope,
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(flatten)]
    pub kind: SessionEventKind,
}
