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
    Task { task_id: TaskId },
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

/// Best-effort external conversation handle for structured agents.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExternalSessionRef {
    None,
    CodexThread {
        thread_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn_id: Option<String>,
    },
    ClaudeSession { session_id: String },
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
}

/// Assistant message text is bounded; use `full_text_artifact` for large content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantMessage {
    pub text: String,
    pub preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_text_artifact: Option<ArtifactRef>,
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
    ToolInvocation(ToolInvocation),
    ToolResult(ToolResult),
    StatusUpdate(StatusUpdate),
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

