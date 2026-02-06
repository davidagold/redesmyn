//! Typed command kinds + payloads for agent execution (T-41).
//!
//! These payloads are carried in the daemon command dispatch `json_payload` and must remain:
//! - repo-scoped (no filesystem paths),
//! - stable (ULID ids + small primitives),
//! - forward-compatible via `serde` defaults.

use redesmyn_ids::{SessionId, TaskId};

use crate::client::{AgentKind, ModelReasoningEffort};
use crate::session::{
    CodexApprovalPolicy, CodexSandboxPolicy, ExternalSessionRef, ImageAttachment,
    PermissionDecision, PermissionsMode,
};

fn default_permissions_mode() -> PermissionsMode {
    PermissionsMode::Ask
}

/// Session-scoped policy overrides derived from durable session events.
///
/// This snapshot is passed along with `session.agent.start` / `session.agent.resume_by_id_turn`
/// so the daemon can hydrate a freshly started runner before sending the next turn.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionPolicySnapshot {
    #[serde(default = "default_permissions_mode")]
    pub permissions_mode: PermissionsMode,
    #[serde(default)]
    pub codex_approval_policy: Option<CodexApprovalPolicy>,
    #[serde(default)]
    pub codex_sandbox_policy: Option<CodexSandboxPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reasoning_effort: Option<ModelReasoningEffort>,
}

pub const SESSION_AGENT_START: &str = "session.agent.start";
pub const SESSION_AGENT_STOP: &str = "session.agent.stop";
pub const SESSION_AGENT_INTERRUPT_TURN: &str = "session.agent.interrupt_turn";
pub const SESSION_AGENT_RESUME_BY_ID_TURN: &str = "session.agent.resume_by_id_turn";
pub const SESSION_AGENT_ATTACH_SESSION: &str = "session.agent.attach_session";
pub const SESSION_AGENT_SET_PERMISSIONS_MODE: &str = "session.agent.set_permissions_mode";
pub const SESSION_AGENT_RESPOND_PERMISSION_REQUEST: &str =
    "session.agent.respond_permission_request";
pub const SESSION_AGENT_SET_CODEX_APPROVAL_POLICY: &str = "session.agent.set_codex_approval_policy";
pub const SESSION_AGENT_SET_CODEX_SANDBOX_POLICY: &str = "session.agent.set_codex_sandbox_policy";
pub const SESSION_AGENT_SET_MODEL: &str = "session.agent.set_model";
pub const SESSION_AGENT_LIST_MODELS: &str = "session.agent.list_models";
pub const AGENT_LIST_MODELS: &str = "agent.list_models";

pub const TASK_AGENT_START: &str = "task.agent.start";
pub const TASK_AGENT_STOP: &str = "task.agent.stop";
pub const TASK_AGENT_RESTART: &str = "task.agent.restart";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StartAgentSessionCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_branch_name: Option<String>,
    pub agent_kind: AgentKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_attachments: Vec<ImageAttachment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_snapshot: Option<SessionPolicySnapshot>,
    /// Session ids to stop before starting this session.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_session_ids: Vec<SessionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StartTaskAgentSessionCommand {
    pub session_id: SessionId,
    pub task_id: TaskId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_branch_name: Option<String>,
    pub agent_kind: AgentKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_snapshot: Option<SessionPolicySnapshot>,
    /// Session ids to stop before starting this new conversation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_session_ids: Vec<SessionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StopTaskAgentSessionCommand {
    pub task_id: TaskId,
    /// If empty, stop any active sessions for the task.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub session_ids: Vec<SessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InterruptTaskAgentTurnCommand {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResumeByIdTaskAgentTurnCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_attachments: Vec<ImageAttachment>,
    pub external_session_ref: ExternalSessionRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_snapshot: Option<SessionPolicySnapshot>,
    #[serde(default)]
    pub interrupt_turn: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AttachTaskAgentSessionCommand {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionPermissionsModeCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub mode: PermissionsMode,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionCodexApprovalPolicyCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_policy: Option<CodexApprovalPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionCodexSandboxPolicyCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_policy: Option<CodexSandboxPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSessionModelCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ModelReasoningEffort>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListSessionModelsCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ListAgentModelsCommand {
    pub agent_kind: AgentKind,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RespondPermissionRequestCommand {
    pub session_id: SessionId,
    pub request_id: String,
    pub decision: PermissionDecision,
}
