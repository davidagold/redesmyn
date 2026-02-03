//! Typed command kinds + payloads for agent execution (T-41).
//!
//! These payloads are carried in the daemon command dispatch `json_payload` and must remain:
//! - repo-scoped (no filesystem paths),
//! - stable (ULID ids + small primitives),
//! - forward-compatible via `serde` defaults.

use redesmyn_ids::{SessionId, TaskId};

use crate::client::{AgentInterfaceMode, AgentKind};
use crate::session::{ExternalSessionRef, PermissionDecision, PermissionsMode};

pub const SESSION_AGENT_START: &str = "session.agent.start";
pub const SESSION_AGENT_STOP: &str = "session.agent.stop";
pub const SESSION_AGENT_INTERRUPT_TURN: &str = "session.agent.interrupt_turn";
pub const SESSION_AGENT_SEND_MESSAGE: &str = "session.agent.send_message";
pub const SESSION_AGENT_RESUME_BY_ID_TURN: &str = "session.agent.resume_by_id_turn";
pub const SESSION_AGENT_ATTACH_SESSION: &str = "session.agent.attach_session";
pub const SESSION_AGENT_SET_PERMISSIONS_MODE: &str = "session.agent.set_permissions_mode";
pub const SESSION_AGENT_RESPOND_PERMISSION_REQUEST: &str = "session.agent.respond_permission_request";

pub const TASK_AGENT_START: &str = "task.agent.start";
pub const TASK_AGENT_STOP: &str = "task.agent.stop";
pub const TASK_AGENT_RESTART: &str = "task.agent.restart";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StartAgentSessionCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub agent_kind: AgentKind,
    pub interface_mode: AgentInterfaceMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
    /// Session ids to stop before starting this session.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_session_ids: Vec<SessionId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StartTaskAgentSessionCommand {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub agent_kind: AgentKind,
    pub interface_mode: AgentInterfaceMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_prompt: Option<String>,
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
pub struct SendTaskAgentMessageCommand {
    pub session_id: SessionId,
    pub text: String,
    #[serde(default)]
    pub interrupt_turn: bool,
    #[serde(default)]
    pub submit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResumeByIdTaskAgentTurnCommand {
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    pub prompt: String,
    pub external_session_ref: ExternalSessionRef,
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
    pub mode: PermissionsMode,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RespondPermissionRequestCommand {
    pub session_id: SessionId,
    pub request_id: String,
    pub decision: PermissionDecision,
}
