//! Typed command kinds + payloads for agent execution (T-41).
//!
//! These payloads are carried in the daemon command dispatch `json_payload` and must remain:
//! - repo-scoped (no filesystem paths),
//! - stable (ULID ids + small primitives),
//! - forward-compatible via `serde` defaults.

use redesmyn_ids::{SessionId, TaskId};

use crate::client::{AgentInterfaceMode, AgentKind};
use crate::session::ExternalSessionRef;

pub const TASK_AGENT_START: &str = "task.agent.start";
pub const TASK_AGENT_STOP: &str = "task.agent.stop";
pub const TASK_AGENT_RESTART: &str = "task.agent.restart";
pub const TASK_AGENT_INTERRUPT_TURN: &str = "task.agent.interrupt_turn";
pub const TASK_AGENT_SEND_MESSAGE: &str = "task.agent.send_message";
pub const TASK_AGENT_RESUME_BY_ID_TURN: &str = "task.agent.resume_by_id_turn";
pub const TASK_AGENT_ATTACH_SESSION: &str = "task.agent.attach_session";

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
    pub prompt: String,
    pub external_session_ref: ExternalSessionRef,
    #[serde(default)]
    pub interrupt_turn: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AttachTaskAgentSessionCommand {
    pub session_id: SessionId,
}
