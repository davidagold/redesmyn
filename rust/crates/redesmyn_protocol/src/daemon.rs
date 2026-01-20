//! Protocol types for the daemon ↔ control plane boundary (T-11).
//!
//! The canonical schema lives in `rust/proto/daemon.proto` and generates the
//! Protobuf bindings in this crate (see `crate::pb`).

use redesmyn_ids::{CommandId, HostId, HostInstanceId, MsgId};

use crate::{ErrorDetail, ErrorEnvelope, ProtocolEnvelope, ProtocolVersion, RepoScope};

/// A single daemon ↔ control plane protocol frame (T-11).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DaemonFrame {
    pub envelope: ProtocolEnvelope,
    pub message: DaemonMessage,
}

impl DaemonFrame {
    #[must_use]
    pub fn new(envelope: ProtocolEnvelope, message: DaemonMessage) -> Self {
        Self { envelope, message }
    }
}

/// Daemon ↔ control plane protocol messages (T-11).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum DaemonMessage {
    DaemonHello(DaemonHello),
    ControlPlaneHelloAck(ControlPlaneHelloAck),
    RepoAttach(RepoAttach),
    RepoDetach(RepoDetach),
    DaemonHeartbeat(DaemonHeartbeat),
    TelemetryEventBatch(TelemetryEventBatch),
    TelemetrySnapshot(TelemetrySnapshot),
    ResyncRequest(ResyncRequest),
    CommandDispatch(CommandDispatch),
    CommandUpdate(CommandUpdate),
    SessionEventBatch(SessionEventBatch),
    Error(ErrorEnvelope),
}

/// Daemon → control plane handshake payload (T-11).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DaemonHello {
    pub host_id: HostId,
    pub host_instance_id: HostInstanceId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    pub supported_protocol: ProtocolVersion,
}

/// Handshake response: control plane → daemon (T-11).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ControlPlaneHelloAck {
    pub accepted_protocol: ProtocolVersion,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RepoAttach {
    pub repo_scope: RepoScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_root_hint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RepoDetach {
    pub repo_scope: RepoScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TelemetryFreshness {
    pub scope: RepoScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_event_batch_msg_id: Option<MsgId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_snapshot_msg_id: Option<MsgId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DaemonHeartbeat {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attached_repo_scopes: Vec<RepoScope>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub telemetry_freshness: Vec<TelemetryFreshness>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum TelemetryEvent {
    Git(GitEvent),
    Worktree(WorktreeEvent),
    Agent(AgentEvent),
    MergeRun(MergeRunEvent),
    Unknown(UnknownEvent),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GitEvent {
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorktreeEvent {
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AgentEvent {
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MergeRunEvent {
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnknownEvent {
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TelemetryEventBatch {
    pub scope: RepoScope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<TelemetryEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TelemetrySnapshot {
    pub scope: RepoScope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResyncRequest {
    pub scope: RepoScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommandDispatch {
    pub command_id: CommandId,
    pub scope: RepoScope,
    pub command_kind: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandState {
    Accepted,
    Running,
    Succeeded,
    Failed,
    Canceled,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommandProgress {
    pub percent: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommandUpdate {
    pub command_id: CommandId,
    pub state: CommandState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<CommandProgress>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ErrorDetail>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorEnvelope>,
}

/// Batch of session events emitted by the daemon (T-35/T-14).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionEventBatch {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<crate::session::SessionEvent>,
}

/// Typed daemon capabilities advertised during handshake.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DaemonCapabilities {
    pub supports_repo_execution: bool,
    pub supports_worktrees: bool,
    pub supports_git_observation: bool,
    pub supports_session_exec: bool,
    pub supports_session_attach_tmux: bool,
    pub supports_artifacts: bool,
}

impl DaemonCapabilities {
    #[must_use]
    pub fn to_wire_strings(self) -> Vec<String> {
        let mut out = Vec::new();
        if self.supports_repo_execution {
            out.push("supports_repo_execution".to_string());
        }
        if self.supports_worktrees {
            out.push("supports_worktrees".to_string());
        }
        if self.supports_git_observation {
            out.push("supports_git_observation".to_string());
        }
        if self.supports_session_exec {
            out.push("supports_session_exec".to_string());
        }
        if self.supports_session_attach_tmux {
            out.push("supports_session_attach_tmux".to_string());
        }
        if self.supports_artifacts {
            out.push("supports_artifacts".to_string());
        }
        out
    }

    #[must_use]
    pub fn from_wire_strings<'a>(capabilities: impl IntoIterator<Item = &'a str>) -> Self {
        let mut out = Self::default();
        for cap in capabilities {
            match cap {
                "supports_repo_execution" => out.supports_repo_execution = true,
                "supports_worktrees" => out.supports_worktrees = true,
                "supports_git_observation" => out.supports_git_observation = true,
                "supports_session_exec" => out.supports_session_exec = true,
                "supports_session_attach_tmux" => out.supports_session_attach_tmux = true,
                "supports_artifacts" => out.supports_artifacts = true,
                _ => {}
            }
        }
        out
    }
}
