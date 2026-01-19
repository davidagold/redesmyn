//! Protocol types for the daemon ↔ control plane boundary.
//!
//! This is intentionally small (Domain 0 scaffolding). Higher-level stream
//! semantics (handshake, multiplexing, resync) will be built on top in Domain 1.

use redesmyn_ids::{CommandId, HostId, MsgId, RunId};

/// Current daemon protocol version.
///
/// This is a coarse version gate for now; Domain 1 will introduce a richer
/// envelope and negotiation story.
pub const DAEMON_PROTOCOL_VERSION: u32 = 1;

/// Versioned message envelope carried by all daemon frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MessageEnvelope {
    pub protocol_version: u32,
    pub msg_id: MsgId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_reply_to: Option<MsgId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_id: Option<CommandId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<RunId>,
}

impl MessageEnvelope {
    #[must_use]
    pub fn new(msg_id: MsgId) -> Self {
        Self {
            protocol_version: DAEMON_PROTOCOL_VERSION,
            msg_id,
            in_reply_to: None,
            command_id: None,
            run_id: None,
        }
    }

    #[must_use]
    pub fn reply(msg_id: MsgId, request: &MessageEnvelope) -> Self {
        Self {
            protocol_version: DAEMON_PROTOCOL_VERSION,
            msg_id,
            in_reply_to: Some(request.msg_id),
            command_id: request.command_id,
            run_id: request.run_id,
        }
    }
}

/// A single daemon protocol frame.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DaemonFrame {
    pub envelope: MessageEnvelope,
    pub message: DaemonMessage,
}

impl DaemonFrame {
    #[must_use]
    pub fn new(envelope: MessageEnvelope, message: DaemonMessage) -> Self {
        Self { envelope, message }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HelloRequest {
    pub client_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HelloResponse {
    pub daemon_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Heartbeat {
    pub host_id: HostId,
}

/// Minimal, typed command placeholder for Domain 0.
///
/// Domain 2 introduces the real command model and routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonCommand {
    Noop,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DispatchCommand {
    pub command: DaemonCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandAckStatus {
    Accepted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommandAck {
    pub status: CommandAckStatus,
}

/// Minimal event placeholder for Domain 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonEvent {
    Noop,
}

/// Daemon ↔ control plane protocol messages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum DaemonMessage {
    HelloRequest(HelloRequest),
    HelloResponse(HelloResponse),
    Heartbeat(Heartbeat),
    DispatchCommand(DispatchCommand),
    CommandAck(CommandAck),
    Event(DaemonEvent),
}
