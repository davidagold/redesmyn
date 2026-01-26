//! Live-only session events.
//!
//! These events are delivered best-effort to active subscribers but are NOT persisted.

use redesmyn_ids::SessionId;

use crate::{Timestamp, session::UnknownSessionEvent};

/// Non-durable session event union for live UX affordances (e.g. streaming deltas).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SessionLiveEventKind {
    AssistantMessageDelta(AssistantMessageDelta),
    Unknown(UnknownSessionLiveEvent),
}

/// Streaming delta for an assistant message.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantMessageDelta {
    pub delta: String,
}

/// Forward-compatible placeholder for live-event union fallbacks.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnknownSessionLiveEvent {
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub json_payload: Vec<u8>,
}

impl From<UnknownSessionEvent> for UnknownSessionLiveEvent {
    fn from(value: UnknownSessionEvent) -> Self {
        Self {
            event_type: value.event_type,
            json_payload: value.json_payload,
        }
    }
}

/// Live-only session event record.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionLiveEvent {
    pub created_at: Timestamp,
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(flatten)]
    pub kind: SessionLiveEventKind,
}

