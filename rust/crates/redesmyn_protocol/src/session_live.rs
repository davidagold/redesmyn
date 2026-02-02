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
    AssistantReasoningSummaryPartAdded(AssistantReasoningSummaryPartAdded),
    AssistantReasoningSummaryDelta(AssistantReasoningSummaryDelta),
    AssistantReasoningRawDelta(AssistantReasoningRawDelta),
    ToolOutputDelta(ToolOutputDelta),
    Unknown(UnknownSessionLiveEvent),
}

/// Streaming delta for an assistant message.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantMessageDelta {
    pub delta: String,
}

/// Begin a new reasoning summary part at `summary_index`.
///
/// Some providers stream the reasoning summary in multiple discrete parts; this event is delivered
/// before the first delta for the corresponding part.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantReasoningSummaryPartAdded {
    #[serde(rename = "summary_index")]
    pub summary_index: i64,
}

/// Streaming delta for an assistant reasoning summary part.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantReasoningSummaryDelta {
    #[serde(rename = "summary_index")]
    pub summary_index: i64,
    pub delta: String,
}

/// Streaming delta for assistant reasoning raw content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssistantReasoningRawDelta {
    #[serde(rename = "content_index")]
    pub content_index: i64,
    pub delta: String,
}

/// Streaming delta for tool output (e.g. terminal output).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolOutputDelta {
    pub tool_name: String,
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
