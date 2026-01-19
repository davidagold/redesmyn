//! Protocol types for the client ↔ control plane API boundary (T-12).
//!
//! This module defines the canonical typed message model used by:
//! - out-of-proc clients over UDS/TCP (framed + Protobuf by default), and
//! - embedded in-proc clients (typed messages; optional codec loopback in tests).

use redesmyn_ids::{EventId, RequestId, SubscriptionId};

use crate::{ErrorEnvelope, ProtocolEnvelope, ProtocolVersion, Timestamp};

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
}

impl RequestPayload {
    #[must_use]
    pub const fn method(&self) -> ClientMethod {
        match self {
            Self::Health(_) => ClientMethod::Health,
            Self::Status(_) => ClientMethod::Status,
            Self::ListEpics(_) => ClientMethod::ListEpics,
            Self::GetEpicGraph(_) => ClientMethod::GetEpicGraph,
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EpicTaskNode {
    pub task_slug: String,
    pub title: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EpicTaskEdge {
    pub from_task_slug: String,
    pub to_task_slug: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EpicGraph {
    pub epic_slug: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<EpicTaskNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edges: Vec<EpicTaskEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetEpicGraphResponse {
    pub graph: EpicGraph,
}

/// Subscription topics for server-pushed streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionTopic {
    EventLog,
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
}

impl SubscriptionFilter {
    #[must_use]
    pub const fn topic(&self) -> SubscriptionTopic {
        match self {
            Self::EventLog(_) => SubscriptionTopic::EventLog,
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
pub enum SubscriptionEvent {
    Subscribed(Subscribed),
    EventLog(EventLogEvent),
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
