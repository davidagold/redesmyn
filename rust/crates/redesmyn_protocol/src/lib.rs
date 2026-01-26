//! Versioned, strongly-typed message schemas for Redesmyn boundaries.
//!
//! This crate contains wire types shared across the Rust workspace, including the
//! cross-boundary error envelope used by:
//! - daemon ↔ control plane (protocol errors),
//! - client ↔ control plane APIs,
//! - CLI formatting (exit codes + user-facing messages).

use std::collections::BTreeMap;
use std::{fmt, str::FromStr};

use redesmyn_ids::{MsgId, RepoId, WorkspaceId};

#[doc(hidden)]
pub mod pb {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/redesmyn_protocol_pb.rs"));
}

pub use redesmyn_errors::ErrorCategory;

pub mod agent_commands;
pub mod artifacts;
pub mod client;
pub mod daemon;
pub mod session;
pub mod session_live;
pub mod ui_driver;

pub use artifacts::{ArtifactKind, ArtifactRef, Hash, StorageHint};
pub use session::{
    ArtifactEmitted, AssistantMessage, ExternalSessionRef, InterfaceMode, SessionEnded,
    SessionEvent, SessionEventKind, SessionScope, SessionStarted, StatusUpdate, ToolInvocation,
    ToolResult, TurnCompleted, TurnStarted, TurnState, UnknownSessionEvent, UserMessage,
};
pub use session_live::{
    AssistantMessageDelta, SessionLiveEvent, SessionLiveEventKind, UnknownSessionLiveEvent,
};

pub use daemon::DaemonHello;

/// Optional structured detail for debugging/UX (no stack traces).
///
/// This is intentionally simple for now (stringly-typed map) and can evolve as
/// richer typed detail payloads become needed.
pub type ErrorDetail = BTreeMap<String, String>;

/// Minimal structured error type that can cross boundaries (protocol/API).
///
/// Conventions:
/// - `message` must be user-actionable and non-noisy.
/// - `detail` should avoid stack traces and large payloads.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ErrorEnvelope {
    pub category: ErrorCategory,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ErrorDetail>,
}

impl ErrorEnvelope {
    pub fn new(category: ErrorCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
            detail: None,
        }
    }

    pub fn with_detail(mut self, detail: ErrorDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    pub fn exit_code(&self) -> i32 {
        self.category.exit_code()
    }

    pub fn http_status(&self) -> u16 {
        self.category.http_status()
    }
}

/// Canonical, stable protocol major version.
///
/// Rules:
/// - major mismatch => reject (structured error + close connection).
/// - minor mismatch => accept if possible (additive-only; ignore unknown fields).
pub const PROTOCOL_MAJOR: u16 = 1;

/// Canonical, stable protocol minor version.
pub const PROTOCOL_MINOR: u16 = 0;

/// A concrete protocol version used during connection establishment (hello/handshake).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const CURRENT: Self = Self {
        major: PROTOCOL_MAJOR,
        minor: PROTOCOL_MINOR,
    };

    #[must_use]
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Negotiate an accepted protocol version with a peer.
    ///
    /// This enforces the major-version compatibility rule (reject on mismatch) and
    /// selects the highest common minor version by choosing `min(self.minor, peer.minor)`.
    pub fn negotiate(self, peer: Self) -> Result<Self, ErrorEnvelope> {
        if self.major != peer.major {
            let detail = ErrorDetail::from([
                ("local_major".to_string(), self.major.to_string()),
                ("local_minor".to_string(), self.minor.to_string()),
                ("peer_major".to_string(), peer.major.to_string()),
                ("peer_minor".to_string(), peer.minor.to_string()),
            ]);
            return Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Protocol major version mismatch.",
            )
            .with_detail(detail));
        }

        Ok(Self {
            major: self.major,
            minor: self.minor.min(peer.minor),
        })
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// A wall-clock timestamp encoded as RFC3339 for JSON diagnostics.
///
/// Notes:
/// - `sent_at` is informational; monotonic ordering is not assumed.
/// - Ordering-sensitive flows must use explicit sequence numbers in payloads.
///
/// Encoding rules:
/// - JSON: RFC3339 string
/// - Protobuf: `google.protobuf.Timestamp` (or equivalent)
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Timestamp(#[serde(with = "time::serde::rfc3339")] time::OffsetDateTime);

impl Timestamp {
    #[must_use]
    pub fn now_utc() -> Self {
        Self(time::OffsetDateTime::now_utc())
    }

    pub fn from_unix_millis(ms: i64) -> Result<Self, time::error::ComponentRange> {
        let nanos = i128::from(ms).saturating_mul(1_000_000);
        Ok(Self(time::OffsetDateTime::from_unix_timestamp_nanos(
            nanos,
        )?))
    }

    #[must_use]
    pub const fn from_offset_date_time(value: time::OffsetDateTime) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn into_offset_date_time(self) -> time::OffsetDateTime {
        self.0
    }
}

impl From<time::OffsetDateTime> for Timestamp {
    fn from(value: time::OffsetDateTime) -> Self {
        Self::from_offset_date_time(value)
    }
}

impl From<Timestamp> for time::OffsetDateTime {
    fn from(value: Timestamp) -> Self {
        value.into_offset_date_time()
    }
}

/// Opaque distributed tracing identifier (16 bytes).
///
/// Encoding rules:
/// - JSON: 32-char lowercase hex string
/// - Protobuf: 16 raw bytes
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TraceId([u8; 16]);

impl TraceId {
    pub const BYTE_LEN: usize = 16;

    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn to_bytes(self) -> [u8; 16] {
        self.0
    }

    pub fn try_from_bytes_slice(bytes: &[u8]) -> Result<Self, TraceIdBytesLengthError> {
        if bytes.len() != Self::BYTE_LEN {
            return Err(TraceIdBytesLengthError::new(bytes.len()));
        }

        let mut array = [0_u8; Self::BYTE_LEN];
        array.copy_from_slice(bytes);
        Ok(Self(array))
    }
}

impl fmt::Debug for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TraceId({self})")
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// An error produced when parsing a trace id from a hex string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseTraceIdError {
    input: String,
}

impl ParseTraceIdError {
    fn new(input: &str) -> Self {
        Self {
            input: input.to_owned(),
        }
    }
}

impl fmt::Display for ParseTraceIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid trace id: {}", self.input)
    }
}

impl std::error::Error for ParseTraceIdError {}

/// An error produced when decoding a trace id from a byte slice of the wrong length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceIdBytesLengthError {
    actual_len: usize,
}

impl TraceIdBytesLengthError {
    fn new(actual_len: usize) -> Self {
        Self { actual_len }
    }

    #[must_use]
    pub const fn actual_len(&self) -> usize {
        self.actual_len
    }
}

impl fmt::Display for TraceIdBytesLengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid trace id bytes length: expected {}, got {}",
            TraceId::BYTE_LEN,
            self.actual_len
        )
    }
}

impl std::error::Error for TraceIdBytesLengthError {}

impl FromStr for TraceId {
    type Err = ParseTraceIdError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 32 {
            return Err(ParseTraceIdError::new(s));
        }

        let mut bytes = [0_u8; 16];
        for (i, chunk) in s.as_bytes().chunks_exact(2).enumerate() {
            let hi = (chunk[0] as char)
                .to_digit(16)
                .ok_or_else(|| ParseTraceIdError::new(s))?;
            let lo = (chunk[1] as char)
                .to_digit(16)
                .ok_or_else(|| ParseTraceIdError::new(s))?;
            bytes[i] = ((hi << 4) | lo) as u8;
        }
        Ok(Self(bytes))
    }
}

impl serde::Serialize for TraceId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> serde::Deserialize<'de> for TraceId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = <std::borrow::Cow<'de, str> as serde::Deserialize<'de>>::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Repo-scoped routing key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RepoScope {
    pub workspace_id: WorkspaceId,
    pub repo_id: RepoId,
}

impl RepoScope {
    #[must_use]
    pub const fn new(workspace_id: WorkspaceId, repo_id: RepoId) -> Self {
        Self {
            workspace_id,
            repo_id,
        }
    }
}

/// Message routing scope (future-proof; additional variants can be added).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Scope {
    /// Repo-scoped message.
    Repo {
        #[serde(flatten)]
        repo: RepoScope,
    },
    /// A scope kind not understood by this binary.
    ///
    /// This exists for forward compatibility (older binaries can still parse the message).
    /// Callers should treat this as an incompatibility for routing decisions.
    #[serde(other)]
    Unknown,
}

impl From<RepoScope> for Scope {
    fn from(value: RepoScope) -> Self {
        Self::Repo { repo: value }
    }
}

/// Canonical protocol envelope for all cross-boundary messages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProtocolEnvelope {
    /// Protocol major version (incompatible changes).
    pub protocol_major: u16,
    /// Protocol minor version (additive-only changes).
    pub protocol_minor: u16,
    /// Globally unique message identifier (idempotency + dedupe key).
    pub msg_id: MsgId,
    /// Informational wall-clock timestamp (monotonic ordering is not assumed).
    pub sent_at: Timestamp,
    /// Optional routing scope for the message.
    ///
    /// Repo-scoped messages should use `Some(Scope::Repo { ... })`. Messages without a routing
    /// scope can omit the field (`None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    /// Optional request/response correlation id.
    ///
    /// This correlates protocol-level responses/acks to initiating messages and is not a
    /// substitute for domain ids (e.g., `command_id`, `run_id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<MsgId>,
    /// Optional distributed trace id (carried across hops).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<TraceId>,
}

impl ProtocolEnvelope {
    #[must_use]
    pub fn new() -> Self {
        Self {
            protocol_major: PROTOCOL_MAJOR,
            protocol_minor: PROTOCOL_MINOR,
            msg_id: MsgId::new(),
            sent_at: Timestamp::now_utc(),
            scope: None,
            correlation_id: None,
            trace_id: None,
        }
    }

    #[must_use]
    pub const fn protocol_version(&self) -> ProtocolVersion {
        ProtocolVersion {
            major: self.protocol_major,
            minor: self.protocol_minor,
        }
    }

    #[must_use]
    pub fn with_scope(mut self, scope: Scope) -> Self {
        self.scope = Some(scope);
        self
    }

    #[must_use]
    pub fn with_correlation_id(mut self, correlation_id: MsgId) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    #[must_use]
    pub fn with_trace_id(mut self, trace_id: TraceId) -> Self {
        self.trace_id = Some(trace_id);
        self
    }
}

impl Default for ProtocolEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

mod protobuf;

#[cfg(test)]
mod tests {
    use super::{
        DaemonHello, ErrorCategory, ErrorEnvelope, MsgId, PROTOCOL_MAJOR, PROTOCOL_MINOR,
        ProtocolEnvelope, ProtocolVersion, RepoScope, Scope, Timestamp, TraceId,
    };

    use redesmyn_ids::{
        CommandId, EpicId, EventId, HostId, HostInstanceId, RepoId, RequestId, SubscriptionId,
        WorkspaceId,
    };

    use prost::Message;

    use crate::client::{
        ClientFrame, ClientMessage, EpicGraph, EpicSummary, EpicTaskEdge, EpicTaskNode, Event,
        EventLogEvent, EventLogFilter, GetEpicGraphRequest, GetEpicGraphResponse, HealthRequest,
        HealthResponse, ListEpicsRequest, ListEpicsResponse, Request, RequestPayload, Response,
        ResponseResult, Subscribe, Subscribed, SubscriptionEvent, SubscriptionFilter,
        SubscriptionTopic, Unsubscribe,
    };
    use crate::daemon::{
        CommandProgress, CommandState, CommandUpdate, DaemonCapabilities, DaemonFrame,
        DaemonMessage, GitEvent, TelemetryEvent, TelemetryEventBatch,
    };
    use crate::pb::redesmyn::protocol::v1 as pbv1;

    #[test]
    fn error_envelope_serializes_with_expected_shape() {
        let envelope = ErrorEnvelope::new(ErrorCategory::NotFound, "Task not found");
        let json = serde_json::to_string(&envelope).unwrap();
        assert_eq!(
            json,
            r#"{"category":"not_found","message":"Task not found"}"#
        );
    }

    #[test]
    fn protocol_version_rejects_major_mismatch() {
        let err = ProtocolVersion::new(1, 0)
            .negotiate(ProtocolVersion::new(2, 0))
            .unwrap_err();
        assert_eq!(err.category, ErrorCategory::InvalidRequest);
    }

    #[test]
    fn trace_id_hex_roundtrips() {
        let trace_id = TraceId::from_bytes([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ]);
        let s = trace_id.to_string();
        assert_eq!(s, "000102030405060708090a0b0c0d0e0f");
        let parsed: TraceId = s.parse().unwrap();
        assert_eq!(parsed, trace_id);

        let json = serde_json::to_string(&trace_id).unwrap();
        assert_eq!(json, r#""000102030405060708090a0b0c0d0e0f""#);
        let decoded: TraceId = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, trace_id);
    }

    #[test]
    fn timestamp_serializes_as_rfc3339_string() {
        let ts = Timestamp::from(time::OffsetDateTime::from_unix_timestamp(0).unwrap());
        let json = serde_json::to_string(&ts).unwrap();
        assert!(
            json.starts_with('\"') && json.ends_with('\"'),
            "json={json}"
        );
        let decoded: Timestamp = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ts);
    }

    #[test]
    fn protocol_envelope_omits_empty_optional_fields() {
        let msg_id: MsgId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
        let sent_at: Timestamp = serde_json::from_str(r#""2026-01-19T00:00:00Z""#).unwrap();
        let envelope = ProtocolEnvelope {
            protocol_major: PROTOCOL_MAJOR,
            protocol_minor: PROTOCOL_MINOR,
            msg_id,
            sent_at,
            scope: None,
            correlation_id: None,
            trace_id: None,
        };

        let json = serde_json::to_string(&envelope).unwrap();
        assert_eq!(
            json,
            r#"{"protocol_major":1,"protocol_minor":0,"msg_id":"01ARZ3NDEKTSV4RRFFQ69G5FAV","sent_at":"2026-01-19T00:00:00Z"}"#
        );
    }

    #[test]
    fn daemon_capabilities_accepts_legacy_and_canonical_wire_strings() {
        let daemon_emitted = DaemonCapabilities::from_wire_strings([
            "repo_execution",
            "worktrees",
            "git_observation",
            "session_exec",
            "session_attach_tmux",
            "artifacts",
        ]);

        assert!(daemon_emitted.supports_repo_execution);
        assert!(daemon_emitted.supports_worktrees);
        assert!(daemon_emitted.supports_git_observation);
        assert!(daemon_emitted.supports_session_exec);
        assert!(daemon_emitted.supports_session_attach_tmux);
        assert!(daemon_emitted.supports_artifacts);

        let wire = daemon_emitted.to_wire_strings();
        assert!(wire.contains(&"repo_execution".to_string()));
        assert!(!wire.contains(&"supports_repo_execution".to_string()));

        let round_tripped = DaemonCapabilities::from_wire_strings(wire.iter().map(String::as_str));
        assert_eq!(round_tripped, daemon_emitted);

        let legacy = DaemonCapabilities::from_wire_strings([
            "supports_repo_execution",
            "supports_worktrees",
            "supports_git_observation",
            "supports_session_exec",
            "supports_session_attach_tmux",
            "supports_artifacts",
        ]);
        assert_eq!(legacy, daemon_emitted);
    }

    #[test]
    fn protocol_envelope_includes_repo_scope_and_optional_ids() {
        let msg_id: MsgId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
        let correlation_id: MsgId = "01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap();
        let sent_at: Timestamp = serde_json::from_str(r#""2026-01-19T00:00:00Z""#).unwrap();
        let trace_id = TraceId::from_bytes([
            0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef, 0xde, 0xad,
            0xbe, 0xef,
        ]);
        let repo_scope = RepoScope::new(
            "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap(),
            "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap(),
        );

        let envelope = ProtocolEnvelope {
            protocol_major: PROTOCOL_MAJOR,
            protocol_minor: PROTOCOL_MINOR,
            msg_id,
            sent_at,
            scope: Some(Scope::Repo { repo: repo_scope }),
            correlation_id: Some(correlation_id),
            trace_id: Some(trace_id),
        };

        let json = serde_json::to_string(&envelope).unwrap();
        assert_eq!(
            json,
            r#"{"protocol_major":1,"protocol_minor":0,"msg_id":"01ARZ3NDEKTSV4RRFFQ69G5FAV","sent_at":"2026-01-19T00:00:00Z","scope":{"type":"repo","workspace_id":"01ARZ3NDEKTSV4RRFFQ69G5FAV","repo_id":"01ARZ3NDEKTSV4RRFFQ69G5FAV"},"correlation_id":"01ARZ3NDEKTSV4RRFFQ69G5FAW","trace_id":"deadbeefdeadbeefdeadbeefdeadbeef"}"#
        );
    }

    #[test]
    fn scope_deserializes_unknown_variant() {
        let scope: Scope = serde_json::from_str(
            r#"{"type":"workspace","workspace_id":"01ARZ3NDEKTSV4RRFFQ69G5FAV","extra":{"nested":true}}"#,
        )
        .unwrap();
        assert_eq!(scope, Scope::Unknown);
    }

    #[test]
    fn session_scope_deserializes_unknown_variant() {
        let scope: crate::SessionScope = serde_json::from_str(
            r#"{"type":"epic","epic_id":"01ARZ3NDEKTSV4RRFFQ69G5FAV","extra":{"nested":true}}"#,
        )
        .unwrap();
        assert_eq!(scope, crate::SessionScope::Unknown);
    }

    #[test]
    fn daemon_hello_json_roundtrips() {
        let hello = DaemonHello {
            host_id: "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap(),
            host_instance_id: "01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap(),
            capabilities: vec!["git".into(), "agents".into()],
            supported_protocol: ProtocolVersion::new(1, 7),
        };

        let json = serde_json::to_value(&hello).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "host_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "host_instance_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
                "capabilities": ["git", "agents"],
                "supported_protocol": { "major": 1, "minor": 7 },
            })
        );

        let decoded: DaemonHello = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, hello);
    }

    #[test]
    fn protobuf_roundtrips_for_envelope_and_daemon_hello() {
        let msg_id: MsgId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
        let correlation_id: MsgId = "01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap();
        let sent_at: Timestamp = serde_json::from_str(r#""2026-01-19T00:00:00Z""#).unwrap();
        let trace_id = TraceId::from_bytes([
            0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef, 0xde, 0xad,
            0xbe, 0xef,
        ]);
        let repo_scope = RepoScope::new(
            "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap(),
            "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap(),
        );

        let envelope = ProtocolEnvelope {
            protocol_major: PROTOCOL_MAJOR,
            protocol_minor: PROTOCOL_MINOR,
            msg_id,
            sent_at,
            scope: Some(Scope::Repo { repo: repo_scope }),
            correlation_id: Some(correlation_id),
            trace_id: Some(trace_id),
        };

        let envelope_pb = envelope.to_protobuf();
        assert_eq!(envelope_pb.msg_id, msg_id.to_bytes().to_vec());
        assert_eq!(
            envelope_pb.correlation_id,
            correlation_id.to_bytes().to_vec()
        );
        assert_eq!(envelope_pb.trace_id, trace_id.to_bytes().to_vec());
        let sent_at_pb = envelope_pb.sent_at.as_ref().unwrap();
        let sent_at_dt = sent_at.into_offset_date_time();
        assert_eq!(sent_at_pb.seconds, sent_at_dt.unix_timestamp());
        assert_eq!(sent_at_pb.nanos, sent_at_dt.nanosecond() as i32);

        let envelope_bytes = envelope_pb.encode_to_vec();
        let envelope_pb_decoded =
            pbv1::ProtocolEnvelope::decode(envelope_bytes.as_slice()).unwrap();
        let envelope_decoded = ProtocolEnvelope::try_from_protobuf(envelope_pb_decoded).unwrap();
        assert_eq!(envelope_decoded, envelope);

        let hello = DaemonHello {
            host_id: HostId::new(),
            host_instance_id: HostInstanceId::new(),
            capabilities: vec!["git".into(), "agents".into()],
            supported_protocol: ProtocolVersion::CURRENT,
        };

        let hello_pb = hello.to_protobuf();
        assert_eq!(hello_pb.host_id, hello.host_id.to_bytes().to_vec());
        assert_eq!(
            hello_pb.host_instance_id,
            hello.host_instance_id.to_bytes().to_vec()
        );

        let hello_bytes = hello_pb.encode_to_vec();
        let hello_pb_decoded = pbv1::DaemonHello::decode(hello_bytes.as_slice()).unwrap();
        let hello_decoded = DaemonHello::try_from_protobuf(hello_pb_decoded).unwrap();
        assert_eq!(hello_decoded, hello);
    }

    #[test]
    fn protobuf_roundtrips_for_daemon_frames() {
        let repo_scope = RepoScope::new(WorkspaceId::new(), RepoId::new());

        let batch = TelemetryEventBatch {
            scope: repo_scope,
            events: vec![TelemetryEvent::Git(GitEvent {
                event_type: "git.commit".to_string(),
                json_payload: br#"{"sha":"deadbeef"}"#.to_vec(),
            })],
        };

        let frame = DaemonFrame::new(
            ProtocolEnvelope::new().with_scope(repo_scope.into()),
            DaemonMessage::TelemetryEventBatch(batch),
        );

        let bytes = frame.to_protobuf().encode_to_vec();
        let pb = pbv1::DaemonFrame::decode(bytes.as_slice()).unwrap();
        let decoded = DaemonFrame::try_from_protobuf(pb).unwrap();
        assert_eq!(decoded, frame);

        let update = CommandUpdate {
            command_id: CommandId::new(),
            state: CommandState::Running,
            message: Some("working".to_string()),
            progress: Some(CommandProgress { percent: 42 }),
            detail: None,
            error: None,
        };

        let frame = DaemonFrame::new(
            ProtocolEnvelope::new().with_scope(repo_scope.into()),
            DaemonMessage::CommandUpdate(update),
        );

        let bytes = frame.to_protobuf().encode_to_vec();
        let pb = pbv1::DaemonFrame::decode(bytes.as_slice()).unwrap();
        let decoded = DaemonFrame::try_from_protobuf(pb).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn protobuf_and_json_roundtrip_for_client_frames() {
        let request_id = RequestId::new();
        let subscription_id = SubscriptionId::new();

        let frames = vec![
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Request(Request {
                    request_id,
                    payload: RequestPayload::Health(HealthRequest {}),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Response(Response {
                    request_id,
                    result: ResponseResult::Health(HealthResponse { ok: true }),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Request(Request {
                    request_id: RequestId::new(),
                    payload: RequestPayload::ListEpics(ListEpicsRequest {}),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Response(Response {
                    request_id: RequestId::new(),
                    result: ResponseResult::ListEpics(ListEpicsResponse {
                        epics: vec![EpicSummary {
                            slug: "gpui".to_string(),
                            name: "GPUI + Rust Port".to_string(),
                            epic_id: Some(EpicId::new()),
                        }],
                    }),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Request(Request {
                    request_id: RequestId::new(),
                    payload: RequestPayload::GetEpicGraph(GetEpicGraphRequest {
                        epic_slug: "gpui".to_string(),
                    }),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Response(Response {
                    request_id: RequestId::new(),
                    result: ResponseResult::GetEpicGraph(GetEpicGraphResponse {
                        graph: EpicGraph {
                            epic_slug: "gpui".to_string(),
                            nodes: vec![EpicTaskNode {
                                task_slug: "T-12".to_string(),
                                title: "Client API over UDS".to_string(),
                                task_id: None,
                                parent_task_id: None,
                                state: crate::client::TaskState::Unknown,
                                branch_name: None,
                                merge_readiness: crate::client::MergeReadiness::Unknown,
                            }],
                            edges: vec![EpicTaskEdge {
                                from_task_slug: "T-10".to_string(),
                                to_task_slug: "T-12".to_string(),
                                from_task_id: None,
                                to_task_id: None,
                            }],
                            epic_id: None,
                            epic_title: None,
                            workspace_id: None,
                            repo_id: None,
                            command_summaries: Vec::new(),
                            daemon_presences: Vec::new(),
                            session_summaries: Vec::new(),
                            as_of_event_id: None,
                        },
                    }),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Subscribe(Subscribe {
                    subscription_id,
                    filter: SubscriptionFilter::EventLog(EventLogFilter {
                        after_event_id: None,
                    }),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Event(Event {
                    subscription_id,
                    event: SubscriptionEvent::Subscribed(Subscribed {
                        topic: SubscriptionTopic::EventLog,
                    }),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Event(Event {
                    subscription_id: SubscriptionId::new(),
                    event: SubscriptionEvent::EventLog(EventLogEvent {
                        event_id: EventId::new(),
                        occurred_at: Timestamp::now_utc(),
                        event_type: "demo.noop".to_string(),
                        json_payload: vec![1, 2, 3],
                    }),
                }),
            ),
            ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Unsubscribe(Unsubscribe {
                    subscription_id: SubscriptionId::new(),
                }),
            ),
        ];

        for frame in frames {
            let json = serde_json::to_string(&frame).unwrap();
            let decoded_json: ClientFrame = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded_json, frame);

            let pb = frame.to_protobuf();
            let bytes = pb.encode_to_vec();
            let decoded_pb = pbv1::ClientFrame::decode(bytes.as_slice()).unwrap();
            let decoded = ClientFrame::try_from_protobuf(decoded_pb).unwrap();
            assert_eq!(decoded, frame);
        }
    }

    #[test]
    fn protobuf_and_json_roundtrip_for_artifacts_and_session_events() {
        use crate::{
            ArtifactEmitted, ArtifactKind, ArtifactRef, ExternalSessionRef, Hash, InterfaceMode,
            SessionEnded, SessionEvent, SessionEventKind, SessionScope, SessionStarted,
            StatusUpdate, StorageHint, ToolInvocation, ToolResult, TurnCompleted, TurnStarted,
            TurnState, UnknownSessionEvent, UserMessage,
        };

        let artifact_ref = ArtifactRef {
            artifact_id: redesmyn_ids::ArtifactId::new(),
            kind: ArtifactKind::Log,
            content_hash: Some(Hash {
                algorithm: "sha256".to_string(),
                digest: vec![1, 2, 3],
            }),
            byte_len: Some(123),
            mime: Some("text/plain".to_string()),
            storage_hint: Some(StorageHint::LocalPath {
                local_path: "/tmp/demo.log".to_string(),
            }),
        };

        let json = serde_json::to_string(&artifact_ref).unwrap();
        let decoded_json: ArtifactRef = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded_json, artifact_ref);

        let pb = artifact_ref.to_protobuf();
        let bytes = pb.encode_to_vec();
        let decoded_pb = pbv1::ArtifactRef::decode(bytes.as_slice()).unwrap();
        let decoded = ArtifactRef::try_from_protobuf(decoded_pb).unwrap();
        assert_eq!(decoded, artifact_ref);

        let created_at: Timestamp = serde_json::from_str(r#""2026-01-19T00:00:00Z""#).unwrap();

        let events = vec![
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Task {
                    task_id: redesmyn_ids::TaskId::new(),
                },
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: None,
                kind: SessionEventKind::SessionStarted(SessionStarted {}),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: Some("turn-1".to_string()),
                kind: SessionEventKind::TurnStarted(TurnStarted {
                    interface_mode: InterfaceMode::Structured,
                    external_session_ref: Some(ExternalSessionRef::CodexThread {
                        thread_id: "thread-1".to_string(),
                        turn_id: Some("ext-turn-1".to_string()),
                    }),
                    idempotency_key: Some("idem-1".to_string()),
                    log_offset_bytes: Some(42),
                }),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: Some("turn-1".to_string()),
                kind: SessionEventKind::UserMessage(UserMessage {
                    text: "hello".to_string(),
                    preview: "hello".to_string(),
                    full_text_artifact: Some(artifact_ref.clone()),
                }),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: Some("turn-1".to_string()),
                kind: SessionEventKind::ToolInvocation(ToolInvocation {
                    tool_name: "read_file".to_string(),
                    tool_call_id: Some("call-1".to_string()),
                    input_preview: "{\"path\":\"README.md\"}".to_string(),
                    input_artifact: Some(artifact_ref.clone()),
                }),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: Some("turn-1".to_string()),
                kind: SessionEventKind::ToolResult(ToolResult {
                    tool_name: "read_file".to_string(),
                    tool_call_id: Some("call-1".to_string()),
                    output_preview: "ok".to_string(),
                    output_artifact: Some(artifact_ref.clone()),
                    error: None,
                }),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: Some("turn-1".to_string()),
                kind: SessionEventKind::StatusUpdate(StatusUpdate {
                    turn_state: TurnState::Running,
                    blocking: Some(false),
                    progress_percent: Some(25),
                    message: Some("working".to_string()),
                }),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: Some("turn-1".to_string()),
                kind: SessionEventKind::ArtifactEmitted(ArtifactEmitted {
                    artifact: artifact_ref.clone(),
                    label: Some("turn log".to_string()),
                }),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: Some("turn-1".to_string()),
                kind: SessionEventKind::TurnCompleted(TurnCompleted {
                    interface_mode: InterfaceMode::Structured,
                    external_session_ref: Some(ExternalSessionRef::CodexThread {
                        thread_id: "thread-1".to_string(),
                        turn_id: Some("ext-turn-1".to_string()),
                    }),
                    exit_code: Some(0),
                    error: None,
                }),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: None,
                kind: SessionEventKind::SessionEnded(SessionEnded {}),
            },
            SessionEvent {
                session_event_id: redesmyn_ids::SessionEventId::new(),
                created_at,
                scope: SessionScope::Chat,
                session_id: redesmyn_ids::SessionId::new(),
                turn_id: None,
                kind: SessionEventKind::Unknown(UnknownSessionEvent {
                    event_type: "demo.unknown".to_string(),
                    json_payload: vec![9, 8, 7],
                }),
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let decoded_json: SessionEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded_json, event);

            let pb = event.to_protobuf();
            let bytes = pb.encode_to_vec();
            let decoded_pb = pbv1::SessionEvent::decode(bytes.as_slice()).unwrap();
            let decoded = SessionEvent::try_from_protobuf(decoded_pb).unwrap();
            assert_eq!(decoded, event);
        }
    }
}
