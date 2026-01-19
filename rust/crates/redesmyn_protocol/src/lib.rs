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

pub use redesmyn_errors::ErrorCategory;

pub mod daemon;

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
            return Err(
                ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Protocol major version mismatch.",
                )
                .with_detail(detail),
            );
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct Timestamp(#[serde(with = "time::serde::rfc3339")] time::OffsetDateTime);

impl Timestamp {
    #[must_use]
    pub fn now_utc() -> Self {
        Self(time::OffsetDateTime::now_utc())
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
        write!(f, "TraceId({})", self)
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
            let hi = (chunk[0] as char).to_digit(16).ok_or_else(|| ParseTraceIdError::new(s))?;
            let lo = (chunk[1] as char).to_digit(16).ok_or_else(|| ParseTraceIdError::new(s))?;
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
        let s =
            <std::borrow::Cow<'de, str> as serde::Deserialize<'de>>::deserialize(deserializer)?;
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

#[cfg(test)]
mod tests {
    use super::{
        ErrorCategory, ErrorEnvelope, MsgId, ProtocolEnvelope, ProtocolVersion, RepoScope, Scope,
        Timestamp, TraceId, PROTOCOL_MAJOR, PROTOCOL_MINOR,
    };

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
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
            0x0d, 0x0e, 0x0f,
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
        assert!(json.starts_with('\"') && json.ends_with('\"'), "json={json}");
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
    fn protocol_envelope_includes_repo_scope_and_optional_ids() {
        let msg_id: MsgId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
        let correlation_id: MsgId = "01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap();
        let sent_at: Timestamp = serde_json::from_str(r#""2026-01-19T00:00:00Z""#).unwrap();
        let trace_id = TraceId::from_bytes([
            0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef, 0xde,
            0xad, 0xbe, 0xef,
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
        let scope: Scope = serde_json::from_str(r#"{"type":"workspace"}"#).unwrap();
        assert_eq!(scope, Scope::Unknown);
    }
}
