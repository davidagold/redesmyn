use std::collections::{BTreeMap, HashMap};

use redesmyn_ids::{HostId, HostInstanceId, MsgId, RepoId, WorkspaceId};

use crate::pb::redesmyn::protocol::v1 as pbv1;
use crate::{DaemonHello, ErrorCategory, ErrorDetail, ErrorEnvelope, ProtocolEnvelope, ProtocolVersion, RepoScope, Scope, Timestamp, TraceId};

fn invalid_request(message: impl Into<String>) -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCategory::InvalidRequest, message)
}

fn missing_required(field: &'static str) -> ErrorEnvelope {
    invalid_request(format!("missing required field: {field}"))
}

fn invalid_field(field: &'static str, message: impl Into<String>) -> ErrorEnvelope {
    invalid_request(format!("invalid {field}: {}", message.into()))
}

fn u32_to_u16(field: &'static str, value: u32) -> Result<u16, ErrorEnvelope> {
    if value > u16::MAX as u32 {
        return Err(invalid_field(field, format!("out of range for u16: {value}")));
    }
    Ok(value as u16)
}

fn decode_required_ulid<T>(field: &'static str, bytes: &[u8]) -> Result<T, ErrorEnvelope>
where
    for<'a> T: TryFrom<&'a [u8]>,
    for<'a> <T as TryFrom<&'a [u8]>>::Error: std::fmt::Display,
{
    if bytes.is_empty() {
        return Err(missing_required(field));
    }
    T::try_from(bytes).map_err(|err| invalid_field(field, err.to_string()))
}

fn decode_optional_ulid<T>(field: &'static str, bytes: &[u8]) -> Result<Option<T>, ErrorEnvelope>
where
    for<'a> T: TryFrom<&'a [u8]>,
    for<'a> <T as TryFrom<&'a [u8]>>::Error: std::fmt::Display,
{
    if bytes.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        T::try_from(bytes).map_err(|err| invalid_field(field, err.to_string()))?,
    ))
}

fn encode_timestamp(value: Timestamp) -> prost_types::Timestamp {
    let dt = value.into_offset_date_time();
    prost_types::Timestamp {
        seconds: dt.unix_timestamp(),
        nanos: dt.nanosecond() as i32,
    }
}

fn decode_required_timestamp(
    field: &'static str,
    value: Option<prost_types::Timestamp>,
) -> Result<Timestamp, ErrorEnvelope> {
    decode_timestamp(field, value.ok_or_else(|| missing_required(field))?)
}

fn decode_timestamp(field: &'static str, value: prost_types::Timestamp) -> Result<Timestamp, ErrorEnvelope> {
    let nanos: i32 = value.nanos;
    if !(0..=999_999_999).contains(&nanos) {
        return Err(invalid_field(field, format!("nanos out of range: {nanos}")));
    }

    let seconds_nanos = (value.seconds as i128)
        .checked_mul(1_000_000_000)
        .and_then(|base| base.checked_add(nanos as i128))
        .ok_or_else(|| invalid_field(field, "timestamp out of range"))?;

    let dt = time::OffsetDateTime::from_unix_timestamp_nanos(seconds_nanos)
        .map_err(|err| invalid_field(field, err.to_string()))?;
    Ok(Timestamp::from_offset_date_time(dt))
}

fn decode_optional_trace_id(field: &'static str, bytes: &[u8]) -> Result<Option<TraceId>, ErrorEnvelope> {
    if bytes.is_empty() {
        return Ok(None);
    }

    TraceId::try_from_bytes_slice(bytes)
        .map(Some)
        .map_err(|err| invalid_field(field, err.to_string()))
}

fn encode_trace_id(value: TraceId) -> Vec<u8> {
    value.to_bytes().to_vec()
}

fn encode_error_category(value: ErrorCategory) -> i32 {
    match value {
        ErrorCategory::InvalidRequest => pbv1::ErrorCategory::InvalidRequest as i32,
        ErrorCategory::NotFound => pbv1::ErrorCategory::NotFound as i32,
        ErrorCategory::Conflict => pbv1::ErrorCategory::Conflict as i32,
        ErrorCategory::Unauthorized => pbv1::ErrorCategory::Unauthorized as i32,
        ErrorCategory::Unavailable => pbv1::ErrorCategory::Unavailable as i32,
        ErrorCategory::Internal => pbv1::ErrorCategory::Internal as i32,
    }
}

fn decode_error_category(value: i32) -> ErrorCategory {
    match pbv1::ErrorCategory::try_from(value) {
        Ok(pbv1::ErrorCategory::InvalidRequest) => ErrorCategory::InvalidRequest,
        Ok(pbv1::ErrorCategory::NotFound) => ErrorCategory::NotFound,
        Ok(pbv1::ErrorCategory::Conflict) => ErrorCategory::Conflict,
        Ok(pbv1::ErrorCategory::Unauthorized) => ErrorCategory::Unauthorized,
        Ok(pbv1::ErrorCategory::Unavailable) => ErrorCategory::Unavailable,
        Ok(pbv1::ErrorCategory::Internal) => ErrorCategory::Internal,
        Ok(pbv1::ErrorCategory::Unspecified) | Err(_) => ErrorCategory::Internal,
    }
}

fn encode_error_detail(detail: &Option<ErrorDetail>) -> HashMap<String, String> {
    match detail {
        Some(detail) => detail.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => HashMap::new(),
    }
}

fn decode_error_detail(detail: HashMap<String, String>) -> Option<ErrorDetail> {
    if detail.is_empty() {
        return None;
    }

    Some(BTreeMap::from_iter(detail))
}

impl ProtocolVersion {
    #[must_use]
    pub fn to_protobuf(self) -> pbv1::ProtocolVersion {
        pbv1::ProtocolVersion {
            major: self.major as u32,
            minor: self.minor as u32,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ProtocolVersion) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            major: u32_to_u16("protocol_version.major", proto.major)?,
            minor: u32_to_u16("protocol_version.minor", proto.minor)?,
        })
    }
}

impl RepoScope {
    #[must_use]
    pub fn to_protobuf(self) -> pbv1::RepoScope {
        pbv1::RepoScope {
            workspace_id: self.workspace_id.to_bytes().to_vec(),
            repo_id: self.repo_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::RepoScope) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            workspace_id: decode_required_ulid::<WorkspaceId>("scope.workspace_id", &proto.workspace_id)?,
            repo_id: decode_required_ulid::<RepoId>("scope.repo_id", &proto.repo_id)?,
        })
    }
}

impl Scope {
    #[must_use]
    pub fn to_protobuf(self) -> pbv1::Scope {
        match self {
            Self::None | Self::Unknown => pbv1::Scope { kind: None },
            Self::Repo { repo } => pbv1::Scope {
                kind: Some(pbv1::scope::Kind::Repo(repo.to_protobuf())),
            },
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Scope) -> Result<Self, ErrorEnvelope> {
        match proto.kind {
            None => Ok(Self::None),
            Some(pbv1::scope::Kind::Repo(repo)) => Ok(Self::Repo {
                repo: RepoScope::try_from_protobuf(repo)?,
            }),
        }
    }
}

impl ErrorEnvelope {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ErrorEnvelope {
        pbv1::ErrorEnvelope {
            category: encode_error_category(self.category),
            message: self.message.clone(),
            detail: encode_error_detail(&self.detail),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::ErrorEnvelope) -> Self {
        Self {
            category: decode_error_category(proto.category),
            message: proto.message,
            detail: decode_error_detail(proto.detail),
        }
    }
}

impl ProtocolEnvelope {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ProtocolEnvelope {
        pbv1::ProtocolEnvelope {
            protocol_major: self.protocol_major as u32,
            protocol_minor: self.protocol_minor as u32,
            msg_id: self.msg_id.to_bytes().to_vec(),
            sent_at: Some(encode_timestamp(self.sent_at)),
            scope: self.scope.map(|scope| scope.to_protobuf()),
            correlation_id: self
                .correlation_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            trace_id: self.trace_id.map(encode_trace_id).unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ProtocolEnvelope) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            protocol_major: u32_to_u16("protocol_major", proto.protocol_major)?,
            protocol_minor: u32_to_u16("protocol_minor", proto.protocol_minor)?,
            msg_id: decode_required_ulid::<MsgId>("msg_id", &proto.msg_id)?,
            sent_at: decode_required_timestamp("sent_at", proto.sent_at)?,
            scope: proto
                .scope
                .map(Scope::try_from_protobuf)
                .transpose()?,
            correlation_id: decode_optional_ulid::<MsgId>("correlation_id", &proto.correlation_id)?,
            trace_id: decode_optional_trace_id("trace_id", &proto.trace_id)?,
        })
    }
}

impl DaemonHello {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::DaemonHello {
        pbv1::DaemonHello {
            host_id: self.host_id.to_bytes().to_vec(),
            host_instance_id: self.host_instance_id.to_bytes().to_vec(),
            capabilities: self.capabilities.clone(),
            supported_protocol: Some(self.supported_protocol.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::DaemonHello) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            host_id: decode_required_ulid::<HostId>("host_id", &proto.host_id)?,
            host_instance_id: decode_required_ulid::<HostInstanceId>(
                "host_instance_id",
                &proto.host_instance_id,
            )?,
            capabilities: proto.capabilities,
            supported_protocol: ProtocolVersion::try_from_protobuf(
                proto.supported_protocol
                    .ok_or_else(|| missing_required("supported_protocol"))?,
            )?,
        })
    }
}
