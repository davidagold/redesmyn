use std::collections::{BTreeMap, HashMap};

use redesmyn_ids::{
    ArtifactId, CommandId, CommandUpdateId, EpicId, EventId, HostId, HostInstanceId, MsgId, RepoId,
    RequestId, SessionEventId, SessionId, SubscriptionId, TaskId, WorkspaceId,
};

use crate::artifacts::{ArtifactKind, ArtifactRef, Hash, StorageHint};
use crate::daemon::{
    AgentEvent, CommandDispatch, CommandProgress, CommandState, CommandUpdate,
    ControlPlaneHelloAck, DaemonFrame, DaemonHeartbeat, DaemonMessage, GitEvent, MergeRunEvent,
    RepoAttach, RepoDetach, ResyncRequest, SessionEventBatch, TelemetryEvent, TelemetryEventBatch,
    TelemetryFreshness, TelemetrySnapshot, UnknownEvent, WorktreeEvent,
};
use crate::pb::redesmyn::protocol::v1 as pbv1;
use crate::session::{
    ArtifactEmitted, AssistantMessage, ExternalSessionRef, InterfaceMode, SessionEvent,
    SessionEventKind, SessionScope, StatusUpdate, ToolInvocation, ToolResult, TurnCompleted,
    TurnStarted, TurnState, UnknownSessionEvent, UserMessage,
};
use crate::{
    DaemonHello, ErrorCategory, ErrorDetail, ErrorEnvelope, ProtocolEnvelope, ProtocolVersion,
    RepoScope, Scope, Timestamp, TraceId,
};

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
        return Err(invalid_field(
            field,
            format!("out of range for u16: {value}"),
        ));
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

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|value| if value.is_empty() { None } else { Some(value) })
}

fn normalize_nonempty_string(value: String) -> Option<String> {
    if value.is_empty() { None } else { Some(value) }
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

fn decode_timestamp(
    field: &'static str,
    value: prost_types::Timestamp,
) -> Result<Timestamp, ErrorEnvelope> {
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

fn decode_optional_trace_id(
    field: &'static str,
    bytes: &[u8],
) -> Result<Option<TraceId>, ErrorEnvelope> {
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
            workspace_id: decode_required_ulid::<WorkspaceId>(
                "scope.workspace_id",
                &proto.workspace_id,
            )?,
            repo_id: decode_required_ulid::<RepoId>("scope.repo_id", &proto.repo_id)?,
        })
    }
}

impl Scope {
    #[must_use]
    pub fn to_protobuf(self) -> pbv1::Scope {
        match self {
            // `Scope` is only present when `ProtocolEnvelope.scope` is `Some(_)`.
            //
            // If we encounter a scope kind we don't understand, we still want to preserve the
            // distinction between:
            // - "no scope" (`ProtocolEnvelope.scope == None`), and
            // - "unknown scope kind" (`ProtocolEnvelope.scope == Some(Scope::Unknown)`).
            //
            // We represent `Unknown` as an empty `pbv1::Scope` message.
            Self::Unknown => pbv1::Scope { kind: None },
            Self::Repo { repo } => pbv1::Scope {
                kind: Some(pbv1::scope::Kind::Repo(repo.to_protobuf())),
            },
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Scope) -> Result<Self, ErrorEnvelope> {
        match proto.kind {
            // Forward compatibility: older binaries may decode a new scope kind into an "empty"
            // `pbv1::Scope` message (prost discards unknown fields by default).
            None => Ok(Self::Unknown),
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
            scope: proto.scope.map(Scope::try_from_protobuf).transpose()?,
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
                proto
                    .supported_protocol
                    .ok_or_else(|| missing_required("supported_protocol"))?,
            )?,
        })
    }
}

impl ControlPlaneHelloAck {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ControlPlaneHelloAck {
        pbv1::ControlPlaneHelloAck {
            accepted_protocol: Some(self.accepted_protocol.to_protobuf()),
            capabilities: self.capabilities.clone(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ControlPlaneHelloAck) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            accepted_protocol: ProtocolVersion::try_from_protobuf(
                proto
                    .accepted_protocol
                    .ok_or_else(|| missing_required("accepted_protocol"))?,
            )?,
            capabilities: proto.capabilities,
        })
    }
}

impl RepoAttach {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::RepoAttach {
        pbv1::RepoAttach {
            repo_scope: Some(self.repo_scope.to_protobuf()),
            repo_root_hint: self.repo_root_hint.clone().unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::RepoAttach) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            repo_scope: RepoScope::try_from_protobuf(
                proto
                    .repo_scope
                    .ok_or_else(|| missing_required("repo_scope"))?,
            )?,
            repo_root_hint: normalize_nonempty_string(proto.repo_root_hint),
        })
    }
}

impl RepoDetach {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::RepoDetach {
        pbv1::RepoDetach {
            repo_scope: Some(self.repo_scope.to_protobuf()),
            reason: self.reason.clone().unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::RepoDetach) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            repo_scope: RepoScope::try_from_protobuf(
                proto
                    .repo_scope
                    .ok_or_else(|| missing_required("repo_scope"))?,
            )?,
            reason: normalize_nonempty_string(proto.reason),
        })
    }
}

impl TelemetryFreshness {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TelemetryFreshness {
        pbv1::TelemetryFreshness {
            scope: Some(self.scope.to_protobuf()),
            last_event_batch_msg_id: self
                .last_event_batch_msg_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            last_snapshot_msg_id: self
                .last_snapshot_msg_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::TelemetryFreshness) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            scope: RepoScope::try_from_protobuf(
                proto.scope.ok_or_else(|| missing_required("scope"))?,
            )?,
            last_event_batch_msg_id: decode_optional_ulid::<MsgId>(
                "last_event_batch_msg_id",
                &proto.last_event_batch_msg_id,
            )?,
            last_snapshot_msg_id: decode_optional_ulid::<MsgId>(
                "last_snapshot_msg_id",
                &proto.last_snapshot_msg_id,
            )?,
        })
    }
}

impl DaemonHeartbeat {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::DaemonHeartbeat {
        pbv1::DaemonHeartbeat {
            attached_repo_scopes: self
                .attached_repo_scopes
                .iter()
                .map(|scope| scope.to_protobuf())
                .collect(),
            telemetry_freshness: self
                .telemetry_freshness
                .iter()
                .map(|freshness| freshness.to_protobuf())
                .collect(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::DaemonHeartbeat) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            attached_repo_scopes: proto
                .attached_repo_scopes
                .into_iter()
                .map(RepoScope::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            telemetry_freshness: proto
                .telemetry_freshness
                .into_iter()
                .map(TelemetryFreshness::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl GitEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GitEvent {
        pbv1::GitEvent {
            event_type: self.event_type.clone(),
            json_payload: self.json_payload.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::GitEvent) -> Self {
        Self {
            event_type: proto.event_type,
            json_payload: proto.json_payload,
        }
    }
}

impl WorktreeEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WorktreeEvent {
        pbv1::WorktreeEvent {
            event_type: self.event_type.clone(),
            json_payload: self.json_payload.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::WorktreeEvent) -> Self {
        Self {
            event_type: proto.event_type,
            json_payload: proto.json_payload,
        }
    }
}

impl AgentEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::AgentEvent {
        pbv1::AgentEvent {
            event_type: self.event_type.clone(),
            json_payload: self.json_payload.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::AgentEvent) -> Self {
        Self {
            event_type: proto.event_type,
            json_payload: proto.json_payload,
        }
    }
}

impl MergeRunEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::MergeRunEvent {
        pbv1::MergeRunEvent {
            event_type: self.event_type.clone(),
            json_payload: self.json_payload.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::MergeRunEvent) -> Self {
        Self {
            event_type: proto.event_type,
            json_payload: proto.json_payload,
        }
    }
}

impl UnknownEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UnknownEvent {
        pbv1::UnknownEvent {
            event_type: self.event_type.clone(),
            json_payload: self.json_payload.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::UnknownEvent) -> Self {
        Self {
            event_type: proto.event_type,
            json_payload: proto.json_payload,
        }
    }
}

impl TelemetryEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TelemetryEvent {
        pbv1::TelemetryEvent {
            kind: Some(match self {
                Self::Git(event) => pbv1::telemetry_event::Kind::Git(event.to_protobuf()),
                Self::Worktree(event) => pbv1::telemetry_event::Kind::Worktree(event.to_protobuf()),
                Self::Agent(event) => pbv1::telemetry_event::Kind::Agent(event.to_protobuf()),
                Self::MergeRun(event) => pbv1::telemetry_event::Kind::MergeRun(event.to_protobuf()),
                Self::Unknown(event) => pbv1::telemetry_event::Kind::Unknown(event.to_protobuf()),
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::TelemetryEvent) -> Result<Self, ErrorEnvelope> {
        match proto.kind {
            // Forward compatibility: older binaries may decode a new event kind into an "empty"
            // message (unknown fields are discarded by default).
            None => Ok(Self::Unknown(UnknownEvent {
                event_type: "<unknown>".to_string(),
                json_payload: Vec::new(),
            })),
            Some(pbv1::telemetry_event::Kind::Git(event)) => {
                Ok(Self::Git(GitEvent::from_protobuf(event)))
            }
            Some(pbv1::telemetry_event::Kind::Worktree(event)) => {
                Ok(Self::Worktree(WorktreeEvent::from_protobuf(event)))
            }
            Some(pbv1::telemetry_event::Kind::Agent(event)) => {
                Ok(Self::Agent(AgentEvent::from_protobuf(event)))
            }
            Some(pbv1::telemetry_event::Kind::MergeRun(event)) => {
                Ok(Self::MergeRun(MergeRunEvent::from_protobuf(event)))
            }
            Some(pbv1::telemetry_event::Kind::Unknown(event)) => {
                Ok(Self::Unknown(UnknownEvent::from_protobuf(event)))
            }
        }
    }
}

impl TelemetryEventBatch {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TelemetryEventBatch {
        pbv1::TelemetryEventBatch {
            scope: Some(self.scope.to_protobuf()),
            events: self
                .events
                .iter()
                .map(|event| event.to_protobuf())
                .collect(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::TelemetryEventBatch) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            scope: RepoScope::try_from_protobuf(
                proto.scope.ok_or_else(|| missing_required("scope"))?,
            )?,
            events: proto
                .events
                .into_iter()
                .map(TelemetryEvent::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl TelemetrySnapshot {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TelemetrySnapshot {
        pbv1::TelemetrySnapshot {
            scope: Some(self.scope.to_protobuf()),
            json_payload: self.json_payload.clone(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::TelemetrySnapshot) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            scope: RepoScope::try_from_protobuf(
                proto.scope.ok_or_else(|| missing_required("scope"))?,
            )?,
            json_payload: proto.json_payload,
        })
    }
}

impl ResyncRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ResyncRequest {
        pbv1::ResyncRequest {
            scope: Some(self.scope.to_protobuf()),
            reason: self.reason.clone().unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ResyncRequest) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            scope: RepoScope::try_from_protobuf(
                proto.scope.ok_or_else(|| missing_required("scope"))?,
            )?,
            reason: normalize_nonempty_string(proto.reason),
        })
    }
}

impl CommandDispatch {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CommandDispatch {
        pbv1::CommandDispatch {
            command_id: self.command_id.to_bytes().to_vec(),
            scope: Some(self.scope.to_protobuf()),
            command_kind: self.command_kind.clone(),
            json_payload: self.json_payload.clone(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CommandDispatch) -> Result<Self, ErrorEnvelope> {
        if proto.command_kind.is_empty() {
            return Err(missing_required("command_kind"));
        }

        Ok(Self {
            command_id: decode_required_ulid::<CommandId>("command_id", &proto.command_id)?,
            scope: RepoScope::try_from_protobuf(
                proto.scope.ok_or_else(|| missing_required("scope"))?,
            )?,
            command_kind: proto.command_kind,
            json_payload: proto.json_payload,
        })
    }
}

fn encode_command_state(value: CommandState) -> i32 {
    match value {
        CommandState::Accepted => pbv1::CommandState::Accepted as i32,
        CommandState::Running => pbv1::CommandState::Running as i32,
        CommandState::Succeeded => pbv1::CommandState::Succeeded as i32,
        CommandState::Failed => pbv1::CommandState::Failed as i32,
        CommandState::Canceled => pbv1::CommandState::Canceled as i32,
        CommandState::Rejected => pbv1::CommandState::Rejected as i32,
    }
}

fn decode_command_state(value: i32) -> Result<CommandState, ErrorEnvelope> {
    match pbv1::CommandState::try_from(value) {
        Ok(pbv1::CommandState::Accepted) => Ok(CommandState::Accepted),
        Ok(pbv1::CommandState::Running) => Ok(CommandState::Running),
        Ok(pbv1::CommandState::Succeeded) => Ok(CommandState::Succeeded),
        Ok(pbv1::CommandState::Failed) => Ok(CommandState::Failed),
        Ok(pbv1::CommandState::Canceled) => Ok(CommandState::Canceled),
        Ok(pbv1::CommandState::Rejected) => Ok(CommandState::Rejected),
        Ok(pbv1::CommandState::Blocked)
        | Ok(pbv1::CommandState::Resumable)
        | Ok(pbv1::CommandState::Unspecified)
        | Err(_) => Err(invalid_field(
            "state",
            format!("unknown enum value for CommandState: {value}"),
        )),
    }
}

impl CommandProgress {
    #[must_use]
    pub fn to_protobuf(self) -> pbv1::CommandProgress {
        pbv1::CommandProgress {
            percent: self.percent,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CommandProgress) -> Result<Self, ErrorEnvelope> {
        if proto.percent > 100 {
            return Err(invalid_field(
                "progress.percent",
                format!("out of range: {}", proto.percent),
            ));
        }

        Ok(Self {
            percent: proto.percent,
        })
    }
}

impl CommandUpdate {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CommandUpdate {
        pbv1::CommandUpdate {
            command_id: self.command_id.to_bytes().to_vec(),
            state: encode_command_state(self.state),
            message: self.message.clone().unwrap_or_default(),
            progress: self.progress.map(|progress| progress.to_protobuf()),
            detail: encode_error_detail(&self.detail),
            error: self.error.as_ref().map(|err| err.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CommandUpdate) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            command_id: decode_required_ulid::<CommandId>("command_id", &proto.command_id)?,
            state: decode_command_state(proto.state)?,
            message: normalize_nonempty_string(proto.message),
            progress: proto
                .progress
                .map(CommandProgress::try_from_protobuf)
                .transpose()?,
            detail: decode_error_detail(proto.detail),
            error: proto.error.map(ErrorEnvelope::from_protobuf),
        })
    }
}

impl SessionEventBatch {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::SessionEventBatch {
        pbv1::SessionEventBatch {
            events: self.events.iter().map(SessionEvent::to_protobuf).collect(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::SessionEventBatch) -> Result<Self, ErrorEnvelope> {
        let mut events = Vec::with_capacity(proto.events.len());
        for (idx, ev) in proto.events.into_iter().enumerate() {
            let decoded = SessionEvent::try_from_protobuf(ev).map_err(|err| {
                invalid_field(
                    "session_event_batch.events",
                    format!("{idx}: {}: {}", err.category, err.message),
                )
            })?;
            events.push(decoded);
        }

        Ok(Self { events })
    }
}

impl DaemonFrame {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::DaemonFrame {
        pbv1::DaemonFrame {
            envelope: Some(self.envelope.to_protobuf()),
            message: Some(match &self.message {
                DaemonMessage::DaemonHello(hello) => {
                    pbv1::daemon_frame::Message::DaemonHello(hello.to_protobuf())
                }
                DaemonMessage::ControlPlaneHelloAck(ack) => {
                    pbv1::daemon_frame::Message::ControlPlaneHelloAck(ack.to_protobuf())
                }
                DaemonMessage::RepoAttach(attach) => {
                    pbv1::daemon_frame::Message::RepoAttach(attach.to_protobuf())
                }
                DaemonMessage::RepoDetach(detach) => {
                    pbv1::daemon_frame::Message::RepoDetach(detach.to_protobuf())
                }
                DaemonMessage::DaemonHeartbeat(heartbeat) => {
                    pbv1::daemon_frame::Message::DaemonHeartbeat(heartbeat.to_protobuf())
                }
                DaemonMessage::TelemetryEventBatch(batch) => {
                    pbv1::daemon_frame::Message::TelemetryEventBatch(batch.to_protobuf())
                }
                DaemonMessage::TelemetrySnapshot(snapshot) => {
                    pbv1::daemon_frame::Message::TelemetrySnapshot(snapshot.to_protobuf())
                }
                DaemonMessage::ResyncRequest(req) => {
                    pbv1::daemon_frame::Message::ResyncRequest(req.to_protobuf())
                }
                DaemonMessage::CommandDispatch(dispatch) => {
                    pbv1::daemon_frame::Message::CommandDispatch(dispatch.to_protobuf())
                }
                DaemonMessage::CommandUpdate(update) => {
                    pbv1::daemon_frame::Message::CommandUpdate(update.to_protobuf())
                }
                DaemonMessage::SessionEventBatch(batch) => {
                    pbv1::daemon_frame::Message::SessionEventBatch(batch.to_protobuf())
                }
                DaemonMessage::Error(error) => {
                    pbv1::daemon_frame::Message::Error(error.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::DaemonFrame) -> Result<Self, ErrorEnvelope> {
        let envelope = ProtocolEnvelope::try_from_protobuf(
            proto.envelope.ok_or_else(|| missing_required("envelope"))?,
        )?;

        let message = match proto.message.ok_or_else(|| missing_required("message"))? {
            pbv1::daemon_frame::Message::DaemonHello(hello) => {
                DaemonMessage::DaemonHello(DaemonHello::try_from_protobuf(hello)?)
            }
            pbv1::daemon_frame::Message::ControlPlaneHelloAck(ack) => {
                DaemonMessage::ControlPlaneHelloAck(ControlPlaneHelloAck::try_from_protobuf(ack)?)
            }
            pbv1::daemon_frame::Message::RepoAttach(attach) => {
                DaemonMessage::RepoAttach(RepoAttach::try_from_protobuf(attach)?)
            }
            pbv1::daemon_frame::Message::RepoDetach(detach) => {
                DaemonMessage::RepoDetach(RepoDetach::try_from_protobuf(detach)?)
            }
            pbv1::daemon_frame::Message::DaemonHeartbeat(heartbeat) => {
                DaemonMessage::DaemonHeartbeat(DaemonHeartbeat::try_from_protobuf(heartbeat)?)
            }
            pbv1::daemon_frame::Message::TelemetryEventBatch(batch) => {
                DaemonMessage::TelemetryEventBatch(TelemetryEventBatch::try_from_protobuf(batch)?)
            }
            pbv1::daemon_frame::Message::TelemetrySnapshot(snapshot) => {
                DaemonMessage::TelemetrySnapshot(TelemetrySnapshot::try_from_protobuf(snapshot)?)
            }
            pbv1::daemon_frame::Message::ResyncRequest(req) => {
                DaemonMessage::ResyncRequest(ResyncRequest::try_from_protobuf(req)?)
            }
            pbv1::daemon_frame::Message::CommandDispatch(dispatch) => {
                DaemonMessage::CommandDispatch(CommandDispatch::try_from_protobuf(dispatch)?)
            }
            pbv1::daemon_frame::Message::CommandUpdate(update) => {
                DaemonMessage::CommandUpdate(CommandUpdate::try_from_protobuf(update)?)
            }
            pbv1::daemon_frame::Message::SessionEventBatch(batch) => {
                DaemonMessage::SessionEventBatch(SessionEventBatch::try_from_protobuf(batch)?)
            }
            pbv1::daemon_frame::Message::Error(error) => {
                DaemonMessage::Error(ErrorEnvelope::from_protobuf(error))
            }
        };

        Ok(Self { envelope, message })
    }
}

fn encode_artifact_kind(value: ArtifactKind) -> i32 {
    match value {
        ArtifactKind::Log => pbv1::ArtifactKind::Log as i32,
        ArtifactKind::Diff => pbv1::ArtifactKind::Diff as i32,
        ArtifactKind::Patch => pbv1::ArtifactKind::Patch as i32,
        ArtifactKind::FileSnapshot => pbv1::ArtifactKind::FileSnapshot as i32,
        ArtifactKind::Trace => pbv1::ArtifactKind::Trace as i32,
        ArtifactKind::Unknown => pbv1::ArtifactKind::Unspecified as i32,
    }
}

fn decode_artifact_kind(value: i32) -> ArtifactKind {
    match pbv1::ArtifactKind::try_from(value) {
        Ok(pbv1::ArtifactKind::Log) => ArtifactKind::Log,
        Ok(pbv1::ArtifactKind::Diff) => ArtifactKind::Diff,
        Ok(pbv1::ArtifactKind::Patch) => ArtifactKind::Patch,
        Ok(pbv1::ArtifactKind::FileSnapshot) => ArtifactKind::FileSnapshot,
        Ok(pbv1::ArtifactKind::Trace) => ArtifactKind::Trace,
        Ok(pbv1::ArtifactKind::Unspecified) | Err(_) => ArtifactKind::Unknown,
    }
}

impl Hash {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::Hash {
        pbv1::Hash {
            algorithm: self.algorithm.clone(),
            digest: self.digest.clone(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Hash) -> Result<Self, ErrorEnvelope> {
        if proto.algorithm.is_empty() {
            return Err(invalid_field("hash.algorithm", "empty"));
        }
        if proto.digest.is_empty() {
            return Err(invalid_field("hash.digest", "empty"));
        }

        Ok(Self {
            algorithm: proto.algorithm,
            digest: proto.digest,
        })
    }
}

impl StorageHint {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::StorageHint {
        pbv1::StorageHint {
            hint: match self {
                Self::LocalPath { local_path } => {
                    Some(pbv1::storage_hint::Hint::LocalPath(local_path.clone()))
                }
                Self::BlobKey { blob_key } => {
                    Some(pbv1::storage_hint::Hint::BlobKey(blob_key.clone()))
                }
                Self::Unknown => None,
            },
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::StorageHint) -> Self {
        match proto.hint {
            Some(pbv1::storage_hint::Hint::LocalPath(local_path)) => Self::LocalPath { local_path },
            Some(pbv1::storage_hint::Hint::BlobKey(blob_key)) => Self::BlobKey { blob_key },
            None => Self::Unknown,
        }
    }
}

impl ArtifactRef {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ArtifactRef {
        pbv1::ArtifactRef {
            artifact_id: self.artifact_id.to_bytes().to_vec(),
            kind: encode_artifact_kind(self.kind),
            content_hash: self.content_hash.as_ref().map(Hash::to_protobuf),
            byte_len: self.byte_len,
            mime: self.mime.clone(),
            storage_hint: self.storage_hint.as_ref().map(StorageHint::to_protobuf),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ArtifactRef) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            artifact_id: decode_required_ulid::<ArtifactId>("artifact_id", &proto.artifact_id)?,
            kind: decode_artifact_kind(proto.kind),
            content_hash: proto
                .content_hash
                .map(Hash::try_from_protobuf)
                .transpose()?,
            byte_len: proto.byte_len,
            mime: normalize_optional_string(proto.mime),
            storage_hint: proto.storage_hint.map(StorageHint::from_protobuf),
        })
    }
}

impl SessionScope {
    #[must_use]
    pub fn to_protobuf(self) -> pbv1::SessionScope {
        match self {
            Self::Unknown => pbv1::SessionScope { kind: None },
            Self::Task { task_id } => pbv1::SessionScope {
                kind: Some(pbv1::session_scope::Kind::Task(pbv1::TaskScope {
                    task_id: task_id.to_bytes().to_vec(),
                })),
            },
            Self::Chat => pbv1::SessionScope {
                kind: Some(pbv1::session_scope::Kind::Chat(pbv1::ChatScope {})),
            },
        }
    }

    pub fn try_from_protobuf(proto: pbv1::SessionScope) -> Result<Self, ErrorEnvelope> {
        match proto.kind {
            // Forward compatibility: prost decodes a new scope kind into an "empty" message
            // (unknown fields are discarded by default).
            None => Ok(Self::Unknown),
            Some(pbv1::session_scope::Kind::Task(task)) => Ok(Self::Task {
                task_id: decode_required_ulid::<TaskId>("session_scope.task_id", &task.task_id)?,
            }),
            Some(pbv1::session_scope::Kind::Chat(_)) => Ok(Self::Chat),
        }
    }
}

fn encode_interface_mode(value: InterfaceMode) -> i32 {
    match value {
        InterfaceMode::Interactive => pbv1::InterfaceMode::Interactive as i32,
        InterfaceMode::Structured => pbv1::InterfaceMode::Structured as i32,
        InterfaceMode::Unknown => pbv1::InterfaceMode::Unspecified as i32,
    }
}

fn decode_interface_mode(value: i32) -> InterfaceMode {
    match pbv1::InterfaceMode::try_from(value) {
        Ok(pbv1::InterfaceMode::Interactive) => InterfaceMode::Interactive,
        Ok(pbv1::InterfaceMode::Structured) => InterfaceMode::Structured,
        Ok(pbv1::InterfaceMode::Unspecified) | Err(_) => InterfaceMode::Unknown,
    }
}

impl ExternalSessionRef {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ExternalSessionRef {
        pbv1::ExternalSessionRef {
            r#ref: Some(match self {
                Self::None => pbv1::external_session_ref::Ref::None(pbv1::ExternalSessionNone {}),
                Self::CodexThread { thread_id, turn_id } => {
                    pbv1::external_session_ref::Ref::CodexThread(pbv1::CodexThreadRef {
                        thread_id: thread_id.clone(),
                        turn_id: normalize_optional_string(turn_id.clone()),
                    })
                }
                Self::ClaudeSession { session_id } => {
                    pbv1::external_session_ref::Ref::ClaudeSession(pbv1::ClaudeSessionRef {
                        session_id: session_id.clone(),
                    })
                }
                Self::Unknown {
                    unknown_type,
                    json_payload,
                } => pbv1::external_session_ref::Ref::Unknown(pbv1::UnknownExternalSessionRef {
                    r#type: unknown_type.clone(),
                    json_payload: json_payload.clone(),
                }),
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ExternalSessionRef) -> Result<Self, ErrorEnvelope> {
        match proto.r#ref {
            // Forward compatibility: prost decodes a new ref kind into an "empty" message.
            None => Ok(Self::None),
            Some(pbv1::external_session_ref::Ref::None(_)) => Ok(Self::None),
            Some(pbv1::external_session_ref::Ref::CodexThread(codex)) => {
                if codex.thread_id.is_empty() {
                    return Err(invalid_field("codex_thread.thread_id", "empty"));
                }
                Ok(Self::CodexThread {
                    thread_id: codex.thread_id,
                    turn_id: normalize_optional_string(codex.turn_id),
                })
            }
            Some(pbv1::external_session_ref::Ref::ClaudeSession(claude)) => {
                if claude.session_id.is_empty() {
                    return Err(invalid_field("claude_session.session_id", "empty"));
                }
                Ok(Self::ClaudeSession {
                    session_id: claude.session_id,
                })
            }
            Some(pbv1::external_session_ref::Ref::Unknown(unknown)) => Ok(Self::Unknown {
                unknown_type: if unknown.r#type.is_empty() {
                    "<unknown>".to_owned()
                } else {
                    unknown.r#type
                },
                json_payload: unknown.json_payload,
            }),
        }
    }
}

fn encode_turn_state(value: TurnState) -> i32 {
    match value {
        TurnState::Running => pbv1::TurnState::Running as i32,
        TurnState::Blocked => pbv1::TurnState::Blocked as i32,
        TurnState::Completed => pbv1::TurnState::Completed as i32,
        TurnState::Unknown => pbv1::TurnState::Unspecified as i32,
    }
}

fn decode_turn_state(value: i32) -> TurnState {
    match pbv1::TurnState::try_from(value) {
        Ok(pbv1::TurnState::Running) => TurnState::Running,
        Ok(pbv1::TurnState::Blocked) => TurnState::Blocked,
        Ok(pbv1::TurnState::Completed) => TurnState::Completed,
        Ok(pbv1::TurnState::Unspecified) | Err(_) => TurnState::Unknown,
    }
}

impl TurnStarted {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TurnStarted {
        pbv1::TurnStarted {
            interface_mode: encode_interface_mode(self.interface_mode),
            external_session_ref: self
                .external_session_ref
                .as_ref()
                .map(ExternalSessionRef::to_protobuf),
            idempotency_key: normalize_optional_string(self.idempotency_key.clone()),
            log_offset_bytes: self.log_offset_bytes,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::TurnStarted) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            interface_mode: decode_interface_mode(proto.interface_mode),
            external_session_ref: proto
                .external_session_ref
                .map(ExternalSessionRef::try_from_protobuf)
                .transpose()?,
            idempotency_key: normalize_optional_string(proto.idempotency_key),
            log_offset_bytes: proto.log_offset_bytes,
        })
    }
}

impl TurnCompleted {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TurnCompleted {
        pbv1::TurnCompleted {
            interface_mode: encode_interface_mode(self.interface_mode),
            external_session_ref: self
                .external_session_ref
                .as_ref()
                .map(ExternalSessionRef::to_protobuf),
            exit_code: self.exit_code,
            error: self.error.as_ref().map(ErrorEnvelope::to_protobuf),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::TurnCompleted) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            interface_mode: decode_interface_mode(proto.interface_mode),
            external_session_ref: proto
                .external_session_ref
                .map(ExternalSessionRef::try_from_protobuf)
                .transpose()?,
            exit_code: proto.exit_code,
            error: proto.error.map(ErrorEnvelope::from_protobuf),
        })
    }
}

impl UserMessage {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UserMessage {
        pbv1::UserMessage {
            text: self.text.clone(),
            preview: self.preview.clone(),
            full_text_artifact: self
                .full_text_artifact
                .as_ref()
                .map(ArtifactRef::to_protobuf),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::UserMessage) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            text: proto.text,
            preview: proto.preview,
            full_text_artifact: proto
                .full_text_artifact
                .map(ArtifactRef::try_from_protobuf)
                .transpose()?,
        })
    }
}

impl AssistantMessage {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::AssistantMessage {
        pbv1::AssistantMessage {
            text: self.text.clone(),
            preview: self.preview.clone(),
            full_text_artifact: self
                .full_text_artifact
                .as_ref()
                .map(ArtifactRef::to_protobuf),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::AssistantMessage) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            text: proto.text,
            preview: proto.preview,
            full_text_artifact: proto
                .full_text_artifact
                .map(ArtifactRef::try_from_protobuf)
                .transpose()?,
        })
    }
}

impl ToolInvocation {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ToolInvocation {
        pbv1::ToolInvocation {
            tool_name: self.tool_name.clone(),
            tool_call_id: normalize_optional_string(self.tool_call_id.clone()),
            input_preview: self.input_preview.clone(),
            input_artifact: self.input_artifact.as_ref().map(ArtifactRef::to_protobuf),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ToolInvocation) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            tool_name: proto.tool_name,
            tool_call_id: normalize_optional_string(proto.tool_call_id),
            input_preview: proto.input_preview,
            input_artifact: proto
                .input_artifact
                .map(ArtifactRef::try_from_protobuf)
                .transpose()?,
        })
    }
}

impl ToolResult {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ToolResult {
        pbv1::ToolResult {
            tool_name: self.tool_name.clone(),
            tool_call_id: normalize_optional_string(self.tool_call_id.clone()),
            output_preview: self.output_preview.clone(),
            output_artifact: self.output_artifact.as_ref().map(ArtifactRef::to_protobuf),
            error: self.error.as_ref().map(ErrorEnvelope::to_protobuf),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ToolResult) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            tool_name: proto.tool_name,
            tool_call_id: normalize_optional_string(proto.tool_call_id),
            output_preview: proto.output_preview,
            output_artifact: proto
                .output_artifact
                .map(ArtifactRef::try_from_protobuf)
                .transpose()?,
            error: proto.error.map(ErrorEnvelope::from_protobuf),
        })
    }
}

impl StatusUpdate {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::StatusUpdate {
        pbv1::StatusUpdate {
            turn_state: encode_turn_state(self.turn_state),
            blocking: self.blocking,
            progress_percent: self.progress_percent,
            message: normalize_optional_string(self.message.clone()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::StatusUpdate) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            turn_state: decode_turn_state(proto.turn_state),
            blocking: proto.blocking,
            progress_percent: proto.progress_percent,
            message: normalize_optional_string(proto.message),
        })
    }
}

impl ArtifactEmitted {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ArtifactEmitted {
        pbv1::ArtifactEmitted {
            artifact: Some(self.artifact.to_protobuf()),
            label: normalize_optional_string(self.label.clone()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ArtifactEmitted) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            artifact: ArtifactRef::try_from_protobuf(
                proto
                    .artifact
                    .ok_or_else(|| missing_required("artifact_emitted.artifact"))?,
            )?,
            label: normalize_optional_string(proto.label),
        })
    }
}

impl UnknownSessionEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UnknownSessionEvent {
        pbv1::UnknownSessionEvent {
            event_type: self.event_type.clone(),
            json_payload: self.json_payload.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::UnknownSessionEvent) -> Self {
        Self {
            event_type: if proto.event_type.is_empty() {
                "<unknown>".to_owned()
            } else {
                proto.event_type
            },
            json_payload: proto.json_payload,
        }
    }
}

impl SessionEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::SessionEvent {
        pbv1::SessionEvent {
            session_event_id: self.session_event_id.to_bytes().to_vec(),
            created_at: Some(encode_timestamp(self.created_at)),
            scope: Some(self.scope.to_protobuf()),
            session_id: self.session_id.to_bytes().to_vec(),
            turn_id: normalize_optional_string(self.turn_id.clone()),
            kind: Some(match &self.kind {
                SessionEventKind::SessionStarted(_) => {
                    pbv1::session_event::Kind::SessionStarted(pbv1::SessionStarted {})
                }
                SessionEventKind::SessionEnded(_) => {
                    pbv1::session_event::Kind::SessionEnded(pbv1::SessionEnded {})
                }
                SessionEventKind::TurnStarted(ev) => {
                    pbv1::session_event::Kind::TurnStarted(ev.to_protobuf())
                }
                SessionEventKind::TurnCompleted(ev) => {
                    pbv1::session_event::Kind::TurnCompleted(ev.to_protobuf())
                }
                SessionEventKind::UserMessage(ev) => {
                    pbv1::session_event::Kind::UserMessage(ev.to_protobuf())
                }
                SessionEventKind::AssistantMessage(ev) => {
                    pbv1::session_event::Kind::AssistantMessage(ev.to_protobuf())
                }
                SessionEventKind::ToolInvocation(ev) => {
                    pbv1::session_event::Kind::ToolInvocation(ev.to_protobuf())
                }
                SessionEventKind::ToolResult(ev) => {
                    pbv1::session_event::Kind::ToolResult(ev.to_protobuf())
                }
                SessionEventKind::StatusUpdate(ev) => {
                    pbv1::session_event::Kind::StatusUpdate(ev.to_protobuf())
                }
                SessionEventKind::ArtifactEmitted(ev) => {
                    pbv1::session_event::Kind::ArtifactEmitted(ev.to_protobuf())
                }
                SessionEventKind::Unknown(ev) => {
                    pbv1::session_event::Kind::Unknown(ev.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::SessionEvent) -> Result<Self, ErrorEnvelope> {
        let kind = match proto.kind {
            Some(pbv1::session_event::Kind::SessionStarted(_)) => {
                SessionEventKind::SessionStarted(crate::session::SessionStarted {})
            }
            Some(pbv1::session_event::Kind::SessionEnded(_)) => {
                SessionEventKind::SessionEnded(crate::session::SessionEnded {})
            }
            Some(pbv1::session_event::Kind::TurnStarted(ev)) => {
                SessionEventKind::TurnStarted(TurnStarted::try_from_protobuf(ev)?)
            }
            Some(pbv1::session_event::Kind::TurnCompleted(ev)) => {
                SessionEventKind::TurnCompleted(TurnCompleted::try_from_protobuf(ev)?)
            }
            Some(pbv1::session_event::Kind::UserMessage(ev)) => {
                SessionEventKind::UserMessage(UserMessage::try_from_protobuf(ev)?)
            }
            Some(pbv1::session_event::Kind::AssistantMessage(ev)) => {
                SessionEventKind::AssistantMessage(AssistantMessage::try_from_protobuf(ev)?)
            }
            Some(pbv1::session_event::Kind::ToolInvocation(ev)) => {
                SessionEventKind::ToolInvocation(ToolInvocation::try_from_protobuf(ev)?)
            }
            Some(pbv1::session_event::Kind::ToolResult(ev)) => {
                SessionEventKind::ToolResult(ToolResult::try_from_protobuf(ev)?)
            }
            Some(pbv1::session_event::Kind::StatusUpdate(ev)) => {
                SessionEventKind::StatusUpdate(StatusUpdate::try_from_protobuf(ev)?)
            }
            Some(pbv1::session_event::Kind::ArtifactEmitted(ev)) => {
                SessionEventKind::ArtifactEmitted(ArtifactEmitted::try_from_protobuf(ev)?)
            }
            Some(pbv1::session_event::Kind::Unknown(ev)) => {
                SessionEventKind::Unknown(UnknownSessionEvent::from_protobuf(ev))
            }
            None => SessionEventKind::Unknown(UnknownSessionEvent {
                event_type: "<unknown>".to_owned(),
                json_payload: Vec::new(),
            }),
        };

        Ok(Self {
            session_event_id: decode_required_ulid::<SessionEventId>(
                "session_event_id",
                &proto.session_event_id,
            )?,
            created_at: decode_required_timestamp("created_at", proto.created_at)?,
            scope: SessionScope::try_from_protobuf(
                proto.scope.ok_or_else(|| missing_required("scope"))?,
            )?,
            session_id: decode_required_ulid::<SessionId>("session_id", &proto.session_id)?,
            turn_id: normalize_optional_string(proto.turn_id),
            kind,
        })
    }
}

fn encode_client_method(value: crate::client::ClientMethod) -> i32 {
    match value {
        crate::client::ClientMethod::Health => pbv1::ClientMethod::Health as i32,
        crate::client::ClientMethod::Status => pbv1::ClientMethod::Status as i32,
        crate::client::ClientMethod::ListEpics => pbv1::ClientMethod::ListEpics as i32,
        crate::client::ClientMethod::GetEpicGraph => pbv1::ClientMethod::GetEpicGraph as i32,
        crate::client::ClientMethod::ListTaskSessions => {
            pbv1::ClientMethod::ListTaskSessions as i32
        }
        crate::client::ClientMethod::CreateChatSession => {
            pbv1::ClientMethod::CreateChatSession as i32
        }
        crate::client::ClientMethod::CloseChatSession => {
            pbv1::ClientMethod::CloseChatSession as i32
        }
        crate::client::ClientMethod::ListChatSessions => {
            pbv1::ClientMethod::ListChatSessions as i32
        }
        crate::client::ClientMethod::PinChatSessionToEpic => {
            pbv1::ClientMethod::PinChatSessionToEpic as i32
        }
        crate::client::ClientMethod::UnpinChatSessionFromEpic => {
            pbv1::ClientMethod::UnpinChatSessionFromEpic as i32
        }
        crate::client::ClientMethod::GetSessionEvents => {
            pbv1::ClientMethod::GetSessionEvents as i32
        }
        crate::client::ClientMethod::GetLatestTaskSession => {
            pbv1::ClientMethod::GetLatestTaskSession as i32
        }
        crate::client::ClientMethod::GetEpicPinnedChatSession => {
            pbv1::ClientMethod::GetEpicPinnedChatSession as i32
        }
        crate::client::ClientMethod::CreateCommand => pbv1::ClientMethod::CreateCommand as i32,
        crate::client::ClientMethod::GetCommand => pbv1::ClientMethod::GetCommand as i32,
        crate::client::ClientMethod::WaitForCommand => pbv1::ClientMethod::WaitForCommand as i32,
        crate::client::ClientMethod::WaitForEvent => pbv1::ClientMethod::WaitForEvent as i32,
        crate::client::ClientMethod::WaitForIdle => pbv1::ClientMethod::WaitForIdle as i32,
    }
}

fn decode_client_method(value: i32) -> Result<crate::client::ClientMethod, ErrorEnvelope> {
    match pbv1::ClientMethod::try_from(value) {
        Ok(pbv1::ClientMethod::Health) => Ok(crate::client::ClientMethod::Health),
        Ok(pbv1::ClientMethod::Status) => Ok(crate::client::ClientMethod::Status),
        Ok(pbv1::ClientMethod::ListEpics) => Ok(crate::client::ClientMethod::ListEpics),
        Ok(pbv1::ClientMethod::GetEpicGraph) => Ok(crate::client::ClientMethod::GetEpicGraph),
        Ok(pbv1::ClientMethod::ListTaskSessions) => {
            Ok(crate::client::ClientMethod::ListTaskSessions)
        }
        Ok(pbv1::ClientMethod::CreateChatSession) => {
            Ok(crate::client::ClientMethod::CreateChatSession)
        }
        Ok(pbv1::ClientMethod::CloseChatSession) => {
            Ok(crate::client::ClientMethod::CloseChatSession)
        }
        Ok(pbv1::ClientMethod::ListChatSessions) => {
            Ok(crate::client::ClientMethod::ListChatSessions)
        }
        Ok(pbv1::ClientMethod::PinChatSessionToEpic) => {
            Ok(crate::client::ClientMethod::PinChatSessionToEpic)
        }
        Ok(pbv1::ClientMethod::UnpinChatSessionFromEpic) => {
            Ok(crate::client::ClientMethod::UnpinChatSessionFromEpic)
        }
        Ok(pbv1::ClientMethod::GetSessionEvents) => {
            Ok(crate::client::ClientMethod::GetSessionEvents)
        }
        Ok(pbv1::ClientMethod::GetLatestTaskSession) => {
            Ok(crate::client::ClientMethod::GetLatestTaskSession)
        }
        Ok(pbv1::ClientMethod::GetEpicPinnedChatSession) => {
            Ok(crate::client::ClientMethod::GetEpicPinnedChatSession)
        }
        Ok(pbv1::ClientMethod::CreateCommand) => Ok(crate::client::ClientMethod::CreateCommand),
        Ok(pbv1::ClientMethod::GetCommand) => Ok(crate::client::ClientMethod::GetCommand),
        Ok(pbv1::ClientMethod::WaitForCommand) => Ok(crate::client::ClientMethod::WaitForCommand),
        Ok(pbv1::ClientMethod::WaitForEvent) => Ok(crate::client::ClientMethod::WaitForEvent),
        Ok(pbv1::ClientMethod::WaitForIdle) => Ok(crate::client::ClientMethod::WaitForIdle),
        Ok(pbv1::ClientMethod::Unspecified) | Err(_) => Err(invalid_field(
            "method",
            format!("unknown enum value for ClientMethod: {value}"),
        )),
    }
}

fn encode_response_status(value: crate::client::ResponseStatus) -> i32 {
    match value {
        crate::client::ResponseStatus::Ok => pbv1::ResponseStatus::Ok as i32,
        crate::client::ResponseStatus::Error => pbv1::ResponseStatus::Error as i32,
    }
}

fn decode_response_status(value: i32) -> Result<crate::client::ResponseStatus, ErrorEnvelope> {
    match pbv1::ResponseStatus::try_from(value) {
        Ok(pbv1::ResponseStatus::Ok) => Ok(crate::client::ResponseStatus::Ok),
        Ok(pbv1::ResponseStatus::Error) => Ok(crate::client::ResponseStatus::Error),
        Ok(pbv1::ResponseStatus::Unspecified) | Err(_) => Err(invalid_field(
            "status",
            format!("unknown enum value for ResponseStatus: {value}"),
        )),
    }
}

fn encode_subscription_topic(value: crate::client::SubscriptionTopic) -> i32 {
    match value {
        crate::client::SubscriptionTopic::EventLog => pbv1::SubscriptionTopic::EventLog as i32,
        crate::client::SubscriptionTopic::SessionEvents => {
            pbv1::SubscriptionTopic::SessionEvents as i32
        }
    }
}

fn decode_subscription_topic(
    value: i32,
) -> Result<crate::client::SubscriptionTopic, ErrorEnvelope> {
    match pbv1::SubscriptionTopic::try_from(value) {
        Ok(pbv1::SubscriptionTopic::EventLog) => Ok(crate::client::SubscriptionTopic::EventLog),
        Ok(pbv1::SubscriptionTopic::SessionEvents) => {
            Ok(crate::client::SubscriptionTopic::SessionEvents)
        }
        Ok(pbv1::SubscriptionTopic::Unspecified) | Err(_) => Err(invalid_field(
            "topic",
            format!("unknown enum value for SubscriptionTopic: {value}"),
        )),
    }
}

fn encode_task_state(value: crate::client::TaskState) -> i32 {
    match value {
        crate::client::TaskState::Unknown => pbv1::TaskState::Unspecified as i32,
        crate::client::TaskState::Todo => pbv1::TaskState::Todo as i32,
        crate::client::TaskState::InProgress => pbv1::TaskState::InProgress as i32,
        crate::client::TaskState::Blocked => pbv1::TaskState::Blocked as i32,
        crate::client::TaskState::Done => pbv1::TaskState::Done as i32,
    }
}

fn decode_task_state(value: i32) -> crate::client::TaskState {
    match pbv1::TaskState::try_from(value) {
        Ok(pbv1::TaskState::Todo) => crate::client::TaskState::Todo,
        Ok(pbv1::TaskState::InProgress) => crate::client::TaskState::InProgress,
        Ok(pbv1::TaskState::Blocked) => crate::client::TaskState::Blocked,
        Ok(pbv1::TaskState::Done) => crate::client::TaskState::Done,
        Ok(pbv1::TaskState::Unspecified) | Err(_) => crate::client::TaskState::Unknown,
    }
}

fn encode_merge_readiness(value: crate::client::MergeReadiness) -> i32 {
    match value {
        crate::client::MergeReadiness::Unknown => pbv1::MergeReadiness::Unknown as i32,
        crate::client::MergeReadiness::Ready => pbv1::MergeReadiness::Ready as i32,
        crate::client::MergeReadiness::Blocked => pbv1::MergeReadiness::Blocked as i32,
    }
}

fn decode_merge_readiness(value: i32) -> crate::client::MergeReadiness {
    match pbv1::MergeReadiness::try_from(value) {
        Ok(pbv1::MergeReadiness::Ready) => crate::client::MergeReadiness::Ready,
        Ok(pbv1::MergeReadiness::Blocked) => crate::client::MergeReadiness::Blocked,
        Ok(pbv1::MergeReadiness::Unspecified) | Ok(pbv1::MergeReadiness::Unknown) | Err(_) => {
            crate::client::MergeReadiness::Unknown
        }
    }
}

fn encode_client_command_state(value: crate::client::CommandState) -> i32 {
    match value {
        crate::client::CommandState::Unknown => pbv1::CommandState::Unspecified as i32,
        crate::client::CommandState::Accepted => pbv1::CommandState::Accepted as i32,
        crate::client::CommandState::Running => pbv1::CommandState::Running as i32,
        crate::client::CommandState::Blocked => pbv1::CommandState::Blocked as i32,
        crate::client::CommandState::Resumable => pbv1::CommandState::Resumable as i32,
        crate::client::CommandState::Succeeded => pbv1::CommandState::Succeeded as i32,
        crate::client::CommandState::Failed => pbv1::CommandState::Failed as i32,
        crate::client::CommandState::Canceled => pbv1::CommandState::Canceled as i32,
    }
}

fn decode_client_command_state(value: i32) -> crate::client::CommandState {
    match pbv1::CommandState::try_from(value) {
        Ok(pbv1::CommandState::Accepted) => crate::client::CommandState::Accepted,
        Ok(pbv1::CommandState::Running) => crate::client::CommandState::Running,
        Ok(pbv1::CommandState::Blocked) => crate::client::CommandState::Blocked,
        Ok(pbv1::CommandState::Resumable) => crate::client::CommandState::Resumable,
        Ok(pbv1::CommandState::Succeeded) => crate::client::CommandState::Succeeded,
        Ok(pbv1::CommandState::Failed) => crate::client::CommandState::Failed,
        Ok(pbv1::CommandState::Canceled) => crate::client::CommandState::Canceled,
        Ok(pbv1::CommandState::Rejected) | Ok(pbv1::CommandState::Unspecified) | Err(_) => {
            crate::client::CommandState::Unknown
        }
    }
}

fn encode_agent_kind(value: crate::client::AgentKind) -> i32 {
    match value {
        crate::client::AgentKind::Codex => pbv1::AgentKind::Codex as i32,
        crate::client::AgentKind::ClaudeCode => pbv1::AgentKind::ClaudeCode as i32,
        crate::client::AgentKind::Shell => pbv1::AgentKind::Shell as i32,
    }
}

fn decode_agent_kind(value: i32) -> Result<crate::client::AgentKind, ErrorEnvelope> {
    match pbv1::AgentKind::try_from(value) {
        Ok(pbv1::AgentKind::Codex) => Ok(crate::client::AgentKind::Codex),
        Ok(pbv1::AgentKind::ClaudeCode) => Ok(crate::client::AgentKind::ClaudeCode),
        Ok(pbv1::AgentKind::Shell) => Ok(crate::client::AgentKind::Shell),
        Ok(pbv1::AgentKind::Unspecified) | Err(_) => Err(invalid_field(
            "agent_kind",
            format!("unknown enum value for AgentKind: {value}"),
        )),
    }
}

fn encode_agent_interface_mode(value: crate::client::AgentInterfaceMode) -> i32 {
    match value {
        crate::client::AgentInterfaceMode::ShellTmux => pbv1::AgentInterfaceMode::ShellTmux as i32,
        crate::client::AgentInterfaceMode::StructuredExec => {
            pbv1::AgentInterfaceMode::StructuredExec as i32
        }
        crate::client::AgentInterfaceMode::AppServer => pbv1::AgentInterfaceMode::AppServer as i32,
    }
}

fn decode_agent_interface_mode(
    value: i32,
) -> Result<crate::client::AgentInterfaceMode, ErrorEnvelope> {
    match pbv1::AgentInterfaceMode::try_from(value) {
        Ok(pbv1::AgentInterfaceMode::ShellTmux) => Ok(crate::client::AgentInterfaceMode::ShellTmux),
        Ok(pbv1::AgentInterfaceMode::StructuredExec) => {
            Ok(crate::client::AgentInterfaceMode::StructuredExec)
        }
        Ok(pbv1::AgentInterfaceMode::AppServer) => Ok(crate::client::AgentInterfaceMode::AppServer),
        Ok(pbv1::AgentInterfaceMode::Unspecified) | Err(_) => Err(invalid_field(
            "interface_mode",
            format!("unknown enum value for AgentInterfaceMode: {value}"),
        )),
    }
}

fn encode_agent_session_scope_kind(value: crate::client::AgentSessionScopeKind) -> i32 {
    match value {
        crate::client::AgentSessionScopeKind::Task => pbv1::AgentSessionScopeKind::Task as i32,
        crate::client::AgentSessionScopeKind::Chat => pbv1::AgentSessionScopeKind::Chat as i32,
    }
}

fn decode_agent_session_scope_kind(
    value: i32,
) -> Result<crate::client::AgentSessionScopeKind, ErrorEnvelope> {
    match pbv1::AgentSessionScopeKind::try_from(value) {
        Ok(pbv1::AgentSessionScopeKind::Task) => Ok(crate::client::AgentSessionScopeKind::Task),
        Ok(pbv1::AgentSessionScopeKind::Chat) => Ok(crate::client::AgentSessionScopeKind::Chat),
        Ok(pbv1::AgentSessionScopeKind::Unspecified) | Err(_) => Err(invalid_field(
            "scope_kind",
            format!("unknown enum value for AgentSessionScopeKind: {value}"),
        )),
    }
}

fn encode_agent_session_status(value: crate::client::AgentSessionStatus) -> i32 {
    match value {
        crate::client::AgentSessionStatus::Running => pbv1::AgentSessionStatus::Running as i32,
        crate::client::AgentSessionStatus::Blocked => pbv1::AgentSessionStatus::Blocked as i32,
        crate::client::AgentSessionStatus::Stopped => pbv1::AgentSessionStatus::Stopped as i32,
        crate::client::AgentSessionStatus::Error => pbv1::AgentSessionStatus::Error as i32,
    }
}

fn decode_agent_session_status(
    value: i32,
) -> Result<crate::client::AgentSessionStatus, ErrorEnvelope> {
    match pbv1::AgentSessionStatus::try_from(value) {
        Ok(pbv1::AgentSessionStatus::Running) => Ok(crate::client::AgentSessionStatus::Running),
        Ok(pbv1::AgentSessionStatus::Blocked) => Ok(crate::client::AgentSessionStatus::Blocked),
        Ok(pbv1::AgentSessionStatus::Stopped) => Ok(crate::client::AgentSessionStatus::Stopped),
        Ok(pbv1::AgentSessionStatus::Error) => Ok(crate::client::AgentSessionStatus::Error),
        Ok(pbv1::AgentSessionStatus::Unspecified) | Err(_) => Err(invalid_field(
            "status",
            format!("unknown enum value for AgentSessionStatus: {value}"),
        )),
    }
}

impl crate::client::ClientFrame {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ClientFrame {
        pbv1::ClientFrame {
            envelope: Some(self.envelope.to_protobuf()),
            message: Some(match &self.message {
                crate::client::ClientMessage::Request(req) => {
                    pbv1::client_frame::Message::Request(req.to_protobuf())
                }
                crate::client::ClientMessage::Response(resp) => {
                    pbv1::client_frame::Message::Response(resp.to_protobuf())
                }
                crate::client::ClientMessage::Subscribe(sub) => {
                    pbv1::client_frame::Message::Subscribe(sub.to_protobuf())
                }
                crate::client::ClientMessage::Event(ev) => {
                    pbv1::client_frame::Message::Event(ev.to_protobuf())
                }
                crate::client::ClientMessage::Unsubscribe(unsub) => {
                    pbv1::client_frame::Message::Unsubscribe(unsub.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ClientFrame) -> Result<Self, ErrorEnvelope> {
        let envelope = ProtocolEnvelope::try_from_protobuf(
            proto.envelope.ok_or_else(|| missing_required("envelope"))?,
        )?;

        let message = match proto.message.ok_or_else(|| missing_required("message"))? {
            pbv1::client_frame::Message::Request(req) => crate::client::ClientMessage::Request(
                crate::client::Request::try_from_protobuf(req)?,
            ),
            pbv1::client_frame::Message::Response(resp) => crate::client::ClientMessage::Response(
                crate::client::Response::try_from_protobuf(resp)?,
            ),
            pbv1::client_frame::Message::Subscribe(sub) => crate::client::ClientMessage::Subscribe(
                crate::client::Subscribe::try_from_protobuf(sub)?,
            ),
            pbv1::client_frame::Message::Event(ev) => {
                crate::client::ClientMessage::Event(crate::client::Event::try_from_protobuf(ev)?)
            }
            pbv1::client_frame::Message::Unsubscribe(unsub) => {
                crate::client::ClientMessage::Unsubscribe(
                    crate::client::Unsubscribe::try_from_protobuf(unsub)?,
                )
            }
        };

        Ok(Self { envelope, message })
    }
}

impl crate::client::Request {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::Request {
        pbv1::Request {
            request_id: self.request_id.to_bytes().to_vec(),
            method: encode_client_method(self.payload.method()),
            payload: Some(match &self.payload {
                crate::client::RequestPayload::Health(req) => {
                    pbv1::request::Payload::Health(req.to_protobuf())
                }
                crate::client::RequestPayload::Status(req) => {
                    pbv1::request::Payload::Status(req.to_protobuf())
                }
                crate::client::RequestPayload::ListEpics(req) => {
                    pbv1::request::Payload::ListEpics(req.to_protobuf())
                }
                crate::client::RequestPayload::GetEpicGraph(req) => {
                    pbv1::request::Payload::GetEpicGraph(req.to_protobuf())
                }
                crate::client::RequestPayload::GetSessionEvents(req) => {
                    pbv1::request::Payload::GetSessionEvents(req.to_protobuf())
                }
                crate::client::RequestPayload::GetLatestTaskSession(req) => {
                    pbv1::request::Payload::GetLatestTaskSession(req.to_protobuf())
                }
                crate::client::RequestPayload::GetEpicPinnedChatSession(req) => {
                    pbv1::request::Payload::GetEpicPinnedChatSession(req.to_protobuf())
                }
                crate::client::RequestPayload::ListTaskSessions(req) => {
                    pbv1::request::Payload::ListTaskSessions(req.to_protobuf())
                }
                crate::client::RequestPayload::CreateChatSession(req) => {
                    pbv1::request::Payload::CreateChatSession(req.to_protobuf())
                }
                crate::client::RequestPayload::CloseChatSession(req) => {
                    pbv1::request::Payload::CloseChatSession(req.to_protobuf())
                }
                crate::client::RequestPayload::ListChatSessions(req) => {
                    pbv1::request::Payload::ListChatSessions(req.to_protobuf())
                }
                crate::client::RequestPayload::PinChatSessionToEpic(req) => {
                    pbv1::request::Payload::PinChatSessionToEpic(req.to_protobuf())
                }
                crate::client::RequestPayload::UnpinChatSessionFromEpic(req) => {
                    pbv1::request::Payload::UnpinChatSessionFromEpic(req.to_protobuf())
                }
                crate::client::RequestPayload::CreateCommand(req) => {
                    pbv1::request::Payload::CreateCommand(req.to_protobuf())
                }
                crate::client::RequestPayload::GetCommand(req) => {
                    pbv1::request::Payload::GetCommand(req.to_protobuf())
                }
                crate::client::RequestPayload::WaitForCommand(req) => {
                    pbv1::request::Payload::WaitForCommand(req.to_protobuf())
                }
                crate::client::RequestPayload::WaitForEvent(req) => {
                    pbv1::request::Payload::WaitForEvent(req.to_protobuf())
                }
                crate::client::RequestPayload::WaitForIdle(req) => {
                    pbv1::request::Payload::WaitForIdle(req.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Request) -> Result<Self, ErrorEnvelope> {
        let request_id = decode_required_ulid::<RequestId>("request_id", &proto.request_id)?;
        let method = decode_client_method(proto.method)?;

        let payload = match proto.payload.ok_or_else(|| missing_required("payload"))? {
            pbv1::request::Payload::Health(req) => crate::client::RequestPayload::Health(
                crate::client::HealthRequest::from_protobuf(req),
            ),
            pbv1::request::Payload::Status(req) => crate::client::RequestPayload::Status(
                crate::client::StatusRequest::from_protobuf(req),
            ),
            pbv1::request::Payload::ListEpics(req) => crate::client::RequestPayload::ListEpics(
                crate::client::ListEpicsRequest::from_protobuf(req),
            ),
            pbv1::request::Payload::GetEpicGraph(req) => {
                crate::client::RequestPayload::GetEpicGraph(
                    crate::client::GetEpicGraphRequest::from_protobuf(req),
                )
            }
            pbv1::request::Payload::GetSessionEvents(req) => {
                crate::client::RequestPayload::GetSessionEvents(
                    crate::client::GetSessionEventsRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::GetLatestTaskSession(req) => {
                crate::client::RequestPayload::GetLatestTaskSession(
                    crate::client::GetLatestTaskSessionRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::GetEpicPinnedChatSession(req) => {
                crate::client::RequestPayload::GetEpicPinnedChatSession(
                    crate::client::GetEpicPinnedChatSessionRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::ListTaskSessions(req) => {
                crate::client::RequestPayload::ListTaskSessions(
                    crate::client::ListTaskSessionsRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::CreateChatSession(req) => {
                crate::client::RequestPayload::CreateChatSession(
                    crate::client::CreateChatSessionRequest::from_protobuf(req),
                )
            }
            pbv1::request::Payload::CloseChatSession(req) => {
                crate::client::RequestPayload::CloseChatSession(
                    crate::client::CloseChatSessionRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::ListChatSessions(req) => {
                crate::client::RequestPayload::ListChatSessions(
                    crate::client::ListChatSessionsRequest::from_protobuf(req),
                )
            }
            pbv1::request::Payload::PinChatSessionToEpic(req) => {
                crate::client::RequestPayload::PinChatSessionToEpic(
                    crate::client::PinChatSessionToEpicRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::UnpinChatSessionFromEpic(req) => {
                crate::client::RequestPayload::UnpinChatSessionFromEpic(
                    crate::client::UnpinChatSessionFromEpicRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::CreateCommand(req) => {
                crate::client::RequestPayload::CreateCommand(
                    crate::client::CreateCommandRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::GetCommand(req) => crate::client::RequestPayload::GetCommand(
                crate::client::GetCommandRequest::try_from_protobuf(req)?,
            ),
            pbv1::request::Payload::WaitForCommand(req) => {
                crate::client::RequestPayload::WaitForCommand(
                    crate::client::WaitForCommandRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::WaitForEvent(req) => {
                crate::client::RequestPayload::WaitForEvent(
                    crate::client::WaitForEventRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::request::Payload::WaitForIdle(req) => crate::client::RequestPayload::WaitForIdle(
                crate::client::WaitForIdleRequest::try_from_protobuf(req)?,
            ),
        };

        let derived_method = payload.method();
        if method != derived_method {
            return Err(invalid_field(
                "method",
                format!("method does not match payload: {method:?} vs {derived_method:?}"),
            ));
        }

        Ok(Self {
            request_id,
            payload,
        })
    }
}

impl crate::client::HealthRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::HealthRequest {
        pbv1::HealthRequest {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::HealthRequest) -> Self {
        Self {}
    }
}

impl crate::client::StatusRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::StatusRequest {
        pbv1::StatusRequest {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::StatusRequest) -> Self {
        Self {}
    }
}

impl crate::client::ListEpicsRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ListEpicsRequest {
        pbv1::ListEpicsRequest {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::ListEpicsRequest) -> Self {
        Self {}
    }
}

impl crate::client::GetEpicGraphRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetEpicGraphRequest {
        pbv1::GetEpicGraphRequest {
            epic_slug: self.epic_slug.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::GetEpicGraphRequest) -> Self {
        Self {
            epic_slug: proto.epic_slug,
        }
    }
}

impl crate::client::SessionEventCursor {
    #[must_use]
    pub fn to_protobuf(self) -> pbv1::SessionEventCursor {
        pbv1::SessionEventCursor {
            created_at: Some(encode_timestamp(self.created_at)),
            session_event_id: self.session_event_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::SessionEventCursor) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            created_at: decode_required_timestamp("created_at", proto.created_at)?,
            session_event_id: decode_required_ulid::<SessionEventId>(
                "session_event_id",
                &proto.session_event_id,
            )?,
        })
    }
}

fn encode_session_event_kind_filter(value: crate::client::SessionEventKindFilter) -> i32 {
    match value {
        crate::client::SessionEventKindFilter::SessionStarted => {
            pbv1::SessionEventKindFilter::SessionStarted as i32
        }
        crate::client::SessionEventKindFilter::SessionEnded => {
            pbv1::SessionEventKindFilter::SessionEnded as i32
        }
        crate::client::SessionEventKindFilter::TurnStarted => {
            pbv1::SessionEventKindFilter::TurnStarted as i32
        }
        crate::client::SessionEventKindFilter::TurnCompleted => {
            pbv1::SessionEventKindFilter::TurnCompleted as i32
        }
        crate::client::SessionEventKindFilter::UserMessage => {
            pbv1::SessionEventKindFilter::UserMessage as i32
        }
        crate::client::SessionEventKindFilter::AssistantMessage => {
            pbv1::SessionEventKindFilter::AssistantMessage as i32
        }
        crate::client::SessionEventKindFilter::ToolInvocation => {
            pbv1::SessionEventKindFilter::ToolInvocation as i32
        }
        crate::client::SessionEventKindFilter::ToolResult => {
            pbv1::SessionEventKindFilter::ToolResult as i32
        }
        crate::client::SessionEventKindFilter::StatusUpdate => {
            pbv1::SessionEventKindFilter::StatusUpdate as i32
        }
        crate::client::SessionEventKindFilter::ArtifactEmitted => {
            pbv1::SessionEventKindFilter::ArtifactEmitted as i32
        }
        crate::client::SessionEventKindFilter::Unknown => {
            pbv1::SessionEventKindFilter::Unspecified as i32
        }
    }
}

fn decode_session_event_kind_filter(value: i32) -> crate::client::SessionEventKindFilter {
    match pbv1::SessionEventKindFilter::try_from(value) {
        Ok(pbv1::SessionEventKindFilter::SessionStarted) => {
            crate::client::SessionEventKindFilter::SessionStarted
        }
        Ok(pbv1::SessionEventKindFilter::SessionEnded) => {
            crate::client::SessionEventKindFilter::SessionEnded
        }
        Ok(pbv1::SessionEventKindFilter::TurnStarted) => {
            crate::client::SessionEventKindFilter::TurnStarted
        }
        Ok(pbv1::SessionEventKindFilter::TurnCompleted) => {
            crate::client::SessionEventKindFilter::TurnCompleted
        }
        Ok(pbv1::SessionEventKindFilter::UserMessage) => {
            crate::client::SessionEventKindFilter::UserMessage
        }
        Ok(pbv1::SessionEventKindFilter::AssistantMessage) => {
            crate::client::SessionEventKindFilter::AssistantMessage
        }
        Ok(pbv1::SessionEventKindFilter::ToolInvocation) => {
            crate::client::SessionEventKindFilter::ToolInvocation
        }
        Ok(pbv1::SessionEventKindFilter::ToolResult) => {
            crate::client::SessionEventKindFilter::ToolResult
        }
        Ok(pbv1::SessionEventKindFilter::StatusUpdate) => {
            crate::client::SessionEventKindFilter::StatusUpdate
        }
        Ok(pbv1::SessionEventKindFilter::ArtifactEmitted) => {
            crate::client::SessionEventKindFilter::ArtifactEmitted
        }
        Ok(pbv1::SessionEventKindFilter::Unspecified) | Err(_) => {
            crate::client::SessionEventKindFilter::Unknown
        }
    }
}

impl crate::client::GetSessionEventsRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetSessionEventsRequest {
        pbv1::GetSessionEventsRequest {
            session_id: self.session_id.to_bytes().to_vec(),
            before: self
                .before
                .map(crate::client::SessionEventCursor::to_protobuf),
            limit: self.limit,
            kinds: self
                .kinds
                .iter()
                .copied()
                .map(encode_session_event_kind_filter)
                .collect(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::GetSessionEventsRequest) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid::<SessionId>("session_id", &proto.session_id)?,
            before: proto
                .before
                .map(crate::client::SessionEventCursor::try_from_protobuf)
                .transpose()?,
            limit: proto.limit,
            kinds: proto
                .kinds
                .into_iter()
                .map(decode_session_event_kind_filter)
                .collect(),
        })
    }
}

impl crate::client::CreateCommandRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CreateCommandRequest {
        pbv1::CreateCommandRequest {
            kind: self.kind.clone(),
            target_task_id: self
                .target_task_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CreateCommandRequest) -> Result<Self, ErrorEnvelope> {
        if proto.kind.is_empty() {
            return Err(missing_required("kind"));
        }

        Ok(Self {
            kind: proto.kind,
            target_task_id: decode_optional_ulid::<TaskId>(
                "target_task_id",
                &proto.target_task_id,
            )?,
        })
    }
}

impl crate::client::CreateCommandResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CreateCommandResponse {
        pbv1::CreateCommandResponse {
            command: Some(self.command.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CreateCommandResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            command: crate::client::CommandSummary::try_from_protobuf(
                proto.command.ok_or_else(|| missing_required("command"))?,
            )?,
        })
    }
}

impl crate::client::GetCommandRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetCommandRequest {
        pbv1::GetCommandRequest {
            command_id: self.command_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::GetCommandRequest) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            command_id: decode_required_ulid("command_id", &proto.command_id)?,
        })
    }
}

impl crate::client::GetCommandResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetCommandResponse {
        pbv1::GetCommandResponse {
            command: Some(self.command.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::GetCommandResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            command: crate::client::CommandSummary::try_from_protobuf(
                proto.command.ok_or_else(|| missing_required("command"))?,
            )?,
        })
    }
}

impl crate::client::WaitForCommandRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForCommandRequest {
        pbv1::WaitForCommandRequest {
            command_id: self.command_id.to_bytes().to_vec(),
            terminal_states: self
                .terminal_states
                .iter()
                .copied()
                .map(encode_client_command_state)
                .collect(),
            timeout_ms: self.timeout_ms,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::WaitForCommandRequest) -> Result<Self, ErrorEnvelope> {
        let terminal_states = proto
            .terminal_states
            .into_iter()
            .map(decode_client_command_state)
            .collect();

        Ok(Self {
            command_id: decode_required_ulid("command_id", &proto.command_id)?,
            terminal_states,
            timeout_ms: proto.timeout_ms,
        })
    }
}

impl crate::client::GetLatestTaskSessionRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetLatestTaskSessionRequest {
        pbv1::GetLatestTaskSessionRequest {
            task_id: self.task_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::GetLatestTaskSessionRequest,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            task_id: decode_required_ulid::<TaskId>("task_id", &proto.task_id)?,
        })
    }
}

impl crate::client::WaitForCommandResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForCommandResponse {
        pbv1::WaitForCommandResponse {
            command: Some(self.command.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::WaitForCommandResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            command: crate::client::CommandSummary::try_from_protobuf(
                proto.command.ok_or_else(|| missing_required("command"))?,
            )?,
        })
    }
}

impl crate::client::GetEpicPinnedChatSessionRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetEpicPinnedChatSessionRequest {
        pbv1::GetEpicPinnedChatSessionRequest {
            epic_id: self.epic_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::GetEpicPinnedChatSessionRequest,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            epic_id: decode_required_ulid::<EpicId>("epic_id", &proto.epic_id)?,
        })
    }
}

impl crate::client::EventWaitFilter {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::EventWaitFilter {
        pbv1::EventWaitFilter {
            event_type_prefix: self.event_type_prefix.clone(),
            after_event_id: self
                .after_event_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::EventWaitFilter) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            event_type_prefix: proto.event_type_prefix,
            after_event_id: decode_optional_ulid("after_event_id", &proto.after_event_id)?,
        })
    }
}

impl crate::client::WaitForEventRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForEventRequest {
        pbv1::WaitForEventRequest {
            filter: Some(self.filter.to_protobuf()),
            timeout_ms: self.timeout_ms,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::WaitForEventRequest) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            filter: crate::client::EventWaitFilter::try_from_protobuf(
                proto.filter.ok_or_else(|| missing_required("filter"))?,
            )?,
            timeout_ms: proto.timeout_ms,
        })
    }
}

impl crate::client::WaitForEventResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForEventResponse {
        pbv1::WaitForEventResponse {
            event_log: Some(self.event_log.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::WaitForEventResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            event_log: crate::client::EventLogEvent::try_from_protobuf(
                proto
                    .event_log
                    .ok_or_else(|| missing_required("event_log"))?,
            )?,
        })
    }
}

impl crate::client::WaitForIdleRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForIdleRequest {
        pbv1::WaitForIdleRequest {
            scope: self.scope.map(|scope| scope.to_protobuf()),
            timeout_ms: self.timeout_ms,
            quiescence_ms: self.quiescence_ms,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::WaitForIdleRequest) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            scope: proto.scope.map(Scope::try_from_protobuf).transpose()?,
            timeout_ms: proto.timeout_ms,
            quiescence_ms: proto.quiescence_ms,
        })
    }
}

impl crate::client::WaitForIdleResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForIdleResponse {
        pbv1::WaitForIdleResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::WaitForIdleResponse) -> Self {
        Self {}
    }
}

impl crate::client::Response {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::Response {
        pbv1::Response {
            request_id: self.request_id.to_bytes().to_vec(),
            status: encode_response_status(self.status()),
            result: Some(match &self.result {
                crate::client::ResponseResult::Health(resp) => {
                    pbv1::response::Result::Health(resp.to_protobuf())
                }
                crate::client::ResponseResult::Status(resp) => {
                    pbv1::response::Result::StatusResponse(resp.to_protobuf())
                }
                crate::client::ResponseResult::ListEpics(resp) => {
                    pbv1::response::Result::ListEpics(resp.to_protobuf())
                }
                crate::client::ResponseResult::GetEpicGraph(resp) => {
                    pbv1::response::Result::GetEpicGraph(resp.to_protobuf())
                }
                crate::client::ResponseResult::GetSessionEvents(resp) => {
                    pbv1::response::Result::GetSessionEvents(resp.to_protobuf())
                }
                crate::client::ResponseResult::GetLatestTaskSession(resp) => {
                    pbv1::response::Result::GetLatestTaskSession(resp.to_protobuf())
                }
                crate::client::ResponseResult::GetEpicPinnedChatSession(resp) => {
                    pbv1::response::Result::GetEpicPinnedChatSession(resp.to_protobuf())
                }
                crate::client::ResponseResult::ListTaskSessions(resp) => {
                    pbv1::response::Result::ListTaskSessions(resp.to_protobuf())
                }
                crate::client::ResponseResult::CreateChatSession(resp) => {
                    pbv1::response::Result::CreateChatSession(resp.to_protobuf())
                }
                crate::client::ResponseResult::CloseChatSession(resp) => {
                    pbv1::response::Result::CloseChatSession(resp.to_protobuf())
                }
                crate::client::ResponseResult::ListChatSessions(resp) => {
                    pbv1::response::Result::ListChatSessions(resp.to_protobuf())
                }
                crate::client::ResponseResult::PinChatSessionToEpic(resp) => {
                    pbv1::response::Result::PinChatSessionToEpic(resp.to_protobuf())
                }
                crate::client::ResponseResult::UnpinChatSessionFromEpic(resp) => {
                    pbv1::response::Result::UnpinChatSessionFromEpic(resp.to_protobuf())
                }
                crate::client::ResponseResult::CreateCommand(resp) => {
                    pbv1::response::Result::CreateCommand(resp.to_protobuf())
                }
                crate::client::ResponseResult::GetCommand(resp) => {
                    pbv1::response::Result::GetCommand(resp.to_protobuf())
                }
                crate::client::ResponseResult::WaitForCommand(resp) => {
                    pbv1::response::Result::WaitForCommand(resp.to_protobuf())
                }
                crate::client::ResponseResult::WaitForEvent(resp) => {
                    pbv1::response::Result::WaitForEvent(resp.to_protobuf())
                }
                crate::client::ResponseResult::WaitForIdle(resp) => {
                    pbv1::response::Result::WaitForIdle(resp.to_protobuf())
                }
                crate::client::ResponseResult::Error(err) => {
                    pbv1::response::Result::Error(err.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Response) -> Result<Self, ErrorEnvelope> {
        let request_id = decode_required_ulid::<RequestId>("request_id", &proto.request_id)?;
        let status = decode_response_status(proto.status)?;

        let result = match proto.result.ok_or_else(|| missing_required("result"))? {
            pbv1::response::Result::Health(resp) => crate::client::ResponseResult::Health(
                crate::client::HealthResponse::from_protobuf(resp),
            ),
            pbv1::response::Result::StatusResponse(resp) => crate::client::ResponseResult::Status(
                crate::client::StatusResponse::try_from_protobuf(resp)?,
            ),
            pbv1::response::Result::ListEpics(resp) => crate::client::ResponseResult::ListEpics(
                crate::client::ListEpicsResponse::from_protobuf(resp),
            ),
            pbv1::response::Result::GetEpicGraph(resp) => {
                crate::client::ResponseResult::GetEpicGraph(
                    crate::client::GetEpicGraphResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::GetSessionEvents(resp) => {
                crate::client::ResponseResult::GetSessionEvents(
                    crate::client::GetSessionEventsResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::GetLatestTaskSession(resp) => {
                crate::client::ResponseResult::GetLatestTaskSession(
                    crate::client::GetLatestTaskSessionResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::GetEpicPinnedChatSession(resp) => {
                crate::client::ResponseResult::GetEpicPinnedChatSession(
                    crate::client::GetEpicPinnedChatSessionResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::ListTaskSessions(resp) => {
                crate::client::ResponseResult::ListTaskSessions(
                    crate::client::ListTaskSessionsResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::CreateChatSession(resp) => {
                crate::client::ResponseResult::CreateChatSession(
                    crate::client::CreateChatSessionResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::CloseChatSession(resp) => {
                crate::client::ResponseResult::CloseChatSession(
                    crate::client::CloseChatSessionResponse::from_protobuf(resp),
                )
            }
            pbv1::response::Result::ListChatSessions(resp) => {
                crate::client::ResponseResult::ListChatSessions(
                    crate::client::ListChatSessionsResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::PinChatSessionToEpic(resp) => {
                crate::client::ResponseResult::PinChatSessionToEpic(
                    crate::client::PinChatSessionToEpicResponse::from_protobuf(resp),
                )
            }
            pbv1::response::Result::UnpinChatSessionFromEpic(resp) => {
                crate::client::ResponseResult::UnpinChatSessionFromEpic(
                    crate::client::UnpinChatSessionFromEpicResponse::from_protobuf(resp),
                )
            }
            pbv1::response::Result::CreateCommand(resp) => {
                crate::client::ResponseResult::CreateCommand(
                    crate::client::CreateCommandResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::GetCommand(resp) => crate::client::ResponseResult::GetCommand(
                crate::client::GetCommandResponse::try_from_protobuf(resp)?,
            ),
            pbv1::response::Result::WaitForCommand(resp) => {
                crate::client::ResponseResult::WaitForCommand(
                    crate::client::WaitForCommandResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::WaitForEvent(resp) => {
                crate::client::ResponseResult::WaitForEvent(
                    crate::client::WaitForEventResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::response::Result::WaitForIdle(resp) => {
                crate::client::ResponseResult::WaitForIdle(
                    crate::client::WaitForIdleResponse::from_protobuf(resp),
                )
            }
            pbv1::response::Result::Error(err) => {
                crate::client::ResponseResult::Error(ErrorEnvelope::from_protobuf(err))
            }
        };

        let derived_status = match &result {
            crate::client::ResponseResult::Error(_) => crate::client::ResponseStatus::Error,
            _ => crate::client::ResponseStatus::Ok,
        };

        if status != derived_status {
            return Err(invalid_field(
                "status",
                format!("status does not match result: {status:?} vs {derived_status:?}"),
            ));
        }

        Ok(Self { request_id, result })
    }
}

impl crate::client::HealthResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::HealthResponse {
        pbv1::HealthResponse { ok: self.ok }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::HealthResponse) -> Self {
        Self { ok: proto.ok }
    }
}

impl crate::client::StatusResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::StatusResponse {
        pbv1::StatusResponse {
            accepted_protocol: Some(self.accepted_protocol.to_protobuf()),
            server_name: self.server_name.clone(),
            server_version: self.server_version.clone().unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::StatusResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            accepted_protocol: ProtocolVersion::try_from_protobuf(
                proto
                    .accepted_protocol
                    .ok_or_else(|| missing_required("accepted_protocol"))?,
            )?,
            server_name: proto.server_name,
            server_version: if proto.server_version.is_empty() {
                None
            } else {
                Some(proto.server_version)
            },
        })
    }
}

impl crate::client::EpicSummary {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::EpicSummary {
        pbv1::EpicSummary {
            slug: self.slug.clone(),
            name: self.name.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::EpicSummary) -> Self {
        Self {
            slug: proto.slug,
            name: proto.name,
        }
    }
}

impl crate::client::ListEpicsResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ListEpicsResponse {
        pbv1::ListEpicsResponse {
            epics: self
                .epics
                .iter()
                .map(crate::client::EpicSummary::to_protobuf)
                .collect(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::ListEpicsResponse) -> Self {
        Self {
            epics: proto
                .epics
                .into_iter()
                .map(crate::client::EpicSummary::from_protobuf)
                .collect(),
        }
    }
}

impl crate::client::EpicTaskNode {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::EpicTaskNode {
        pbv1::EpicTaskNode {
            task_slug: self.task_slug.clone(),
            title: self.title.clone(),
            task_id: self
                .task_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            parent_task_id: self
                .parent_task_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            state: encode_task_state(self.state),
            branch_name: self.branch_name.clone().unwrap_or_default(),
            merge_readiness: encode_merge_readiness(self.merge_readiness),
        }
    }

    pub fn from_protobuf(proto: pbv1::EpicTaskNode) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            task_slug: proto.task_slug,
            title: proto.title,
            task_id: decode_optional_ulid::<TaskId>("task_id", &proto.task_id)?,
            parent_task_id: decode_optional_ulid::<TaskId>(
                "parent_task_id",
                &proto.parent_task_id,
            )?,
            state: decode_task_state(proto.state),
            branch_name: if proto.branch_name.is_empty() {
                None
            } else {
                Some(proto.branch_name)
            },
            merge_readiness: decode_merge_readiness(proto.merge_readiness),
        })
    }
}

impl crate::client::EpicTaskEdge {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::EpicTaskEdge {
        pbv1::EpicTaskEdge {
            from_task_slug: self.from_task_slug.clone(),
            to_task_slug: self.to_task_slug.clone(),
            from_task_id: self
                .from_task_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            to_task_id: self
                .to_task_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn from_protobuf(proto: pbv1::EpicTaskEdge) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            from_task_slug: proto.from_task_slug,
            to_task_slug: proto.to_task_slug,
            from_task_id: decode_optional_ulid::<TaskId>("from_task_id", &proto.from_task_id)?,
            to_task_id: decode_optional_ulid::<TaskId>("to_task_id", &proto.to_task_id)?,
        })
    }
}

impl crate::client::CommandUpdateSummary {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CommandUpdateSummary {
        pbv1::CommandUpdateSummary {
            update_id: self.update_id.to_bytes().to_vec(),
            created_at: Some(encode_timestamp(self.created_at)),
            state: encode_client_command_state(self.state),
            message: self.message.clone().unwrap_or_default(),
            progress_current: self.progress_current,
            progress_total: self.progress_total,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CommandUpdateSummary) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            update_id: decode_required_ulid::<CommandUpdateId>("update_id", &proto.update_id)?,
            created_at: decode_required_timestamp("created_at", proto.created_at)?,
            state: decode_client_command_state(proto.state),
            message: if proto.message.is_empty() {
                None
            } else {
                Some(proto.message)
            },
            progress_current: proto.progress_current,
            progress_total: proto.progress_total,
        })
    }
}

impl crate::client::CommandSummary {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CommandSummary {
        pbv1::CommandSummary {
            command_id: self.command_id.to_bytes().to_vec(),
            created_at: Some(encode_timestamp(self.created_at)),
            updated_at: Some(encode_timestamp(self.updated_at)),
            kind: self.kind.clone(),
            state: encode_client_command_state(self.state),
            target_task_id: self
                .target_task_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            last_update: self.last_update.as_ref().map(|update| update.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CommandSummary) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            command_id: decode_required_ulid::<CommandId>("command_id", &proto.command_id)?,
            created_at: decode_required_timestamp("created_at", proto.created_at)?,
            updated_at: decode_required_timestamp("updated_at", proto.updated_at)?,
            kind: proto.kind,
            state: decode_client_command_state(proto.state),
            target_task_id: decode_optional_ulid::<TaskId>(
                "target_task_id",
                &proto.target_task_id,
            )?,
            last_update: proto
                .last_update
                .map(crate::client::CommandUpdateSummary::try_from_protobuf)
                .transpose()?,
        })
    }
}

impl crate::client::DaemonPresenceSummary {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::DaemonPresenceSummary {
        pbv1::DaemonPresenceSummary {
            host_instance_id: self.host_instance_id.to_bytes().to_vec(),
            host_id: self.host_id.to_bytes().to_vec(),
            hostname: self.hostname.clone().unwrap_or_default(),
            connected_at: Some(encode_timestamp(self.connected_at)),
            last_heartbeat_at: Some(encode_timestamp(self.last_heartbeat_at)),
            disconnected_at: self.disconnected_at.map(encode_timestamp),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::DaemonPresenceSummary) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            host_instance_id: decode_required_ulid::<HostInstanceId>(
                "host_instance_id",
                &proto.host_instance_id,
            )?,
            host_id: decode_required_ulid::<HostId>("host_id", &proto.host_id)?,
            hostname: if proto.hostname.is_empty() {
                None
            } else {
                Some(proto.hostname)
            },
            connected_at: decode_required_timestamp("connected_at", proto.connected_at)?,
            last_heartbeat_at: decode_required_timestamp(
                "last_heartbeat_at",
                proto.last_heartbeat_at,
            )?,
            disconnected_at: proto
                .disconnected_at
                .map(|ts| decode_timestamp("disconnected_at", ts))
                .transpose()?,
        })
    }
}

impl crate::client::SessionSummary {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::SessionSummary {
        pbv1::SessionSummary {
            session_id: self.session_id.to_bytes().to_vec(),
            session_event_id: self.session_event_id.to_bytes().to_vec(),
            task_id: self.task_id.to_bytes().to_vec(),
            last_event_at: Some(encode_timestamp(self.last_event_at)),
            kind: self.kind.clone(),
            turn_id: self.turn_id.clone().unwrap_or_default(),
            message_preview: self.message_preview.clone().unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::SessionSummary) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid::<SessionId>("session_id", &proto.session_id)?,
            session_event_id: decode_required_ulid::<SessionEventId>(
                "session_event_id",
                &proto.session_event_id,
            )?,
            task_id: decode_required_ulid::<TaskId>("task_id", &proto.task_id)?,
            last_event_at: decode_required_timestamp("last_event_at", proto.last_event_at)?,
            kind: proto.kind,
            turn_id: if proto.turn_id.is_empty() {
                None
            } else {
                Some(proto.turn_id)
            },
            message_preview: if proto.message_preview.is_empty() {
                None
            } else {
                Some(proto.message_preview)
            },
        })
    }
}

impl crate::client::EpicGraph {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::EpicGraph {
        pbv1::EpicGraph {
            epic_slug: self.epic_slug.clone(),
            nodes: self
                .nodes
                .iter()
                .map(crate::client::EpicTaskNode::to_protobuf)
                .collect(),
            edges: self
                .edges
                .iter()
                .map(crate::client::EpicTaskEdge::to_protobuf)
                .collect(),
            epic_id: self
                .epic_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            epic_title: self.epic_title.clone().unwrap_or_default(),
            workspace_id: self
                .workspace_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            repo_id: self
                .repo_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            command_summaries: self
                .command_summaries
                .iter()
                .map(crate::client::CommandSummary::to_protobuf)
                .collect(),
            daemon_presences: self
                .daemon_presences
                .iter()
                .map(crate::client::DaemonPresenceSummary::to_protobuf)
                .collect(),
            session_summaries: self
                .session_summaries
                .iter()
                .map(crate::client::SessionSummary::to_protobuf)
                .collect(),
            as_of_event_id: self
                .as_of_event_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::EpicGraph) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            epic_slug: proto.epic_slug,
            nodes: proto
                .nodes
                .into_iter()
                .map(crate::client::EpicTaskNode::from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            edges: proto
                .edges
                .into_iter()
                .map(crate::client::EpicTaskEdge::from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            epic_id: decode_optional_ulid::<EpicId>("epic_id", &proto.epic_id)?,
            epic_title: if proto.epic_title.is_empty() {
                None
            } else {
                Some(proto.epic_title)
            },
            workspace_id: decode_optional_ulid::<WorkspaceId>("workspace_id", &proto.workspace_id)?,
            repo_id: decode_optional_ulid::<RepoId>("repo_id", &proto.repo_id)?,
            command_summaries: proto
                .command_summaries
                .into_iter()
                .map(crate::client::CommandSummary::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            daemon_presences: proto
                .daemon_presences
                .into_iter()
                .map(crate::client::DaemonPresenceSummary::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            session_summaries: proto
                .session_summaries
                .into_iter()
                .map(crate::client::SessionSummary::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            as_of_event_id: decode_optional_ulid::<EventId>(
                "as_of_event_id",
                &proto.as_of_event_id,
            )?,
        })
    }
}

impl crate::client::GetEpicGraphResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetEpicGraphResponse {
        pbv1::GetEpicGraphResponse {
            graph: Some(self.graph.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::GetEpicGraphResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            graph: crate::client::EpicGraph::try_from_protobuf(
                proto.graph.ok_or_else(|| missing_required("graph"))?,
            )?,
        })
    }
}

impl crate::client::GetSessionEventsResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetSessionEventsResponse {
        pbv1::GetSessionEventsResponse {
            events: self.events.iter().map(SessionEvent::to_protobuf).collect(),
            next_cursor: self
                .next_cursor
                .map(crate::client::SessionEventCursor::to_protobuf),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::GetSessionEventsResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            events: proto
                .events
                .into_iter()
                .map(SessionEvent::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            next_cursor: proto
                .next_cursor
                .map(crate::client::SessionEventCursor::try_from_protobuf)
                .transpose()?,
        })
    }
}

impl crate::client::GetLatestTaskSessionResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetLatestTaskSessionResponse {
        pbv1::GetLatestTaskSessionResponse {
            session_id: self
                .session_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::GetLatestTaskSessionResponse,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_optional_ulid::<SessionId>("session_id", &proto.session_id)?,
        })
    }
}

impl crate::client::GetEpicPinnedChatSessionResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetEpicPinnedChatSessionResponse {
        pbv1::GetEpicPinnedChatSessionResponse {
            session_id: self
                .session_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::GetEpicPinnedChatSessionResponse,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_optional_ulid::<SessionId>("session_id", &proto.session_id)?,
        })
    }
}

impl crate::client::AgentSessionSummary {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::AgentSessionSummary {
        pbv1::AgentSessionSummary {
            session_id: self.session_id.to_bytes().to_vec(),
            scope_kind: encode_agent_session_scope_kind(self.scope_kind),
            task_id: self
                .task_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            agent_kind: encode_agent_kind(self.agent_kind),
            interface_mode: encode_agent_interface_mode(self.interface_mode),
            status: encode_agent_session_status(self.status),
            title: self.title.clone().unwrap_or_default(),
            closed_at: self.closed_at.map(encode_timestamp),
            created_at: self.created_at.map(encode_timestamp),
            updated_at: self.updated_at.map(encode_timestamp),
            ended_at: self.ended_at.map(encode_timestamp),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::AgentSessionSummary) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid::<SessionId>("session_id", &proto.session_id)?,
            scope_kind: decode_agent_session_scope_kind(proto.scope_kind)?,
            task_id: decode_optional_ulid::<TaskId>("task_id", &proto.task_id)?,
            agent_kind: decode_agent_kind(proto.agent_kind)?,
            interface_mode: decode_agent_interface_mode(proto.interface_mode)?,
            status: decode_agent_session_status(proto.status)?,
            title: normalize_nonempty_string(proto.title),
            closed_at: proto
                .closed_at
                .map(|ts| decode_timestamp("closed_at", ts))
                .transpose()?,
            created_at: proto
                .created_at
                .map(|ts| decode_timestamp("created_at", ts))
                .transpose()?,
            updated_at: proto
                .updated_at
                .map(|ts| decode_timestamp("updated_at", ts))
                .transpose()?,
            ended_at: proto
                .ended_at
                .map(|ts| decode_timestamp("ended_at", ts))
                .transpose()?,
        })
    }
}

impl crate::client::ListTaskSessionsRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ListTaskSessionsRequest {
        pbv1::ListTaskSessionsRequest {
            task_id: self.task_id.to_bytes().to_vec(),
            limit: self.limit,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ListTaskSessionsRequest) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            task_id: decode_required_ulid::<TaskId>("task_id", &proto.task_id)?,
            limit: proto.limit,
        })
    }
}

impl crate::client::ListTaskSessionsResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ListTaskSessionsResponse {
        pbv1::ListTaskSessionsResponse {
            sessions: self
                .sessions
                .iter()
                .map(crate::client::AgentSessionSummary::to_protobuf)
                .collect(),
            active_session_id: self
                .active_session_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ListTaskSessionsResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            sessions: proto
                .sessions
                .into_iter()
                .map(crate::client::AgentSessionSummary::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            active_session_id: decode_optional_ulid::<SessionId>(
                "active_session_id",
                &proto.active_session_id,
            )?,
        })
    }
}

impl crate::client::CreateChatSessionRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CreateChatSessionRequest {
        pbv1::CreateChatSessionRequest {
            title: self.title.clone().unwrap_or_default(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::CreateChatSessionRequest) -> Self {
        Self {
            title: normalize_nonempty_string(proto.title),
        }
    }
}

impl crate::client::CreateChatSessionResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CreateChatSessionResponse {
        pbv1::CreateChatSessionResponse {
            session_id: self.session_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::CreateChatSessionResponse,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid::<SessionId>("session_id", &proto.session_id)?,
        })
    }
}

impl crate::client::CloseChatSessionRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CloseChatSessionRequest {
        pbv1::CloseChatSessionRequest {
            session_id: self.session_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CloseChatSessionRequest) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid::<SessionId>("session_id", &proto.session_id)?,
        })
    }
}

impl crate::client::CloseChatSessionResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CloseChatSessionResponse {
        pbv1::CloseChatSessionResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::CloseChatSessionResponse) -> Self {
        Self {}
    }
}

impl crate::client::ListChatSessionsRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ListChatSessionsRequest {
        pbv1::ListChatSessionsRequest {
            include_closed: self.include_closed,
            limit: self.limit,
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::ListChatSessionsRequest) -> Self {
        Self {
            include_closed: proto.include_closed,
            limit: proto.limit,
        }
    }
}

impl crate::client::ListChatSessionsResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::ListChatSessionsResponse {
        pbv1::ListChatSessionsResponse {
            sessions: self
                .sessions
                .iter()
                .map(crate::client::AgentSessionSummary::to_protobuf)
                .collect(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::ListChatSessionsResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            sessions: proto
                .sessions
                .into_iter()
                .map(crate::client::AgentSessionSummary::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl crate::client::PinChatSessionToEpicRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::PinChatSessionToEpicRequest {
        pbv1::PinChatSessionToEpicRequest {
            epic_id: self.epic_id.to_bytes().to_vec(),
            session_id: self.session_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::PinChatSessionToEpicRequest,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            epic_id: decode_required_ulid::<EpicId>("epic_id", &proto.epic_id)?,
            session_id: decode_required_ulid::<SessionId>("session_id", &proto.session_id)?,
        })
    }
}

impl crate::client::PinChatSessionToEpicResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::PinChatSessionToEpicResponse {
        pbv1::PinChatSessionToEpicResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::PinChatSessionToEpicResponse) -> Self {
        Self {}
    }
}

impl crate::client::UnpinChatSessionFromEpicRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UnpinChatSessionFromEpicRequest {
        pbv1::UnpinChatSessionFromEpicRequest {
            epic_id: self.epic_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::UnpinChatSessionFromEpicRequest,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            epic_id: decode_required_ulid::<EpicId>("epic_id", &proto.epic_id)?,
        })
    }
}

impl crate::client::UnpinChatSessionFromEpicResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UnpinChatSessionFromEpicResponse {
        pbv1::UnpinChatSessionFromEpicResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::UnpinChatSessionFromEpicResponse) -> Self {
        Self {}
    }
}

impl crate::client::Subscribe {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::Subscribe {
        pbv1::Subscribe {
            subscription_id: self.subscription_id.to_bytes().to_vec(),
            topic: encode_subscription_topic(self.filter.topic()),
            filter: Some(match &self.filter {
                crate::client::SubscriptionFilter::EventLog(filter) => {
                    pbv1::subscribe::Filter::EventLog(filter.to_protobuf())
                }
                crate::client::SubscriptionFilter::SessionEvents(filter) => {
                    pbv1::subscribe::Filter::SessionEvents(filter.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Subscribe) -> Result<Self, ErrorEnvelope> {
        let subscription_id =
            decode_required_ulid::<SubscriptionId>("subscription_id", &proto.subscription_id)?;
        let topic = decode_subscription_topic(proto.topic)?;

        let filter = match proto.filter.ok_or_else(|| missing_required("filter"))? {
            pbv1::subscribe::Filter::EventLog(filter) => {
                crate::client::SubscriptionFilter::EventLog(
                    crate::client::EventLogFilter::from_protobuf(filter)?,
                )
            }
            pbv1::subscribe::Filter::SessionEvents(filter) => {
                crate::client::SubscriptionFilter::SessionEvents(
                    crate::client::SessionEventsFilter::try_from_protobuf(filter)?,
                )
            }
        };

        let derived_topic = filter.topic();
        if topic != derived_topic {
            return Err(invalid_field(
                "topic",
                format!("topic does not match filter: {topic:?} vs {derived_topic:?}"),
            ));
        }

        Ok(Self {
            subscription_id,
            filter,
        })
    }
}

impl crate::client::EventLogFilter {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::EventLogFilter {
        pbv1::EventLogFilter {
            after_event_id: self
                .after_event_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn from_protobuf(proto: pbv1::EventLogFilter) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            after_event_id: decode_optional_ulid::<EventId>(
                "after_event_id",
                &proto.after_event_id,
            )?,
        })
    }
}

impl crate::client::SessionEventsFilter {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::SessionEventsFilter {
        pbv1::SessionEventsFilter {
            session_id: self.session_id.to_bytes().to_vec(),
            after: self
                .after
                .map(crate::client::SessionEventCursor::to_protobuf),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::SessionEventsFilter) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid::<SessionId>("session_id", &proto.session_id)?,
            after: proto
                .after
                .map(crate::client::SessionEventCursor::try_from_protobuf)
                .transpose()?,
        })
    }
}

impl crate::client::Unsubscribe {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::Unsubscribe {
        pbv1::Unsubscribe {
            subscription_id: self.subscription_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Unsubscribe) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            subscription_id: decode_required_ulid::<SubscriptionId>(
                "subscription_id",
                &proto.subscription_id,
            )?,
        })
    }
}

impl crate::client::Subscribed {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::Subscribed {
        pbv1::Subscribed {
            topic: encode_subscription_topic(self.topic),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Subscribed) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            topic: decode_subscription_topic(proto.topic)?,
        })
    }
}

impl crate::client::EventLogEvent {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::EventLogEvent {
        pbv1::EventLogEvent {
            event_id: self.event_id.to_bytes().to_vec(),
            occurred_at: Some(encode_timestamp(self.occurred_at)),
            event_type: self.event_type.clone(),
            json_payload: self.json_payload.clone(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::EventLogEvent) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            event_id: decode_required_ulid::<EventId>("event_id", &proto.event_id)?,
            occurred_at: decode_required_timestamp("occurred_at", proto.occurred_at)?,
            event_type: proto.event_type,
            json_payload: proto.json_payload,
        })
    }
}

impl crate::client::Event {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::Event {
        pbv1::Event {
            subscription_id: self.subscription_id.to_bytes().to_vec(),
            event: Some(match &self.event {
                crate::client::SubscriptionEvent::Subscribed(ev) => {
                    pbv1::event::Event::Subscribed(ev.to_protobuf())
                }
                crate::client::SubscriptionEvent::EventLog(ev) => {
                    pbv1::event::Event::EventLog(ev.to_protobuf())
                }
                crate::client::SubscriptionEvent::SessionEvent(ev) => {
                    pbv1::event::Event::SessionEvent(ev.to_protobuf())
                }
                crate::client::SubscriptionEvent::Error(err) => {
                    pbv1::event::Event::Error(err.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::Event) -> Result<Self, ErrorEnvelope> {
        let subscription_id =
            decode_required_ulid::<SubscriptionId>("subscription_id", &proto.subscription_id)?;
        let event = match proto.event.ok_or_else(|| missing_required("event"))? {
            pbv1::event::Event::Subscribed(ev) => crate::client::SubscriptionEvent::Subscribed(
                crate::client::Subscribed::try_from_protobuf(ev)?,
            ),
            pbv1::event::Event::EventLog(ev) => crate::client::SubscriptionEvent::EventLog(
                crate::client::EventLogEvent::try_from_protobuf(ev)?,
            ),
            pbv1::event::Event::SessionEvent(ev) => {
                crate::client::SubscriptionEvent::SessionEvent(SessionEvent::try_from_protobuf(ev)?)
            }
            pbv1::event::Event::Error(err) => {
                crate::client::SubscriptionEvent::Error(ErrorEnvelope::from_protobuf(err))
            }
        };

        Ok(Self {
            subscription_id,
            event,
        })
    }
}

fn encode_ui_driver_method(value: crate::ui_driver::UiDriverMethod) -> i32 {
    match value {
        crate::ui_driver::UiDriverMethod::GetSnapshot => pbv1::UiDriverMethod::GetSnapshot as i32,
        crate::ui_driver::UiDriverMethod::OpenEpic => pbv1::UiDriverMethod::OpenEpic as i32,
        crate::ui_driver::UiDriverMethod::SelectTask => pbv1::UiDriverMethod::SelectTask as i32,
        crate::ui_driver::UiDriverMethod::TriggerMerge => pbv1::UiDriverMethod::TriggerMerge as i32,
        crate::ui_driver::UiDriverMethod::OpenSessionView => {
            pbv1::UiDriverMethod::OpenSessionView as i32
        }
        crate::ui_driver::UiDriverMethod::OpenDiffView => pbv1::UiDriverMethod::OpenDiffView as i32,
        crate::ui_driver::UiDriverMethod::CaptureScreenshot => {
            pbv1::UiDriverMethod::CaptureScreenshot as i32
        }
        crate::ui_driver::UiDriverMethod::SetLeftPaneCollapsed => {
            pbv1::UiDriverMethod::SetLeftPaneCollapsed as i32
        }
        crate::ui_driver::UiDriverMethod::CreateChatSession => {
            pbv1::UiDriverMethod::CreateChatSession as i32
        }
        crate::ui_driver::UiDriverMethod::CloseChatSession => {
            pbv1::UiDriverMethod::CloseChatSession as i32
        }
        crate::ui_driver::UiDriverMethod::PinChatSession => {
            pbv1::UiDriverMethod::PinChatSession as i32
        }
        crate::ui_driver::UiDriverMethod::UnpinChatSession => {
            pbv1::UiDriverMethod::UnpinChatSession as i32
        }
        crate::ui_driver::UiDriverMethod::TriggerRefresh => {
            pbv1::UiDriverMethod::TriggerRefresh as i32
        }
        crate::ui_driver::UiDriverMethod::WaitForSnapshot => {
            pbv1::UiDriverMethod::WaitForSnapshot as i32
        }
        crate::ui_driver::UiDriverMethod::WaitForIdle => pbv1::UiDriverMethod::WaitForIdle as i32,
    }
}

fn decode_ui_driver_method(value: i32) -> Result<crate::ui_driver::UiDriverMethod, ErrorEnvelope> {
    match pbv1::UiDriverMethod::try_from(value) {
        Ok(pbv1::UiDriverMethod::GetSnapshot) => Ok(crate::ui_driver::UiDriverMethod::GetSnapshot),
        Ok(pbv1::UiDriverMethod::OpenEpic) => Ok(crate::ui_driver::UiDriverMethod::OpenEpic),
        Ok(pbv1::UiDriverMethod::SelectTask) => Ok(crate::ui_driver::UiDriverMethod::SelectTask),
        Ok(pbv1::UiDriverMethod::TriggerMerge) => {
            Ok(crate::ui_driver::UiDriverMethod::TriggerMerge)
        }
        Ok(pbv1::UiDriverMethod::OpenSessionView) => {
            Ok(crate::ui_driver::UiDriverMethod::OpenSessionView)
        }
        Ok(pbv1::UiDriverMethod::OpenDiffView) => {
            Ok(crate::ui_driver::UiDriverMethod::OpenDiffView)
        }
        Ok(pbv1::UiDriverMethod::CaptureScreenshot) => {
            Ok(crate::ui_driver::UiDriverMethod::CaptureScreenshot)
        }
        Ok(pbv1::UiDriverMethod::SetLeftPaneCollapsed) => {
            Ok(crate::ui_driver::UiDriverMethod::SetLeftPaneCollapsed)
        }
        Ok(pbv1::UiDriverMethod::CreateChatSession) => {
            Ok(crate::ui_driver::UiDriverMethod::CreateChatSession)
        }
        Ok(pbv1::UiDriverMethod::CloseChatSession) => {
            Ok(crate::ui_driver::UiDriverMethod::CloseChatSession)
        }
        Ok(pbv1::UiDriverMethod::PinChatSession) => Ok(crate::ui_driver::UiDriverMethod::PinChatSession),
        Ok(pbv1::UiDriverMethod::UnpinChatSession) => {
            Ok(crate::ui_driver::UiDriverMethod::UnpinChatSession)
        }
        Ok(pbv1::UiDriverMethod::TriggerRefresh) => Ok(crate::ui_driver::UiDriverMethod::TriggerRefresh),
        Ok(pbv1::UiDriverMethod::WaitForSnapshot) => Ok(crate::ui_driver::UiDriverMethod::WaitForSnapshot),
        Ok(pbv1::UiDriverMethod::WaitForIdle) => Ok(crate::ui_driver::UiDriverMethod::WaitForIdle),
        Ok(pbv1::UiDriverMethod::Unspecified) | Err(_) => Err(invalid_field(
            "method",
            format!("unknown enum value for UiDriverMethod: {value}"),
        )),
    }
}

fn encode_ui_screenshot_window(value: crate::ui_driver::UiScreenshotWindow) -> i32 {
    match value {
        crate::ui_driver::UiScreenshotWindow::Primary => pbv1::UiScreenshotWindow::Primary as i32,
        crate::ui_driver::UiScreenshotWindow::All => pbv1::UiScreenshotWindow::All as i32,
    }
}

fn encode_ui_driver_response_status(value: crate::ui_driver::UiDriverResponseStatus) -> i32 {
    match value {
        crate::ui_driver::UiDriverResponseStatus::Ok => pbv1::UiDriverResponseStatus::Ok as i32,
        crate::ui_driver::UiDriverResponseStatus::Error => {
            pbv1::UiDriverResponseStatus::Error as i32
        }
    }
}

fn decode_ui_driver_response_status(
    value: i32,
) -> Result<crate::ui_driver::UiDriverResponseStatus, ErrorEnvelope> {
    match pbv1::UiDriverResponseStatus::try_from(value) {
        Ok(pbv1::UiDriverResponseStatus::Ok) => Ok(crate::ui_driver::UiDriverResponseStatus::Ok),
        Ok(pbv1::UiDriverResponseStatus::Error) => {
            Ok(crate::ui_driver::UiDriverResponseStatus::Error)
        }
        Ok(pbv1::UiDriverResponseStatus::Unspecified) | Err(_) => Err(invalid_field(
            "status",
            format!("unknown enum value for UiDriverResponseStatus: {value}"),
        )),
    }
}

fn encode_ui_primary_view(value: crate::ui_driver::UiPrimaryView) -> i32 {
    match value {
        crate::ui_driver::UiPrimaryView::EpicSelector => pbv1::UiPrimaryView::EpicSelector as i32,
        crate::ui_driver::UiPrimaryView::EpicWorkspace => pbv1::UiPrimaryView::EpicWorkspace as i32,
    }
}

fn decode_ui_primary_view(value: i32) -> Result<crate::ui_driver::UiPrimaryView, ErrorEnvelope> {
    match pbv1::UiPrimaryView::try_from(value) {
        Ok(pbv1::UiPrimaryView::EpicSelector) => Ok(crate::ui_driver::UiPrimaryView::EpicSelector),
        Ok(pbv1::UiPrimaryView::EpicWorkspace) => {
            Ok(crate::ui_driver::UiPrimaryView::EpicWorkspace)
        }
        Ok(pbv1::UiPrimaryView::Unspecified) | Err(_) => Err(invalid_field(
            "primary_view",
            format!("unknown enum value for UiPrimaryView: {value}"),
        )),
    }
}

impl crate::ui_driver::UiDriverFrame {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiDriverFrame {
        pbv1::UiDriverFrame {
            envelope: Some(self.envelope.to_protobuf()),
            message: Some(match &self.message {
                crate::ui_driver::UiDriverMessage::Request(req) => {
                    pbv1::ui_driver_frame::Message::Request(req.to_protobuf())
                }
                crate::ui_driver::UiDriverMessage::Response(resp) => {
                    pbv1::ui_driver_frame::Message::Response(resp.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::UiDriverFrame) -> Result<Self, ErrorEnvelope> {
        let envelope = ProtocolEnvelope::try_from_protobuf(
            proto.envelope.ok_or_else(|| missing_required("envelope"))?,
        )?;

        let message = match proto.message.ok_or_else(|| missing_required("message"))? {
            pbv1::ui_driver_frame::Message::Request(req) => {
                crate::ui_driver::UiDriverMessage::Request(
                    crate::ui_driver::UiDriverRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::ui_driver_frame::Message::Response(resp) => {
                crate::ui_driver::UiDriverMessage::Response(
                    crate::ui_driver::UiDriverResponse::try_from_protobuf(resp)?,
                )
            }
        };

        Ok(Self { envelope, message })
    }
}

impl crate::ui_driver::UiDriverRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiDriverRequest {
        pbv1::UiDriverRequest {
            request_id: self.request_id.to_bytes().to_vec(),
            method: encode_ui_driver_method(self.payload.method()),
            payload: Some(match &self.payload {
                crate::ui_driver::UiDriverRequestPayload::GetSnapshot(req) => {
                    pbv1::ui_driver_request::Payload::GetSnapshot(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::OpenEpic(req) => {
                    pbv1::ui_driver_request::Payload::OpenEpic(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::SelectTask(req) => {
                    pbv1::ui_driver_request::Payload::SelectTask(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::TriggerMerge(req) => {
                    pbv1::ui_driver_request::Payload::TriggerMerge(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::OpenSessionView(req) => {
                    pbv1::ui_driver_request::Payload::OpenSessionView(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::OpenDiffView(req) => {
                    pbv1::ui_driver_request::Payload::OpenDiffView(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::CaptureScreenshot(req) => {
                    pbv1::ui_driver_request::Payload::CaptureScreenshot(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::SetLeftPaneCollapsed(req) => {
                    pbv1::ui_driver_request::Payload::SetLeftPaneCollapsed(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::CreateChatSession(req) => {
                    pbv1::ui_driver_request::Payload::CreateChatSession(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::CloseChatSession(req) => {
                    pbv1::ui_driver_request::Payload::CloseChatSession(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::PinChatSession(req) => {
                    pbv1::ui_driver_request::Payload::PinChatSession(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::UnpinChatSession(req) => {
                    pbv1::ui_driver_request::Payload::UnpinChatSession(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::TriggerRefresh(req) => {
                    pbv1::ui_driver_request::Payload::TriggerRefresh(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::WaitForSnapshot(req) => {
                    pbv1::ui_driver_request::Payload::WaitForSnapshot(req.to_protobuf())
                }
                crate::ui_driver::UiDriverRequestPayload::WaitForIdle(req) => {
                    pbv1::ui_driver_request::Payload::WaitForIdle(req.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::UiDriverRequest) -> Result<Self, ErrorEnvelope> {
        let request_id = decode_required_ulid::<RequestId>("request_id", &proto.request_id)?;
        let method = decode_ui_driver_method(proto.method)?;

        let payload = match proto.payload.ok_or_else(|| missing_required("payload"))? {
            pbv1::ui_driver_request::Payload::GetSnapshot(req) => {
                crate::ui_driver::UiDriverRequestPayload::GetSnapshot(
                    crate::ui_driver::GetUiSnapshotRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::OpenEpic(req) => {
                crate::ui_driver::UiDriverRequestPayload::OpenEpic(
                    crate::ui_driver::OpenEpicRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::SelectTask(req) => {
                crate::ui_driver::UiDriverRequestPayload::SelectTask(
                    crate::ui_driver::SelectTaskRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::TriggerMerge(req) => {
                crate::ui_driver::UiDriverRequestPayload::TriggerMerge(
                    crate::ui_driver::TriggerMergeRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::OpenSessionView(req) => {
                crate::ui_driver::UiDriverRequestPayload::OpenSessionView(
                    crate::ui_driver::OpenSessionViewRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::OpenDiffView(req) => {
                crate::ui_driver::UiDriverRequestPayload::OpenDiffView(
                    crate::ui_driver::OpenDiffViewRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::CaptureScreenshot(req) => {
                crate::ui_driver::UiDriverRequestPayload::CaptureScreenshot(
                    crate::ui_driver::CaptureScreenshotRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::ui_driver_request::Payload::SetLeftPaneCollapsed(req) => {
                crate::ui_driver::UiDriverRequestPayload::SetLeftPaneCollapsed(
                    crate::ui_driver::SetLeftPaneCollapsedRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::CreateChatSession(req) => {
                crate::ui_driver::UiDriverRequestPayload::CreateChatSession(
                    crate::ui_driver::CreateChatSessionRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::CloseChatSession(req) => {
                crate::ui_driver::UiDriverRequestPayload::CloseChatSession(
                    crate::ui_driver::CloseChatSessionRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::ui_driver_request::Payload::PinChatSession(req) => {
                crate::ui_driver::UiDriverRequestPayload::PinChatSession(
                    crate::ui_driver::PinChatSessionRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::ui_driver_request::Payload::UnpinChatSession(req) => {
                crate::ui_driver::UiDriverRequestPayload::UnpinChatSession(
                    crate::ui_driver::UnpinChatSessionRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::TriggerRefresh(req) => {
                crate::ui_driver::UiDriverRequestPayload::TriggerRefresh(
                    crate::ui_driver::TriggerRefreshRequest::from_protobuf(req),
                )
            }
            pbv1::ui_driver_request::Payload::WaitForSnapshot(req) => {
                crate::ui_driver::UiDriverRequestPayload::WaitForSnapshot(
                    crate::ui_driver::WaitForUiSnapshotRequest::try_from_protobuf(req)?,
                )
            }
            pbv1::ui_driver_request::Payload::WaitForIdle(req) => {
                crate::ui_driver::UiDriverRequestPayload::WaitForIdle(
                    crate::ui_driver::WaitForUiIdleRequest::from_protobuf(req),
                )
            }
        };

        let derived_method = payload.method();
        if method != derived_method {
            return Err(invalid_field(
                "method",
                format!("method does not match payload: {method:?} vs {derived_method:?}"),
            ));
        }

        Ok(Self {
            request_id,
            payload,
        })
    }
}

impl crate::ui_driver::UiDriverResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiDriverResponse {
        pbv1::UiDriverResponse {
            request_id: self.request_id.to_bytes().to_vec(),
            status: encode_ui_driver_response_status(self.status()),
            result: Some(match &self.result {
                crate::ui_driver::UiDriverResponseResult::GetSnapshot(resp) => {
                    pbv1::ui_driver_response::Result::GetSnapshot(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::OpenEpic(resp) => {
                    pbv1::ui_driver_response::Result::OpenEpic(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::SelectTask(resp) => {
                    pbv1::ui_driver_response::Result::SelectTask(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::TriggerMerge(resp) => {
                    pbv1::ui_driver_response::Result::TriggerMerge(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::OpenSessionView(resp) => {
                    pbv1::ui_driver_response::Result::OpenSessionView(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::OpenDiffView(resp) => {
                    pbv1::ui_driver_response::Result::OpenDiffView(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::CaptureScreenshot(resp) => {
                    pbv1::ui_driver_response::Result::CaptureScreenshot(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::SetLeftPaneCollapsed(resp) => {
                    pbv1::ui_driver_response::Result::SetLeftPaneCollapsed(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::CreateChatSession(resp) => {
                    pbv1::ui_driver_response::Result::CreateChatSession(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::CloseChatSession(resp) => {
                    pbv1::ui_driver_response::Result::CloseChatSession(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::PinChatSession(resp) => {
                    pbv1::ui_driver_response::Result::PinChatSession(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::UnpinChatSession(resp) => {
                    pbv1::ui_driver_response::Result::UnpinChatSession(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::TriggerRefresh(resp) => {
                    pbv1::ui_driver_response::Result::TriggerRefresh(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::WaitForSnapshot(resp) => {
                    pbv1::ui_driver_response::Result::WaitForSnapshot(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::WaitForIdle(resp) => {
                    pbv1::ui_driver_response::Result::WaitForIdle(resp.to_protobuf())
                }
                crate::ui_driver::UiDriverResponseResult::Error(err) => {
                    pbv1::ui_driver_response::Result::Error(err.to_protobuf())
                }
            }),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::UiDriverResponse) -> Result<Self, ErrorEnvelope> {
        let request_id = decode_required_ulid::<RequestId>("request_id", &proto.request_id)?;
        let status = decode_ui_driver_response_status(proto.status)?;

        let result = match proto.result.ok_or_else(|| missing_required("result"))? {
            pbv1::ui_driver_response::Result::GetSnapshot(resp) => {
                crate::ui_driver::UiDriverResponseResult::GetSnapshot(
                    crate::ui_driver::GetUiSnapshotResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::ui_driver_response::Result::OpenEpic(resp) => {
                crate::ui_driver::UiDriverResponseResult::OpenEpic(
                    crate::ui_driver::OpenEpicResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::SelectTask(resp) => {
                crate::ui_driver::UiDriverResponseResult::SelectTask(
                    crate::ui_driver::SelectTaskResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::TriggerMerge(resp) => {
                crate::ui_driver::UiDriverResponseResult::TriggerMerge(
                    crate::ui_driver::TriggerMergeResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::ui_driver_response::Result::OpenSessionView(resp) => {
                crate::ui_driver::UiDriverResponseResult::OpenSessionView(
                    crate::ui_driver::OpenSessionViewResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::OpenDiffView(resp) => {
                crate::ui_driver::UiDriverResponseResult::OpenDiffView(
                    crate::ui_driver::OpenDiffViewResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::CaptureScreenshot(resp) => {
                crate::ui_driver::UiDriverResponseResult::CaptureScreenshot(
                    crate::ui_driver::CaptureScreenshotResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::SetLeftPaneCollapsed(resp) => {
                crate::ui_driver::UiDriverResponseResult::SetLeftPaneCollapsed(
                    crate::ui_driver::SetLeftPaneCollapsedResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::CreateChatSession(resp) => {
                crate::ui_driver::UiDriverResponseResult::CreateChatSession(
                    crate::ui_driver::CreateChatSessionResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::ui_driver_response::Result::CloseChatSession(resp) => {
                crate::ui_driver::UiDriverResponseResult::CloseChatSession(
                    crate::ui_driver::CloseChatSessionResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::PinChatSession(resp) => {
                crate::ui_driver::UiDriverResponseResult::PinChatSession(
                    crate::ui_driver::PinChatSessionResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::UnpinChatSession(resp) => {
                crate::ui_driver::UiDriverResponseResult::UnpinChatSession(
                    crate::ui_driver::UnpinChatSessionResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::TriggerRefresh(resp) => {
                crate::ui_driver::UiDriverResponseResult::TriggerRefresh(
                    crate::ui_driver::TriggerRefreshResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::ui_driver_response::Result::WaitForSnapshot(resp) => {
                crate::ui_driver::UiDriverResponseResult::WaitForSnapshot(
                    crate::ui_driver::WaitForUiSnapshotResponse::try_from_protobuf(resp)?,
                )
            }
            pbv1::ui_driver_response::Result::WaitForIdle(resp) => {
                crate::ui_driver::UiDriverResponseResult::WaitForIdle(
                    crate::ui_driver::WaitForUiIdleResponse::from_protobuf(resp),
                )
            }
            pbv1::ui_driver_response::Result::Error(err) => {
                crate::ui_driver::UiDriverResponseResult::Error(ErrorEnvelope::from_protobuf(err))
            }
        };

        let derived_status = match &result {
            crate::ui_driver::UiDriverResponseResult::Error(_) => {
                crate::ui_driver::UiDriverResponseStatus::Error
            }
            _ => crate::ui_driver::UiDriverResponseStatus::Ok,
        };

        if status != derived_status {
            return Err(invalid_field(
                "status",
                format!("status does not match result: {status:?} vs {derived_status:?}"),
            ));
        }

        Ok(Self { request_id, result })
    }
}

impl crate::ui_driver::GetUiSnapshotRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetUiSnapshotRequest {
        pbv1::GetUiSnapshotRequest {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::GetUiSnapshotRequest) -> Self {
        Self {}
    }
}

impl crate::ui_driver::GetUiSnapshotResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::GetUiSnapshotResponse {
        pbv1::GetUiSnapshotResponse {
            snapshot: Some(self.snapshot.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::GetUiSnapshotResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            snapshot: crate::ui_driver::UiSnapshot::try_from_protobuf(
                proto.snapshot.ok_or_else(|| missing_required("snapshot"))?,
            )?,
        })
    }
}

impl crate::ui_driver::OpenEpicRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::OpenEpicRequest {
        pbv1::OpenEpicRequest {
            epic_slug: self.epic_slug.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::OpenEpicRequest) -> Self {
        Self {
            epic_slug: proto.epic_slug,
        }
    }
}

impl crate::ui_driver::OpenEpicResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::OpenEpicResponse {
        pbv1::OpenEpicResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::OpenEpicResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::SelectTaskRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::SelectTaskRequest {
        pbv1::SelectTaskRequest {
            task_slug: self.task_slug.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::SelectTaskRequest) -> Self {
        Self {
            task_slug: proto.task_slug,
        }
    }
}

impl crate::ui_driver::SelectTaskResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::SelectTaskResponse {
        pbv1::SelectTaskResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::SelectTaskResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::TriggerMergeRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TriggerMergeRequest {
        pbv1::TriggerMergeRequest {
            task_slug: self.task_slug.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::TriggerMergeRequest) -> Self {
        Self {
            task_slug: proto.task_slug,
        }
    }
}

impl crate::ui_driver::TriggerMergeResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TriggerMergeResponse {
        pbv1::TriggerMergeResponse {
            command_id: self
                .command_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::TriggerMergeResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            command_id: decode_optional_ulid("command_id", &proto.command_id)?,
        })
    }
}

impl crate::ui_driver::OpenSessionViewRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::OpenSessionViewRequest {
        pbv1::OpenSessionViewRequest {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::OpenSessionViewRequest) -> Self {
        Self {}
    }
}

impl crate::ui_driver::OpenSessionViewResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::OpenSessionViewResponse {
        pbv1::OpenSessionViewResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::OpenSessionViewResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::OpenDiffViewRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::OpenDiffViewRequest {
        pbv1::OpenDiffViewRequest {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::OpenDiffViewRequest) -> Self {
        Self {}
    }
}

impl crate::ui_driver::OpenDiffViewResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::OpenDiffViewResponse {
        pbv1::OpenDiffViewResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::OpenDiffViewResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::SetLeftPaneCollapsedRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::SetLeftPaneCollapsedRequest {
        pbv1::SetLeftPaneCollapsedRequest {
            collapsed: self.collapsed,
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::SetLeftPaneCollapsedRequest) -> Self {
        Self {
            collapsed: proto.collapsed,
        }
    }
}

impl crate::ui_driver::SetLeftPaneCollapsedResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::SetLeftPaneCollapsedResponse {
        pbv1::SetLeftPaneCollapsedResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::SetLeftPaneCollapsedResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::CreateChatSessionRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiDriverCreateChatSessionRequest {
        pbv1::UiDriverCreateChatSessionRequest {
            name_hint: self.name_hint.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::UiDriverCreateChatSessionRequest) -> Self {
        Self {
            name_hint: proto.name_hint,
        }
    }
}

impl crate::ui_driver::CreateChatSessionResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiDriverCreateChatSessionResponse {
        pbv1::UiDriverCreateChatSessionResponse {
            session_id: self.session_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::UiDriverCreateChatSessionResponse,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid("session_id", &proto.session_id)?,
        })
    }
}

impl crate::ui_driver::CloseChatSessionRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiDriverCloseChatSessionRequest {
        pbv1::UiDriverCloseChatSessionRequest {
            session_id: self.session_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::UiDriverCloseChatSessionRequest,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid("session_id", &proto.session_id)?,
        })
    }
}

impl crate::ui_driver::CloseChatSessionResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiDriverCloseChatSessionResponse {
        pbv1::UiDriverCloseChatSessionResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::UiDriverCloseChatSessionResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::PinChatSessionRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::PinChatSessionRequest {
        pbv1::PinChatSessionRequest {
            session_id: self.session_id.to_bytes().to_vec(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::PinChatSessionRequest) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            session_id: decode_required_ulid("session_id", &proto.session_id)?,
        })
    }
}

impl crate::ui_driver::PinChatSessionResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::PinChatSessionResponse {
        pbv1::PinChatSessionResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::PinChatSessionResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::UnpinChatSessionRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UnpinChatSessionRequest {
        pbv1::UnpinChatSessionRequest {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::UnpinChatSessionRequest) -> Self {
        Self {}
    }
}

impl crate::ui_driver::UnpinChatSessionResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UnpinChatSessionResponse {
        pbv1::UnpinChatSessionResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::UnpinChatSessionResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::TriggerRefreshRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TriggerRefreshRequest {
        pbv1::TriggerRefreshRequest {
            epic_slug: self.epic_slug.clone(),
            name_hint: self.name_hint.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::TriggerRefreshRequest) -> Self {
        Self {
            epic_slug: proto.epic_slug,
            name_hint: proto.name_hint,
        }
    }
}

impl crate::ui_driver::TriggerRefreshResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::TriggerRefreshResponse {
        pbv1::TriggerRefreshResponse {
            command_id: self
                .command_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::TriggerRefreshResponse) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            command_id: decode_optional_ulid("command_id", &proto.command_id)?,
        })
    }
}

impl crate::ui_driver::UiSnapshotPredicate {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiSnapshotPredicate {
        pbv1::UiSnapshotPredicate {
            primary_view: self
                .primary_view
                .map(encode_ui_primary_view)
                .unwrap_or(pbv1::UiPrimaryView::Unspecified as i32),
            epic_slug: self.epic_slug.clone(),
            in_flight_empty: self.in_flight_empty,
        }
    }

    pub fn from_protobuf(proto: pbv1::UiSnapshotPredicate) -> Result<Self, ErrorEnvelope> {
        let primary_view = match pbv1::UiPrimaryView::try_from(proto.primary_view) {
            Ok(pbv1::UiPrimaryView::EpicSelector) => Some(crate::ui_driver::UiPrimaryView::EpicSelector),
            Ok(pbv1::UiPrimaryView::EpicWorkspace) => Some(crate::ui_driver::UiPrimaryView::EpicWorkspace),
            Ok(pbv1::UiPrimaryView::Unspecified) | Err(_) => None,
        };

        Ok(Self {
            primary_view,
            epic_slug: proto.epic_slug,
            in_flight_empty: proto.in_flight_empty,
        })
    }
}

impl crate::ui_driver::WaitForUiSnapshotRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForUiSnapshotRequest {
        pbv1::WaitForUiSnapshotRequest {
            timeout_ms: self.timeout_ms,
            predicate: Some(self.predicate.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::WaitForUiSnapshotRequest,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            timeout_ms: proto.timeout_ms,
            predicate: crate::ui_driver::UiSnapshotPredicate::from_protobuf(
                proto
                    .predicate
                    .ok_or_else(|| missing_required("predicate"))?,
            )?,
        })
    }
}

impl crate::ui_driver::WaitForUiSnapshotResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForUiSnapshotResponse {
        pbv1::WaitForUiSnapshotResponse {
            snapshot: Some(self.snapshot.to_protobuf()),
        }
    }

    pub fn try_from_protobuf(
        proto: pbv1::WaitForUiSnapshotResponse,
    ) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            snapshot: crate::ui_driver::UiSnapshot::try_from_protobuf(
                proto.snapshot.ok_or_else(|| missing_required("snapshot"))?,
            )?,
        })
    }
}

impl crate::ui_driver::WaitForUiIdleRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForUiIdleRequest {
        pbv1::WaitForUiIdleRequest {
            timeout_ms: self.timeout_ms,
            quiescence_ms: self.quiescence_ms,
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::WaitForUiIdleRequest) -> Self {
        Self {
            timeout_ms: proto.timeout_ms,
            quiescence_ms: proto.quiescence_ms,
        }
    }
}

impl crate::ui_driver::WaitForUiIdleResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::WaitForUiIdleResponse {
        pbv1::WaitForUiIdleResponse {}
    }

    #[must_use]
    pub fn from_protobuf(_proto: pbv1::WaitForUiIdleResponse) -> Self {
        Self {}
    }
}

impl crate::ui_driver::CaptureScreenshotRequest {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CaptureScreenshotRequest {
        pbv1::CaptureScreenshotRequest {
            name_hint: self.name_hint.clone(),
            window: self
                .window
                .map(encode_ui_screenshot_window)
                .unwrap_or(pbv1::UiScreenshotWindow::Unspecified as i32),
            include_decorations: self.include_decorations,
        }
    }

    pub fn try_from_protobuf(proto: pbv1::CaptureScreenshotRequest) -> Result<Self, ErrorEnvelope> {
        let window = match pbv1::UiScreenshotWindow::try_from(proto.window) {
            Ok(pbv1::UiScreenshotWindow::Primary) => Some(crate::ui_driver::UiScreenshotWindow::Primary),
            Ok(pbv1::UiScreenshotWindow::All) => Some(crate::ui_driver::UiScreenshotWindow::All),
            Ok(pbv1::UiScreenshotWindow::Unspecified) | Err(_) => None,
        };

        Ok(Self {
            name_hint: proto.name_hint,
            window,
            include_decorations: proto.include_decorations,
        })
    }
}

impl crate::ui_driver::CaptureScreenshotResponse {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::CaptureScreenshotResponse {
        pbv1::CaptureScreenshotResponse {
            png_path: self.png_path.clone(),
            png_data: self.png_data.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::CaptureScreenshotResponse) -> Self {
        Self {
            png_path: proto.png_path,
            png_data: proto.png_data,
        }
    }
}

impl crate::ui_driver::UiLeftPaneState {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiLeftPaneState {
        pbv1::UiLeftPaneState {
            visible: self.visible,
            collapsed: self.collapsed,
            width: self.width,
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::UiLeftPaneState) -> Self {
        Self {
            visible: proto.visible,
            collapsed: proto.collapsed,
            width: proto.width,
        }
    }
}

impl crate::ui_driver::UiSelectionState {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiSelectionState {
        pbv1::UiSelectionState {
            epic_id: self
                .epic_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            epic_slug: self.epic_slug.clone(),
            task_id: self
                .task_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
            task_slug: self.task_slug.clone(),
            edge_id: self
                .edge_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::UiSelectionState) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            epic_id: decode_optional_ulid("epic_id", &proto.epic_id)?,
            epic_slug: proto.epic_slug,
            task_id: decode_optional_ulid("task_id", &proto.task_id)?,
            task_slug: proto.task_slug,
            edge_id: decode_optional_ulid("edge_id", &proto.edge_id)?,
        })
    }
}

impl crate::ui_driver::UiInFlightAction {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiInFlightAction {
        pbv1::UiInFlightAction {
            label: self.label.clone(),
            command_id: self
                .command_id
                .map(|id| id.to_bytes().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::UiInFlightAction) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            label: proto.label,
            command_id: decode_optional_ulid("command_id", &proto.command_id)?,
        })
    }
}

impl crate::ui_driver::UiErrorCallout {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiErrorCallout {
        pbv1::UiErrorCallout {
            message: self.message.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::UiErrorCallout) -> Self {
        Self {
            message: proto.message,
        }
    }
}

impl crate::ui_driver::UiSnapshot {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::UiSnapshot {
        pbv1::UiSnapshot {
            captured_at: Some(encode_timestamp(self.captured_at)),
            primary_view: encode_ui_primary_view(self.primary_view),
            left_pane: Some(self.left_pane.to_protobuf()),
            selection: Some(self.selection.to_protobuf()),
            in_flight: self
                .in_flight
                .iter()
                .map(crate::ui_driver::UiInFlightAction::to_protobuf)
                .collect(),
            errors: self
                .errors
                .iter()
                .map(crate::ui_driver::UiErrorCallout::to_protobuf)
                .collect(),
        }
    }

    pub fn try_from_protobuf(proto: pbv1::UiSnapshot) -> Result<Self, ErrorEnvelope> {
        Ok(Self {
            captured_at: decode_required_timestamp("captured_at", proto.captured_at)?,
            primary_view: decode_ui_primary_view(proto.primary_view)?,
            left_pane: crate::ui_driver::UiLeftPaneState::from_protobuf(
                proto
                    .left_pane
                    .ok_or_else(|| missing_required("left_pane"))?,
            ),
            selection: crate::ui_driver::UiSelectionState::try_from_protobuf(
                proto
                    .selection
                    .ok_or_else(|| missing_required("selection"))?,
            )?,
            in_flight: proto
                .in_flight
                .into_iter()
                .map(crate::ui_driver::UiInFlightAction::try_from_protobuf)
                .collect::<Result<Vec<_>, _>>()?,
            errors: proto
                .errors
                .into_iter()
                .map(crate::ui_driver::UiErrorCallout::from_protobuf)
                .collect(),
        })
    }
}
