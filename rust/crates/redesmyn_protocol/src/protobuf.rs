use std::collections::{BTreeMap, HashMap};

use redesmyn_ids::{
    EventId, HostId, HostInstanceId, MsgId, RepoId, RequestId, SubscriptionId, WorkspaceId,
};

use crate::pb::redesmyn::protocol::v1 as pbv1;
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

fn encode_client_method(value: crate::client::ClientMethod) -> i32 {
    match value {
        crate::client::ClientMethod::Health => pbv1::ClientMethod::Health as i32,
        crate::client::ClientMethod::Status => pbv1::ClientMethod::Status as i32,
        crate::client::ClientMethod::ListEpics => pbv1::ClientMethod::ListEpics as i32,
        crate::client::ClientMethod::GetEpicGraph => pbv1::ClientMethod::GetEpicGraph as i32,
    }
}

fn decode_client_method(value: i32) -> Result<crate::client::ClientMethod, ErrorEnvelope> {
    match pbv1::ClientMethod::try_from(value) {
        Ok(pbv1::ClientMethod::Health) => Ok(crate::client::ClientMethod::Health),
        Ok(pbv1::ClientMethod::Status) => Ok(crate::client::ClientMethod::Status),
        Ok(pbv1::ClientMethod::ListEpics) => Ok(crate::client::ClientMethod::ListEpics),
        Ok(pbv1::ClientMethod::GetEpicGraph) => Ok(crate::client::ClientMethod::GetEpicGraph),
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
    }
}

fn decode_subscription_topic(
    value: i32,
) -> Result<crate::client::SubscriptionTopic, ErrorEnvelope> {
    match pbv1::SubscriptionTopic::try_from(value) {
        Ok(pbv1::SubscriptionTopic::EventLog) => Ok(crate::client::SubscriptionTopic::EventLog),
        Ok(pbv1::SubscriptionTopic::Unspecified) | Err(_) => Err(invalid_field(
            "topic",
            format!("unknown enum value for SubscriptionTopic: {value}"),
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
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::EpicTaskNode) -> Self {
        Self {
            task_slug: proto.task_slug,
            title: proto.title,
        }
    }
}

impl crate::client::EpicTaskEdge {
    #[must_use]
    pub fn to_protobuf(&self) -> pbv1::EpicTaskEdge {
        pbv1::EpicTaskEdge {
            from_task_slug: self.from_task_slug.clone(),
            to_task_slug: self.to_task_slug.clone(),
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::EpicTaskEdge) -> Self {
        Self {
            from_task_slug: proto.from_task_slug,
            to_task_slug: proto.to_task_slug,
        }
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
        }
    }

    #[must_use]
    pub fn from_protobuf(proto: pbv1::EpicGraph) -> Self {
        Self {
            epic_slug: proto.epic_slug,
            nodes: proto
                .nodes
                .into_iter()
                .map(crate::client::EpicTaskNode::from_protobuf)
                .collect(),
            edges: proto
                .edges
                .into_iter()
                .map(crate::client::EpicTaskEdge::from_protobuf)
                .collect(),
        }
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
            graph: crate::client::EpicGraph::from_protobuf(
                proto.graph.ok_or_else(|| missing_required("graph"))?,
            ),
        })
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
