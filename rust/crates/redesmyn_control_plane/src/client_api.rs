use std::collections::HashMap;
use std::path::{Path, PathBuf};

use redesmyn_logging::tracing;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, Event, EventLogEvent, GetEpicGraphResponse, HealthResponse,
    ListEpicsResponse, Response, ResponseResult, StatusResponse, Subscribed, SubscriptionEvent,
};
use redesmyn_protocol::{
    ErrorCategory, ErrorDetail, ErrorEnvelope, ProtocolEnvelope, ProtocolVersion, Scope, Timestamp,
};
use redesmyn_transport::client::codec::{Codec, JsonCodec, ProtobufCodec};
use redesmyn_transport::client::framed::FramedEndpoint;
use redesmyn_transport::client::{ClientConnection, ClientTransportError};

use crate::ControlPlane;
use crate::error::ControlPlaneError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientApiCodec {
    Json,
    Protobuf,
}

#[derive(Debug, thiserror::Error)]
pub enum ClientApiServeError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Transport(#[from] ClientTransportError),
}

#[cfg(not(unix))]
pub fn bind_client_socket(_path: &Path) -> Result<(), ClientApiServeError> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "unix domain sockets are not supported on this platform",
    )
    .into())
}

#[cfg(unix)]
pub fn bind_client_socket(path: &Path) -> Result<tokio::net::UnixListener, ClientApiServeError> {
    use std::os::unix::fs::PermissionsExt as _;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if path.exists() {
        std::fs::remove_file(path)?;
    }

    let listener = tokio::net::UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

#[cfg(unix)]
pub async fn serve_client_api_uds(
    control_plane: ControlPlane,
    socket_path: PathBuf,
    codec: ClientApiCodec,
) -> Result<(), ClientApiServeError> {
    let listener = bind_client_socket(&socket_path)?;

    tracing::info!(
        socket_path = %socket_path.display(),
        codec = ?codec,
        "client API UDS server listening"
    );

    loop {
        let (stream, _addr) = listener.accept().await?;
        let task_codec = codec;
        let task_control_plane = control_plane.clone();

        tokio::spawn(async move {
            let result = match task_codec {
                ClientApiCodec::Json => {
                    handle_connection(stream, JsonCodec::new(), task_control_plane).await
                }
                ClientApiCodec::Protobuf => {
                    handle_connection(stream, ProtobufCodec::new(), task_control_plane).await
                }
            };

            if let Err(err) = result {
                tracing::warn!(error = %err, "client API connection terminated with error");
            }
        });
    }
}

async fn handle_connection<C>(
    stream: tokio::net::UnixStream,
    codec: C,
    control_plane: ControlPlane,
) -> Result<(), ClientApiServeError>
where
    C: Codec + 'static,
{
    let peer = stream.peer_addr().ok();
    let span = tracing::info_span!(
        "client.api.connection",
        peer = peer
            .as_ref()
            .map(|addr| addr.as_pathname().map(|p| p.display().to_string()))
            .flatten()
            .unwrap_or_else(|| "<unknown>".to_string())
    );
    let _enter = span.enter();

    let mut conn = FramedEndpoint::new(stream, codec);
    run_session(&mut conn, control_plane).await?;
    Ok(())
}

async fn run_session<C>(conn: &mut C, control_plane: ControlPlane) -> Result<(), ClientApiServeError>
where
    C: ClientConnection + Send + 'static,
{
    let mut accepted_protocol: Option<ProtocolVersion> = None;
    let mut subscriptions: HashMap<redesmyn_ids::SubscriptionId, tokio::task::JoinHandle<()>> =
        HashMap::new();
    let (events_tx, mut events_rx) = tokio::sync::mpsc::channel::<ClientFrame>(128);

    loop {
        tokio::select! {
            frame = conn.recv() => {
                let frame = match frame {
                    Ok(frame) => frame,
                    Err(ClientTransportError::ChannelClosed) => return Ok(()),
                    Err(err) => return Err(err.into()),
                };

                let peer_version = frame.envelope.protocol_version();
                let accepted = match accepted_protocol {
                    Some(version) => version,
                    None => match ProtocolVersion::CURRENT.negotiate(peer_version) {
                        Ok(version) => {
                            accepted_protocol = Some(version);
                            version
                        }
                        Err(err) => {
                            if let ClientMessage::Request(req) = frame.message {
                                let response = Response {
                                    request_id: req.request_id,
                                    result: ResponseResult::Error(err),
                                };

                                let response_frame = ClientFrame::new(
                                    error_response_envelope(&frame.envelope),
                                    ClientMessage::Response(response),
                                );

                                let _ = conn.send(response_frame).await;
                            }
                            return Ok(());
                        }
                    },
                };

                match frame.message {
                    ClientMessage::Request(req) => {
                        let response = handle_request(req, accepted).await;
                        let response_frame = ClientFrame::new(
                            response_envelope(&frame.envelope, accepted),
                            ClientMessage::Response(response),
                        );
                        conn.send(response_frame).await?;
                    }
                    ClientMessage::Subscribe(sub) => {
                        let subscription_id = sub.subscription_id;
                        if let Some(prev) = subscriptions.remove(&subscription_id) {
                            prev.abort();
                        }

                        let topic = sub.topic();
                        let mut subscribe_error: Option<ErrorEnvelope> = None;
                        let mut subscription_start: Option<tokio::sync::oneshot::Sender<()>> = None;

                        let handle = match sub.filter {
                            redesmyn_protocol::client::SubscriptionFilter::EventLog(filter) => {
                                let scope = match storage_scope_from_envelope(&frame.envelope) {
                                    Ok(scope) => Some(scope),
                                    Err(err) => {
                                        subscribe_error = Some(err);
                                        None
                                    }
                                };

                                scope.map(|scope| {
                                    let mut feed = control_plane
                                        .event_log()
                                        .subscribe(scope, filter.after_event_id);
                                    let tx = events_tx.clone();
                                    let request_envelope = frame.envelope.clone();
                                    let (start_tx, start_rx) = tokio::sync::oneshot::channel::<()>();

                                    let handle = tokio::spawn(async move {
                                        if start_rx.await.is_err() {
                                            return;
                                        }
                                        while let Some(item) = feed.recv().await {
                                            let (event, terminal) = match item {
                                                crate::event_log::EventLogSubscriptionItem::Event(
                                                    record,
                                                ) => {
                                                    (
                                                        SubscriptionEvent::EventLog(EventLogEvent {
                                                            event_id: record.id,
                                                            occurred_at: timestamp_from_ms(
                                                                record.created_at_ms,
                                                            ),
                                                            event_type: record.kind,
                                                            json_payload: record.payload,
                                                        }),
                                                        false,
                                                    )
                                                }
                                                crate::event_log::EventLogSubscriptionItem::ResyncRequired(
                                                    resync,
                                                ) => {
                                                    (
                                                        SubscriptionEvent::Error(
                                                            resync_error_envelope(resync),
                                                        ),
                                                        true,
                                                    )
                                                }
                                            };

                                            let frame = ClientFrame::new(
                                                response_envelope(&request_envelope, accepted),
                                                ClientMessage::Event(Event {
                                                    subscription_id,
                                                    event,
                                                }),
                                            );

                                            if tx.send(frame).await.is_err() {
                                                return;
                                            }

                                            if terminal {
                                                return;
                                            }
                                        }
                                    });

                                    (handle, start_tx)
                                })
                            }
                        };

                        if let Some((handle, start_tx)) = handle {
                            subscriptions.insert(subscription_id, handle);
                            subscription_start = Some(start_tx);
                        }

                        let ack = ClientFrame::new(
                            response_envelope(&frame.envelope, accepted),
                            ClientMessage::Event(Event {
                                subscription_id,
                                event: SubscriptionEvent::Subscribed(Subscribed { topic }),
                            }),
                        );
                        conn.send(ack).await?;

                        if let Some(start_tx) = subscription_start {
                            let _ = start_tx.send(());
                        }

                        if let Some(err) = subscribe_error {
                            let event = ClientFrame::new(
                                response_envelope(&frame.envelope, accepted),
                                ClientMessage::Event(Event {
                                    subscription_id,
                                    event: SubscriptionEvent::Error(err),
                                }),
                            );
                            conn.send(event).await?;
                        }
                    }
                    ClientMessage::Unsubscribe(unsub) => {
                        if let Some(handle) = subscriptions.remove(&unsub.subscription_id) {
                            handle.abort();
                        }
                    }
                    ClientMessage::Response(_) | ClientMessage::Event(_) => {
                        let err: ErrorEnvelope = ControlPlaneError::InvalidTaskId {
                            task_id: "client.sent_server_message".to_string(),
                        }
                        .into();

                        tracing::debug!(error = %err.message, "ignoring unexpected client message");
                    }
                }
            }
            maybe_event = events_rx.recv() => {
                let Some(event) = maybe_event else {
                    continue;
                };
                conn.send(event).await?;
            }
        }
    }
}

async fn handle_request(
    req: redesmyn_protocol::client::Request,
    accepted: ProtocolVersion,
) -> Response {
    let request_id = req.request_id;
    let span = tracing::debug_span!(
        "client.api.request",
        request_id = %request_id,
        method = ?req.method(),
    );
    let _enter = span.enter();

    let result = match req.payload {
        redesmyn_protocol::client::RequestPayload::Health(_) => {
            ResponseResult::Health(HealthResponse { ok: true })
        }
        redesmyn_protocol::client::RequestPayload::Status(_) => {
            ResponseResult::Status(StatusResponse {
                accepted_protocol: accepted,
                server_name: "redesmyn-control-plane".to_string(),
                server_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            })
        }
        redesmyn_protocol::client::RequestPayload::ListEpics(_) => {
            ResponseResult::ListEpics(ListEpicsResponse { epics: Vec::new() })
        }
        redesmyn_protocol::client::RequestPayload::GetEpicGraph(get) => {
            ResponseResult::GetEpicGraph(GetEpicGraphResponse {
                graph: redesmyn_protocol::client::EpicGraph {
                    epic_slug: get.epic_slug,
                    nodes: Vec::new(),
                    edges: Vec::new(),
                },
            })
        }
    };

    Response { request_id, result }
}

fn response_envelope(
    request_envelope: &ProtocolEnvelope,
    accepted: ProtocolVersion,
) -> ProtocolEnvelope {
    let mut envelope = ProtocolEnvelope::new();
    envelope.protocol_major = accepted.major;
    envelope.protocol_minor = accepted.minor;
    envelope.scope = request_envelope.scope;
    envelope.trace_id = request_envelope.trace_id;
    envelope.correlation_id = Some(request_envelope.msg_id);
    envelope
}

fn error_response_envelope(request_envelope: &ProtocolEnvelope) -> ProtocolEnvelope {
    let mut envelope = ProtocolEnvelope::new();
    envelope.scope = request_envelope.scope;
    envelope.trace_id = request_envelope.trace_id;
    envelope.correlation_id = Some(request_envelope.msg_id);
    envelope
}

fn storage_scope_from_envelope(
    envelope: &ProtocolEnvelope,
) -> Result<redesmyn_storage::events::EventScope, ErrorEnvelope> {
    match envelope.scope {
        None => Ok(redesmyn_storage::events::EventScope::None),
        Some(Scope::Repo { repo }) => Ok(redesmyn_storage::events::EventScope::Repo {
            workspace_id: repo.workspace_id,
            repo_id: repo.repo_id,
        }),
        Some(Scope::Unknown) => Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Unknown scope kind.",
        )),
        Some(_) => Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Unsupported scope kind.",
        )),
    }
}

fn resync_error_envelope(resync: crate::event_log::EventLogResync) -> ErrorEnvelope {
    let reason = match resync.reason {
        crate::event_log::EventLogResyncReason::Lagged => "lagged",
        crate::event_log::EventLogResyncReason::CursorNotFound => "cursor_not_found",
        crate::event_log::EventLogResyncReason::CursorScopeMismatch => "cursor_scope_mismatch",
        crate::event_log::EventLogResyncReason::DbError => "db_error",
    };

    let mut detail: ErrorDetail = ErrorDetail::new();
    detail.insert("reason".to_string(), reason.to_string());
    if let Some(event_id) = resync.resume_after_event_id {
        detail.insert("resume_after_event_id".to_string(), event_id.to_string());
    }
    if let Some(skipped) = resync.dropped_events {
        detail.insert("dropped_events".to_string(), skipped.to_string());
    }

    ErrorEnvelope::new(
        ErrorCategory::Unavailable,
        "Event subscription requires resync.",
    )
    .with_detail(detail)
}

fn timestamp_from_ms(ms: i64) -> Timestamp {
    let nanos = i128::from(ms).saturating_mul(1_000_000);
    let datetime = time::OffsetDateTime::from_unix_timestamp_nanos(nanos)
        .unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
    Timestamp::from_offset_date_time(datetime)
}
