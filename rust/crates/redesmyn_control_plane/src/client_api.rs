use std::collections::HashMap;
use std::path::{Path, PathBuf};

use redesmyn_logging::tracing;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, CommandState, CommandSummary, CommandUpdateSummary,
    DaemonPresenceSummary, EpicGraph, EpicSummary, Event, EventLogEvent, GetEpicGraphResponse,
    GetEpicPinnedChatSessionResponse, GetLatestTaskSessionResponse, GetSessionEventsResponse,
    HealthResponse, ListEpicsResponse, MergeReadiness, Response, ResponseResult, SessionSummary,
    StatusResponse, Subscribed, SubscriptionEvent, TaskState,
};
use redesmyn_protocol::{
    ErrorCategory, ErrorDetail, ErrorEnvelope, ProtocolEnvelope, ProtocolVersion, Scope, Timestamp,
};
use redesmyn_transport::client::codec::{Codec, JsonCodec, ProtobufCodec};
use redesmyn_transport::client::framed::FramedEndpoint;
use redesmyn_transport::client::{ClientConnection, ClientTransportError};

use crate::ControlPlane;
use crate::error::ControlPlaneError;
use crate::session_events::{
    SessionEventsResync, SessionEventsResyncReason, SessionEventsSubscriptionItem,
};

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
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError> {
    let listener = bind_client_socket(&socket_path)?;

    serve_client_api_listener(listener, control_plane, socket_path, codec, shutdown).await
}

#[cfg(unix)]
pub async fn serve_client_api_listener(
    listener: tokio::net::UnixListener,
    control_plane: ControlPlane,
    socket_path: PathBuf,
    codec: ClientApiCodec,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError> {
    tracing::info!(
        socket_path = %socket_path.display(),
        codec = ?codec,
        "client API UDS server listening"
    );

    let mut connections = tokio::task::JoinSet::new();

    loop {
        tokio::select! {
            _ = shutdown.recv() => break,
            accept = listener.accept() => {
                let (stream, _addr) = accept?;
                let task_codec = codec;
                let task_control_plane = control_plane.clone();
                let mut task_shutdown = shutdown.resubscribe();

                connections.spawn(async move {
                    let result = match task_codec {
                        ClientApiCodec::Json => handle_connection(stream, JsonCodec::new(), task_control_plane, &mut task_shutdown).await,
                        ClientApiCodec::Protobuf => handle_connection(stream, ProtobufCodec::new(), task_control_plane, &mut task_shutdown).await,
                    };

                    if let Err(err) = result {
                        tracing::warn!(error = %err, "client API connection terminated with error");
                    }
                });
            }
            Some(join_result) = connections.join_next() => {
                if let Err(err) = join_result {
                    tracing::error!(error = %err, "client API connection task panicked");
                }
            }
        }
    }

    while let Some(join_result) = connections.join_next().await {
        if let Err(err) = join_result {
            tracing::error!(error = %err, "client API connection task panicked");
        }
    }

    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    tracing::info!("client API UDS server shut down");
    Ok(())
}

async fn handle_connection<C>(
    stream: tokio::net::UnixStream,
    codec: C,
    control_plane: ControlPlane,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
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
    serve_connection(&mut conn, control_plane, shutdown).await?;
    Ok(())
}

/// Serve a single client ↔ control plane connection (T-12).
///
/// This is the shared per-connection implementation used by:
/// - the UDS server (`serve_client_api_*`), and
/// - embedded/in-proc callers (`ControlPlaneHandle::connect_in_proc_client`).
pub async fn serve_connection<C>(
    conn: &mut C,
    control_plane: ControlPlane,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError>
where
    C: ClientConnection + Send + 'static,
{
    run_session(conn, control_plane, shutdown).await
}

async fn run_session<C>(
    conn: &mut C,
    control_plane: ControlPlane,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError>
where
    C: ClientConnection + Send + 'static,
{
    fn abort_subscriptions(
        subscriptions: &mut HashMap<redesmyn_ids::SubscriptionId, tokio::task::JoinHandle<()>>,
    ) {
        for (_subscription_id, handle) in subscriptions.drain() {
            handle.abort();
        }
    }

    let mut accepted_protocol: Option<ProtocolVersion> = None;
    let mut subscriptions: HashMap<redesmyn_ids::SubscriptionId, tokio::task::JoinHandle<()>> =
        HashMap::new();
    let (events_tx, mut events_rx) = tokio::sync::mpsc::channel::<ClientFrame>(128);

    loop {
        tokio::select! {
            _ = shutdown.recv() => {
                abort_subscriptions(&mut subscriptions);
                return Ok(());
            }
            frame = conn.recv() => {
                let frame = match frame {
                    Ok(frame) => frame,
                    Err(ClientTransportError::ChannelClosed) => {
                        abort_subscriptions(&mut subscriptions);
                        return Ok(());
                    }
                    Err(err) => {
                        abort_subscriptions(&mut subscriptions);
                        return Err(err.into());
                    }
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
                            abort_subscriptions(&mut subscriptions);
                            return Ok(());
                        }
                    },
                };

                match frame.message {
                    ClientMessage::Request(req) => {
                        let response =
                            handle_request(req, accepted, &frame.envelope, &control_plane).await;
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
                            redesmyn_protocol::client::SubscriptionFilter::SessionEvents(filter) => {
                                let mut feed = control_plane
                                    .session_events()
                                    .subscribe(filter.session_id, filter.after);
                                let tx = events_tx.clone();
                                let request_envelope = frame.envelope.clone();
                                let (start_tx, start_rx) = tokio::sync::oneshot::channel::<()>();

                                let handle = tokio::spawn(async move {
                                    if start_rx.await.is_err() {
                                        return;
                                    }

                                    while let Some(item) = feed.recv().await {
                                        let (event, terminal) = match item {
                                            SessionEventsSubscriptionItem::Event(event) => {
                                                (SubscriptionEvent::SessionEvent(event), false)
                                            }
                                            SessionEventsSubscriptionItem::ResyncRequired(
                                                resync,
                                            ) => (
                                                SubscriptionEvent::Error(
                                                    session_events_resync_error_envelope(resync),
                                                ),
                                                true,
                                            ),
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

                                Some((handle, start_tx))
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
    envelope: &ProtocolEnvelope,
    control_plane: &ControlPlane,
) -> Response {
    let request_id = req.request_id;
    let span = tracing::debug_span!(
        "client.api.request",
        request_id = %request_id,
        method = ?req.method(),
    );
    let _enter = span.enter();

    let result = match handle_request_result(req.payload, accepted, envelope, control_plane).await {
        Ok(result) => result,
        Err(err) => ResponseResult::Error(err.into()),
    };

    Response { request_id, result }
}

async fn handle_request_result(
    payload: redesmyn_protocol::client::RequestPayload,
    accepted: ProtocolVersion,
    envelope: &ProtocolEnvelope,
    control_plane: &ControlPlane,
) -> Result<ResponseResult, ControlPlaneError> {
    match payload {
        redesmyn_protocol::client::RequestPayload::Health(_) => {
            Ok(ResponseResult::Health(HealthResponse { ok: true }))
        }
        redesmyn_protocol::client::RequestPayload::Status(_) => {
            Ok(ResponseResult::Status(StatusResponse {
                accepted_protocol: accepted,
                server_name: "redesmyn-control-plane".to_string(),
                server_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }))
        }
        redesmyn_protocol::client::RequestPayload::ListEpics(_) => {
            let scope = repo_scope_from_envelope(envelope);
            let epics = redesmyn_storage::epic_graph::list_epics(control_plane.pool(), scope).await?;
            Ok(ResponseResult::ListEpics(ListEpicsResponse {
                epics: epics
                    .into_iter()
                    .map(|epic| EpicSummary {
                        slug: epic.slug,
                        name: epic.title,
                    })
                    .collect(),
            }))
        }
        redesmyn_protocol::client::RequestPayload::GetEpicGraph(get) => {
            let epic_slug = get.epic_slug;
            if epic_slug.trim().is_empty() {
                return Err(ControlPlaneError::InvalidEpicSlug { epic_slug });
            }

            let scope = repo_scope_from_envelope(envelope);
            let load_span = tracing::debug_span!("client.api.load_epic_graph", epic_slug = %epic_slug);
            let Some(graph) = ({
                let _enter = load_span.enter();
                redesmyn_storage::epic_graph::load_epic_graph(control_plane.pool(), &epic_slug, scope).await?
            }) else {
                return Err(ControlPlaneError::EpicNotFound { epic_slug });
            };

            let build_span =
                tracing::debug_span!("client.api.build_epic_graph", epic_slug = %epic_slug);
            let graph = {
                let _enter = build_span.enter();
                build_epic_graph(graph)?
            };

            Ok(ResponseResult::GetEpicGraph(GetEpicGraphResponse { graph }))
        }
        redesmyn_protocol::client::RequestPayload::GetSessionEvents(get) => {
            const MAX_LIMIT: u32 = 512;
            if get.limit == 0 {
                Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "GetSessionEvents.limit must be > 0.",
                )))
            } else if get.limit > MAX_LIMIT {
                Ok(ResponseResult::Error(
                    ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "GetSessionEvents.limit is too large.",
                    )
                    .with_detail(ErrorDetail::from([(
                        "max_limit".to_string(),
                        MAX_LIMIT.to_string(),
                    )])),
                ))
            } else {
                match control_plane
                    .session_events()
                    .get_session_events(get.session_id, get.before, get.limit, &get.kinds)
                    .await
                {
                    Ok((events, next_cursor)) => Ok(ResponseResult::GetSessionEvents(
                        GetSessionEventsResponse {
                            events,
                            next_cursor,
                        },
                    )),
                    Err(err) => {
                        tracing::warn!(error = %err, "GetSessionEvents failed");
                        Ok(ResponseResult::Error(ErrorEnvelope::new(
                            ErrorCategory::Internal,
                            "Failed to query session events.",
                        )))
                    }
                }
            }
        }
        redesmyn_protocol::client::RequestPayload::GetLatestTaskSession(get) => {
            match control_plane
                .session_events()
                .get_latest_task_session(get.task_id)
                .await
            {
                Ok(session_id) => Ok(ResponseResult::GetLatestTaskSession(
                    GetLatestTaskSessionResponse { session_id },
                )),
                Err(err) => {
                    tracing::warn!(error = %err, "GetLatestTaskSession failed");
                    Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Internal,
                        "Failed to query latest task session.",
                    )))
                }
            }
        }
        redesmyn_protocol::client::RequestPayload::GetEpicPinnedChatSession(get) => {
            match control_plane
                .session_events()
                .get_epic_pinned_chat_session(get.epic_id)
                .await
            {
                Ok(session_id) => Ok(ResponseResult::GetEpicPinnedChatSession(
                    GetEpicPinnedChatSessionResponse { session_id },
                )),
                Err(err) => {
                    tracing::warn!(error = %err, "GetEpicPinnedChatSession failed");
                    Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Internal,
                        "Failed to query epic pinned chat session.",
                    )))
                }
            }
        }
    }
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

fn session_events_resync_error_envelope(resync: SessionEventsResync) -> ErrorEnvelope {
    let reason = match resync.reason {
        SessionEventsResyncReason::Lagged => "lagged",
        SessionEventsResyncReason::CursorNotFound => "cursor_not_found",
        SessionEventsResyncReason::CursorSessionMismatch => "cursor_session_mismatch",
        SessionEventsResyncReason::DbError => "db_error",
    };

    let mut detail: ErrorDetail = ErrorDetail::new();
    detail.insert("reason".to_string(), reason.to_string());

    if let Some(cursor) = resync.resume_after {
        detail.insert(
            "resume_after_session_event_id".to_string(),
            cursor.session_event_id.to_string(),
        );

        let created_at = cursor
            .created_at
            .into_offset_date_time()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "<invalid>".to_string());
        detail.insert("resume_after_created_at".to_string(), created_at);
    }

    if let Some(skipped) = resync.dropped_events {
        detail.insert("dropped_events".to_string(), skipped.to_string());
    }

    ErrorEnvelope::new(
        ErrorCategory::Unavailable,
        "Session event subscription requires resync.",
    )
    .with_detail(detail)
}

fn timestamp_from_ms(ms: i64) -> Timestamp {
    let nanos = i128::from(ms).saturating_mul(1_000_000);
    let datetime = time::OffsetDateTime::from_unix_timestamp_nanos(nanos)
        .unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
    Timestamp::from_offset_date_time(datetime)
}

fn repo_scope_from_envelope(
    envelope: &ProtocolEnvelope,
) -> Option<redesmyn_storage::epic_graph::RepoScope> {
    let scope = envelope.scope?;
    let redesmyn_protocol::Scope::Repo { repo } = scope else {
        return None;
    };

    Some(redesmyn_storage::epic_graph::RepoScope {
        workspace_id: repo.workspace_id,
        repo_id: repo.repo_id,
    })
}

fn timestamp_from_unix_ms(unix_ms: i64) -> Result<Timestamp, ControlPlaneError> {
    let nanos = i128::from(unix_ms)
        .checked_mul(1_000_000)
        .ok_or(ControlPlaneError::InvalidTimestamp { unix_ms })?;

    let dt = time::OffsetDateTime::from_unix_timestamp_nanos(nanos)
        .map_err(|_| ControlPlaneError::InvalidTimestamp { unix_ms })?;
    Ok(Timestamp::from_offset_date_time(dt))
}

fn build_epic_graph(
    graph: redesmyn_storage::epic_graph::EpicGraphData,
) -> Result<EpicGraph, ControlPlaneError> {
    let task_slug_by_id: HashMap<redesmyn_ids::TaskId, String> = graph
        .tasks
        .iter()
        .map(|task| {
            (
                task.task_id,
                task.local_ref
                    .clone()
                    .unwrap_or_else(|| task.task_id.to_string()),
            )
        })
        .collect();

    let nodes = graph
        .tasks
        .iter()
        .map(|task| {
            let task_slug = task_slug_by_id
                .get(&task.task_id)
                .cloned()
                .unwrap_or_else(|| task.task_id.to_string());

            redesmyn_protocol::client::EpicTaskNode {
                task_slug,
                title: task.title.clone(),
                task_id: Some(task.task_id),
                parent_task_id: task.parent_task_id,
                state: match task.state {
                    redesmyn_storage::schema::TaskState::Todo => TaskState::Todo,
                    redesmyn_storage::schema::TaskState::InProgress => TaskState::InProgress,
                    redesmyn_storage::schema::TaskState::Blocked => TaskState::Blocked,
                    redesmyn_storage::schema::TaskState::Done => TaskState::Done,
                },
                branch_name: task.branch_name.clone(),
                merge_readiness: match task.merge_readiness {
                    redesmyn_storage::schema::MergeReadiness::Unknown => MergeReadiness::Unknown,
                    redesmyn_storage::schema::MergeReadiness::Ready => MergeReadiness::Ready,
                    redesmyn_storage::schema::MergeReadiness::Blocked => MergeReadiness::Blocked,
                },
            }
        })
        .collect();

    let mut edges: Vec<redesmyn_protocol::client::EpicTaskEdge> = Vec::new();
    for task in &graph.tasks {
        let Some(parent_task_id) = task.parent_task_id else {
            continue;
        };
        let from_task_slug = task_slug_by_id
            .get(&parent_task_id)
            .cloned()
            .unwrap_or_else(|| parent_task_id.to_string());
        let to_task_slug = task_slug_by_id
            .get(&task.task_id)
            .cloned()
            .unwrap_or_else(|| task.task_id.to_string());

        edges.push(redesmyn_protocol::client::EpicTaskEdge {
            from_task_slug,
            to_task_slug,
            from_task_id: Some(parent_task_id),
            to_task_id: Some(task.task_id),
        });
    }
    edges.sort_by(|a, b| a.to_task_slug.cmp(&b.to_task_slug));

    let command_summaries = graph
        .commands
        .iter()
        .map(|command| {
            let last_update = graph.command_last_updates.get(&command.command_id).map(
                |update| -> Result<CommandUpdateSummary, ControlPlaneError> {
                    let progress_current = update
                        .progress_current
                        .and_then(|value| u64::try_from(value).ok());
                    let progress_total = update
                        .progress_total
                        .and_then(|value| u64::try_from(value).ok());

                    Ok(CommandUpdateSummary {
                        update_id: update.update_id,
                        created_at: timestamp_from_unix_ms(update.created_at_ms)?,
                        state: map_command_state(update.state),
                        message: update.message.clone(),
                        progress_current,
                        progress_total,
                    })
                },
            );

            Ok(CommandSummary {
                command_id: command.command_id,
                created_at: timestamp_from_unix_ms(command.created_at_ms)?,
                updated_at: timestamp_from_unix_ms(command.updated_at_ms)?,
                kind: command.kind.clone(),
                state: map_command_state(command.state),
                target_task_id: command.target_task_id,
                last_update: last_update.transpose()?,
            })
        })
        .collect::<Result<Vec<_>, ControlPlaneError>>()?;

    let daemon_presences = graph
        .daemon_presences
        .iter()
        .map(|presence| {
            Ok(DaemonPresenceSummary {
                host_instance_id: presence.host_instance_id,
                host_id: presence.host_id,
                hostname: presence.hostname.clone(),
                connected_at: timestamp_from_unix_ms(presence.connected_at_ms)?,
                last_heartbeat_at: timestamp_from_unix_ms(presence.last_heartbeat_at_ms)?,
                disconnected_at: presence
                    .disconnected_at_ms
                    .map(timestamp_from_unix_ms)
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>, ControlPlaneError>>()?;

    let session_summaries = graph
        .session_summaries
        .iter()
        .map(|session| {
            Ok(SessionSummary {
                session_id: session.session_id,
                session_event_id: session.session_event_id,
                task_id: session.task_id,
                last_event_at: timestamp_from_unix_ms(session.created_at_ms)?,
                kind: session.kind.clone(),
                turn_id: session.turn_id.clone(),
                message_preview: session.message_preview.clone(),
            })
        })
        .collect::<Result<Vec<_>, ControlPlaneError>>()?;

    Ok(EpicGraph {
        epic_slug: graph.epic.slug,
        nodes,
        edges,
        epic_id: Some(graph.epic.epic_id),
        epic_title: Some(graph.epic.title),
        workspace_id: Some(graph.epic.scope.workspace_id),
        repo_id: Some(graph.epic.scope.repo_id),
        command_summaries,
        daemon_presences,
        session_summaries,
        as_of_event_id: graph.as_of_event_id,
    })
}

fn map_command_state(state: redesmyn_storage::schema::CommandState) -> CommandState {
    match state {
        redesmyn_storage::schema::CommandState::Accepted => CommandState::Accepted,
        redesmyn_storage::schema::CommandState::Running => CommandState::Running,
        redesmyn_storage::schema::CommandState::Blocked => CommandState::Blocked,
        redesmyn_storage::schema::CommandState::Resumable => CommandState::Resumable,
        redesmyn_storage::schema::CommandState::Succeeded => CommandState::Succeeded,
        redesmyn_storage::schema::CommandState::Failed => CommandState::Failed,
        redesmyn_storage::schema::CommandState::Canceled => CommandState::Canceled,
    }
}
