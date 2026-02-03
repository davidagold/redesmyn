use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::ControlPlane;
use crate::agent_orchestration::conflicts::{
    CONFLICT_CODE_KEY, CONFLICT_CODE_SESSION_CONFLICT, CONFLICT_CODE_TURN_IN_PROGRESS,
};
use crate::error::ControlPlaneError;
use crate::session_events::{
    SessionEventsResync, SessionEventsResyncReason, SessionEventsSubscriptionItem,
};
use redesmyn_ids::{RepoId, SessionEventId, SessionId, WorkspaceId};
use redesmyn_logging::tracing;
use redesmyn_protocol::agent_commands::{
    ResumeByIdTaskAgentTurnCommand, SendTaskAgentMessageCommand, StartAgentSessionCommand,
    RespondPermissionRequestCommand, SetSessionPermissionsModeCommand,
    SESSION_AGENT_RESUME_BY_ID_TURN, SESSION_AGENT_RESPOND_PERMISSION_REQUEST,
    SESSION_AGENT_SEND_MESSAGE, SESSION_AGENT_SET_PERMISSIONS_MODE, SESSION_AGENT_START,
};
use redesmyn_protocol::client::{
    AgentInterfaceMode, AgentKind, AgentMessageConflictAction, AgentSessionScopeKind,
    AgentSessionStatus, AgentSessionSummary, ClientFrame, ClientMessage, CloseChatSessionResponse,
    CommandState, CommandSummary, CommandUpdateSummary, CreateChatSessionResponse,
    CreateCommandResponse, DaemonPresenceSummary, EpicGraph, EpicSummary, Event, EventLogEvent,
    EventWaitFilter, GetCommandResponse, GetEpicGraphResponse, GetEpicPinnedChatSessionResponse,
    GetLatestTaskSessionResponse, GetSessionEventsResponse, HealthResponse,
    ListChatSessionsResponse, ListEpicsResponse, ListTaskSessionsResponse, MergeReadiness,
    PinChatSessionToEpicResponse, Response, ResponseResult, SendSessionMessageResponse,
    RespondPermissionRequestResponse, SessionSummary, SetSessionPermissionsModeResponse,
    StatusResponse, Subscribed, SubscriptionEvent, TaskState,
    UnpinChatSessionFromEpicResponse, WaitForCommandResponse, WaitForEventResponse,
    WaitForIdleResponse,
};
use redesmyn_protocol::{
    ErrorCategory, ErrorDetail, ErrorEnvelope, ExternalSessionRef, ProtocolEnvelope,
    PermissionDecision, PermissionsMode, ProtocolVersion, Scope, SessionEvent, SessionEventKind,
    SessionScope, Timestamp, UserMessage,
};
use redesmyn_transport::client::codec::{Codec, JsonCodec, ProtobufCodec};
use redesmyn_transport::client::framed::FramedEndpoint;
use redesmyn_transport::client::{ClientConnection, ClientTransportError};

use redesmyn_storage::schema::{
    AgentInterfaceMode as StorageAgentInterfaceMode, AgentKind as StorageAgentKind,
    AgentSessionScopeKind as StorageAgentSessionScopeKind,
    AgentSessionStatus as StorageAgentSessionStatus,
};
use redesmyn_storage::sessions::AgentSessionRecord;

const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_IDLE_QUIESCENCE: Duration = Duration::from_millis(200);

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
            .and_then(|addr| addr.as_pathname().map(|p| p.display().to_string()))
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
    let mut request_tasks = tokio::task::JoinSet::new();
    let (outbound_tx, mut outbound_rx) = tokio::sync::mpsc::channel::<ClientFrame>(128);

    loop {
        tokio::select! {
            _ = shutdown.recv() => {
                abort_subscriptions(&mut subscriptions);
                request_tasks.abort_all();
                return Ok(());
            }
            frame = conn.recv() => {
                let frame = match frame {
                    Ok(frame) => frame,
                    Err(ClientTransportError::ChannelClosed) => {
                        abort_subscriptions(&mut subscriptions);
                        request_tasks.abort_all();
                        return Ok(());
                    }
                    Err(err) => {
                        abort_subscriptions(&mut subscriptions);
                        request_tasks.abort_all();
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
                        let request_envelope = frame.envelope.clone();
                        let task_control_plane = control_plane.clone();
                        let tx = outbound_tx.clone();

                        request_tasks.spawn(async move {
                            let response = handle_request(
                                req,
                                accepted,
                                &request_envelope,
                                &task_control_plane,
                            )
                            .await;
                            let response_frame = ClientFrame::new(
                                response_envelope(&request_envelope, accepted),
                                ClientMessage::Response(response),
                            );
                            let _ = tx.send(response_frame).await;
                        });
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
                                    let tx = outbound_tx.clone();
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
                                let tx = outbound_tx.clone();
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
                                            SessionEventsSubscriptionItem::Live(event) => {
                                                (SubscriptionEvent::SessionLiveEvent(event), false)
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
            Some(join_result) = request_tasks.join_next(), if !request_tasks.is_empty() => {
                if let Err(err) = join_result {
                    tracing::error!(error = %err, "client API request task panicked");
                }
            }
            maybe_frame = outbound_rx.recv() => {
                let Some(frame) = maybe_frame else {
                    continue;
                };
                conn.send(frame).await?;
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
    let default_timeout = || DEFAULT_WAIT_TIMEOUT;

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
            let epics =
                redesmyn_storage::epic_graph::list_epics(control_plane.pool(), scope).await?;
            Ok(ResponseResult::ListEpics(ListEpicsResponse {
                epics: epics
                    .into_iter()
                    .map(|epic| EpicSummary {
                        slug: epic.slug,
                        name: epic.title,
                        epic_id: Some(epic.epic_id),
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
            let load_span =
                tracing::debug_span!("client.api.load_epic_graph", epic_slug = %epic_slug);
            let Some(graph) = ({
                let _enter = load_span.enter();
                redesmyn_storage::epic_graph::load_epic_graph(
                    control_plane.pool(),
                    &epic_slug,
                    scope,
                )
                .await?
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
                    Ok((events, next_cursor)) => {
                        Ok(ResponseResult::GetSessionEvents(GetSessionEventsResponse {
                            events,
                            next_cursor,
                        }))
                    }
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
        redesmyn_protocol::client::RequestPayload::GetLatestTaskSession(get) => match control_plane
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
        },
        redesmyn_protocol::client::RequestPayload::GetEpicPinnedChatSession(get) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let exists = redesmyn_storage::sessions::epic_exists_in_repo(
                control_plane.pool(),
                workspace_id,
                repo_id,
                get.epic_id,
            )
            .await?;
            if !exists {
                let detail = ErrorDetail::from([("epic_id".to_string(), get.epic_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Epic not found.")
                        .with_detail(detail),
                ));
            }

            let session_id = redesmyn_storage::sessions::get_pinned_chat_session_for_epic(
                control_plane.pool(),
                get.epic_id,
            )
            .await?;

            Ok(ResponseResult::GetEpicPinnedChatSession(
                GetEpicPinnedChatSessionResponse { session_id },
            ))
        }
        redesmyn_protocol::client::RequestPayload::SendSessionMessage(send) => {
            let span = tracing::info_span!(
                "control_plane.client_api.send_session_message",
                session_id = %send.session_id,
                on_conflict = ?send.on_conflict
            );
            let _guard = span.enter();

            let trimmed = send.message.trim();
            if trimmed.is_empty() {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Message text is required.",
                )));
            }

            const MAX_TEXT_CHARS: usize = 20_000;
            if trimmed.chars().count() > MAX_TEXT_CHARS {
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::InvalidRequest, "Message text is too long.")
                        .with_detail(ErrorDetail::from([(
                            "max_chars".to_string(),
                            MAX_TEXT_CHARS.to_string(),
                        )])),
                ));
            }

            let text = send.message.trim_end();

            let session = redesmyn_storage::sessions::get_agent_session(
                control_plane.pool(),
                send.session_id,
            )
            .await?;
            let Some(session) = session else {
                let detail =
                    ErrorDetail::from([("session_id".to_string(), send.session_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Session not found.")
                        .with_detail(detail),
                ));
            };

            match envelope.scope {
                Some(Scope::Repo { repo }) => {
                    if session.scope_workspace_id != repo.workspace_id
                        || session.scope_repo_id != repo.repo_id
                    {
                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                            ErrorCategory::NotFound,
                            "Session not found in repo scope.",
                        )));
                    }
                }
                Some(Scope::Unknown) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "Unknown scope kind.",
                    )));
                }
                Some(_) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "Unsupported scope kind.",
                    )));
                }
                None => {}
            }

            let (session_id, scope, interface_mode) = match (session.scope_kind, session.task_id) {
                (StorageAgentSessionScopeKind::Chat, _) => {
                    if session.closed_at_ms.is_some() {
                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                            ErrorCategory::InvalidRequest,
                            "Chat session is closed.",
                        )));
                    }
                    let interface_mode = protocol_interface_mode_from_storage(session.interface_mode);
                    if send.on_conflict == AgentMessageConflictAction::StopSessionAndStartNew {
                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                            ErrorCategory::InvalidRequest,
                            "stop_session_and_start_new is not supported for chat sessions. Create a new chat session instead.",
                        )));
                    }
                    if matches!(
                        interface_mode,
                        AgentInterfaceMode::StructuredExec | AgentInterfaceMode::AppServer
                    ) && session.ended_at_ms.is_none()
                    {
                        let in_progress = match crate::turn_state::structured_turn_in_progress(
                            control_plane.pool(),
                            session.session_id,
                        )
                        .await
                        {
                            Ok(in_progress) => in_progress,
                            Err(err) => return Ok(ResponseResult::Error(err)),
                        };

                        if in_progress {
                            match send.on_conflict {
                                AgentMessageConflictAction::Fail => {
                                    return Ok(ResponseResult::Error(conflict_error(
                                        session.session_id,
                                        CONFLICT_CODE_TURN_IN_PROGRESS,
                                        "A structured agent turn is currently in progress.",
                                    )));
                                }
                                AgentMessageConflictAction::InterruptTurn => {
                                    let Some(external_session_ref) =
                                        parse_external_session_ref(&session.external_session_ref)
                                    else {
                                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                                            ErrorCategory::InvalidRequest,
                                            "Cannot interrupt: chat session is not resumable.",
                                        )));
                                    };
                                    if !external_ref_matches_agent_kind(
                                        &external_session_ref,
                                        protocol_agent_kind_from_storage(session.agent_kind),
                                    ) {
                                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                                            ErrorCategory::InvalidRequest,
                                            "Cannot interrupt: chat session is not resumable.",
                                        )));
                                    }
                                }
                                AgentMessageConflictAction::StopSessionAndStartNew => {}
                            }
                        }
                    }

                    (session.session_id, SessionScope::Chat, None)
                }
                (StorageAgentSessionScopeKind::Task, Some(task_id)) => (
                    session.session_id,
                    SessionScope::Task { task_id },
                    Some(session.interface_mode),
                ),
                (StorageAgentSessionScopeKind::Task, None) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Internal,
                        "Task session is missing task_id.",
                    )));
                }
            };

            let mut stop_session_ids = Vec::new();

            let (session_id, scope) = if let Some(interface_mode) = interface_mode {
                let task_id = match scope {
                    SessionScope::Task { task_id } => task_id,
                    _ => {
                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                            ErrorCategory::Internal,
                            "Expected task scope for task session.",
                        )));
                    }
                };

                let turn_in_progress =
                    if (interface_mode == StorageAgentInterfaceMode::StructuredExec
                        || interface_mode == StorageAgentInterfaceMode::AppServer)
                        && session.ended_at_ms.is_none()
                    {
                        match crate::turn_state::structured_turn_in_progress(
                            control_plane.pool(),
                            session.session_id,
                        )
                        .await
                        {
                            Ok(turn_in_progress) => turn_in_progress,
                            Err(err) => return Ok(ResponseResult::Error(err)),
                        }
                    } else {
                        false
                    };

                let sessions = redesmyn_storage::sessions::list_task_sessions(
                    control_plane.pool(),
                    task_id,
                    50,
                )
                .await?;
                let other_active_task_session = sessions.iter().any(|row| {
                    row.session_id != session.session_id
                        && row.ended_at_ms.is_none()
                        && matches!(
                            row.status,
                            StorageAgentSessionStatus::Running | StorageAgentSessionStatus::Blocked
                        )
                });

                let mut session_id = session_id;
                let mut scope = scope;

                match send.on_conflict {
                    AgentMessageConflictAction::Fail => {
                        if turn_in_progress {
                            return Ok(ResponseResult::Error(conflict_error(
                                session.session_id,
                                CONFLICT_CODE_TURN_IN_PROGRESS,
                                "A structured agent turn is currently in progress.",
                            )));
                        }
                        if other_active_task_session {
                            return Ok(ResponseResult::Error(conflict_error(
                                session.session_id,
                                CONFLICT_CODE_SESSION_CONFLICT,
                                "Another agent session is already running for this task.",
                            )));
                        }
                    }
                    AgentMessageConflictAction::InterruptTurn => {
                        if other_active_task_session && !turn_in_progress {
                            return Ok(ResponseResult::Error(ErrorEnvelope::new(
                                ErrorCategory::InvalidRequest,
                                "Cannot interrupt: another session is running. Use stop_session_and_start_new instead.",
                            )));
                        }

                        if turn_in_progress {
                            let Some(external_session_ref) =
                                parse_external_session_ref(&session.external_session_ref)
                            else {
                                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                                    ErrorCategory::InvalidRequest,
                                    "Cannot interrupt: session is not resumable.",
                                )));
                            };
                            if !external_ref_matches_agent_kind(
                                &external_session_ref,
                                protocol_agent_kind_from_storage(session.agent_kind),
                            ) {
                                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                                    ErrorCategory::InvalidRequest,
                                    "Cannot interrupt: session is not resumable.",
                                )));
                            }
                        }
                    }
                    AgentMessageConflictAction::StopSessionAndStartNew => {
                        if turn_in_progress {
                            return Ok(ResponseResult::Error(conflict_error(
                                session.session_id,
                                CONFLICT_CODE_TURN_IN_PROGRESS,
                                "A structured agent turn is currently in progress. Interrupt the current turn before starting a new session.",
                            )));
                        }

                        // TODO(T41): Replace direct session termination with daemon command(s) so the
                        // runtime can shut down cleanly before we create a new session row.
                        stop_session_ids = sessions
                            .iter()
                            .filter(|row| row.ended_at_ms.is_none())
                            .map(|row| row.session_id)
                            .collect();
                        let _ended = redesmyn_storage::sessions::end_task_sessions(
                            control_plane.pool(),
                            task_id,
                        )
                        .await?;

                        session_id = redesmyn_storage::sessions::create_task_session(
                            control_plane.pool(),
                            session.scope_workspace_id,
                            session.scope_repo_id,
                            task_id,
                            session.agent_kind,
                            interface_mode,
                            None,
                        )
                        .await?;

                        scope = SessionScope::Task { task_id };
                    }
                }

                (session_id, scope)
            } else {
                (session_id, scope)
            };

            let event = SessionEvent {
                session_event_id: SessionEventId::new(),
                created_at: Timestamp::now_utc(),
                scope,
                session_id,
                turn_id: None,
                kind: SessionEventKind::UserMessage(UserMessage {
                    text: text.to_string(),
                    preview: session_message_preview(trimmed),
                    full_text_artifact: None,
                }),
            };

            control_plane
                .session_events()
                .append_session_event(&event)
                .await?;

            let current = redesmyn_storage::sessions::get_agent_session(
                control_plane.pool(),
                session_id,
            )
            .await?;
            let Some(current) = current else {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "Session disappeared while sending message.",
                )));
            };

            let agent_kind = protocol_agent_kind_from_storage(current.agent_kind);
            let interface_mode = protocol_interface_mode_from_storage(current.interface_mode);
            let task_id = current.task_id;

            let turn_in_progress = matches!(
                interface_mode,
                AgentInterfaceMode::StructuredExec | AgentInterfaceMode::AppServer
            ) && match crate::turn_state::structured_turn_in_progress(control_plane.pool(), session_id)
                .await
            {
                Ok(turn_in_progress) => turn_in_progress,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let external_session_ref = parse_external_session_ref(&current.external_session_ref)
                .filter(|r| external_ref_matches_agent_kind(r, agent_kind));

            let interrupt_turn = send.on_conflict == AgentMessageConflictAction::InterruptTurn
                && turn_in_progress;

            let (command_kind, json_payload) = match interface_mode {
                AgentInterfaceMode::StructuredExec | AgentInterfaceMode::AppServer => {
                    if let Some(external_session_ref) = external_session_ref {
                        let json_payload = match encode_agent_command_payload(
                            &ResumeByIdTaskAgentTurnCommand {
                                session_id,
                                task_id,
                                prompt: text.to_string(),
                                external_session_ref,
                                interrupt_turn,
                            },
                        ) {
                            Ok(payload) => payload,
                            Err(err) => return Ok(ResponseResult::Error(err)),
                        };

                        (SESSION_AGENT_RESUME_BY_ID_TURN.to_string(), json_payload)
                    } else {
                        let json_payload = match encode_agent_command_payload(
                            &StartAgentSessionCommand {
                                session_id,
                                task_id,
                                agent_kind,
                                interface_mode,
                                initial_prompt: Some(text.to_string()),
                                stop_session_ids,
                            },
                        ) {
                            Ok(payload) => payload,
                            Err(err) => return Ok(ResponseResult::Error(err)),
                        };

                        (SESSION_AGENT_START.to_string(), json_payload)
                    }
                }
                AgentInterfaceMode::ShellTmux => {
                    let json_payload = match encode_agent_command_payload(
                        &SendTaskAgentMessageCommand {
                            session_id,
                            text: text.to_string(),
                            interrupt_turn,
                            submit: true,
                        },
                    ) {
                        Ok(payload) => payload,
                        Err(err) => return Ok(ResponseResult::Error(err)),
                    };

                    (SESSION_AGENT_SEND_MESSAGE.to_string(), json_payload)
                }
            };

            let command = control_plane
                .issue_command(
                    redesmyn_storage::commands::CommandScope::Repo {
                        workspace_id: current.scope_workspace_id,
                        repo_id: current.scope_repo_id,
                    },
                    command_kind,
                    task_id,
                    None,
                    None,
                    json_payload,
                )
                .await?;

            Ok(ResponseResult::SendSessionMessage(
                SendSessionMessageResponse {
                    event,
                    session_id,
                    command: Some(command),
                },
            ))
        }
        redesmyn_protocol::client::RequestPayload::SetSessionPermissionsMode(req) => {
            let span = tracing::info_span!(
                "control_plane.client_api.set_session_permissions_mode",
                session_id = %req.session_id,
                mode = ?req.mode,
            );
            let _guard = span.enter();

            if matches!(req.mode, PermissionsMode::Unknown) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Unknown permissions mode.",
                )));
            }

            let session =
                redesmyn_storage::sessions::get_agent_session(control_plane.pool(), req.session_id)
                    .await?;
            let Some(session) = session else {
                let detail =
                    ErrorDetail::from([("session_id".to_string(), req.session_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Session not found.")
                        .with_detail(detail),
                ));
            };

            match envelope.scope {
                Some(Scope::Repo { repo }) => {
                    if session.scope_workspace_id != repo.workspace_id
                        || session.scope_repo_id != repo.repo_id
                    {
                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                            ErrorCategory::NotFound,
                            "Session not found in repo scope.",
                        )));
                    }
                }
                Some(Scope::Unknown) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "Unknown scope kind.",
                    )));
                }
                Some(_) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "Unsupported scope kind.",
                    )));
                }
                None => {}
            }

            let interface_mode = protocol_interface_mode_from_storage(session.interface_mode);
            if !matches!(
                interface_mode,
                AgentInterfaceMode::StructuredExec | AgentInterfaceMode::AppServer
            ) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Permissions mode is only supported for structured sessions.",
                )));
            }

            let json_payload = match encode_agent_command_payload(&SetSessionPermissionsModeCommand {
                session_id: req.session_id,
                mode: req.mode,
            }) {
                Ok(payload) => payload,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let command = control_plane
                .issue_command(
                    redesmyn_storage::commands::CommandScope::Repo {
                        workspace_id: session.scope_workspace_id,
                        repo_id: session.scope_repo_id,
                    },
                    SESSION_AGENT_SET_PERMISSIONS_MODE.to_string(),
                    session.task_id,
                    None,
                    None,
                    json_payload,
                )
                .await?;

            Ok(ResponseResult::SetSessionPermissionsMode(
                SetSessionPermissionsModeResponse {
                    command: Some(command),
                },
            ))
        }
        redesmyn_protocol::client::RequestPayload::RespondPermissionRequest(req) => {
            let span = tracing::info_span!(
                "control_plane.client_api.respond_permission_request",
                session_id = %req.session_id,
                request_id = %req.request_id,
                decision = ?req.decision,
            );
            let _guard = span.enter();

            if req.request_id.trim().is_empty() {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "request_id is required.",
                )));
            }

            if matches!(req.decision, PermissionDecision::Unknown) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Unknown permission decision.",
                )));
            }

            let session =
                redesmyn_storage::sessions::get_agent_session(control_plane.pool(), req.session_id)
                    .await?;
            let Some(session) = session else {
                let detail =
                    ErrorDetail::from([("session_id".to_string(), req.session_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Session not found.")
                        .with_detail(detail),
                ));
            };

            match envelope.scope {
                Some(Scope::Repo { repo }) => {
                    if session.scope_workspace_id != repo.workspace_id
                        || session.scope_repo_id != repo.repo_id
                    {
                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                            ErrorCategory::NotFound,
                            "Session not found in repo scope.",
                        )));
                    }
                }
                Some(Scope::Unknown) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "Unknown scope kind.",
                    )));
                }
                Some(_) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "Unsupported scope kind.",
                    )));
                }
                None => {}
            }

            let interface_mode = protocol_interface_mode_from_storage(session.interface_mode);
            if !matches!(
                interface_mode,
                AgentInterfaceMode::StructuredExec | AgentInterfaceMode::AppServer
            ) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Permission requests are only supported for structured sessions.",
                )));
            }

            let json_payload = match encode_agent_command_payload(&RespondPermissionRequestCommand {
                session_id: req.session_id,
                request_id: req.request_id.clone(),
                decision: req.decision,
            }) {
                Ok(payload) => payload,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let command = control_plane
                .issue_command(
                    redesmyn_storage::commands::CommandScope::Repo {
                        workspace_id: session.scope_workspace_id,
                        repo_id: session.scope_repo_id,
                    },
                    SESSION_AGENT_RESPOND_PERMISSION_REQUEST.to_string(),
                    session.task_id,
                    None,
                    None,
                    json_payload,
                )
                .await?;

            Ok(ResponseResult::RespondPermissionRequest(
                RespondPermissionRequestResponse {
                    command: Some(command),
                },
            ))
        }
        redesmyn_protocol::client::RequestPayload::StartAgent(req) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            match control_plane.start_agent(workspace_id, repo_id, req).await {
                Ok(resp) => Ok(ResponseResult::StartAgent(resp)),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::StopAgent(req) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            match control_plane.stop_agent(workspace_id, repo_id, req).await {
                Ok(resp) => Ok(ResponseResult::StopAgent(resp)),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::RestartAgent(req) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            match control_plane
                .restart_agent(workspace_id, repo_id, req)
                .await
            {
                Ok(resp) => Ok(ResponseResult::RestartAgent(resp)),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::SendTaskAgentMessage(req) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            match control_plane
                .send_task_agent_message(workspace_id, repo_id, req)
                .await
            {
                Ok(resp) => Ok(ResponseResult::SendTaskAgentMessage(resp)),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::AttachAgentSession(req) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            match control_plane
                .attach_agent_session(workspace_id, repo_id, req)
                .await
            {
                Ok(resp) => Ok(ResponseResult::AttachAgentSession(resp)),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::CreateCommand(req) => {
            let scope = match command_scope_from_envelope(envelope) {
                Ok(scope) => scope,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let command = control_plane
                .issue_command(
                    scope,
                    req.kind,
                    req.target_task_id,
                    req.idempotency_key,
                    req.created_by,
                    req.json_payload,
                )
                .await?;

            Ok(ResponseResult::CreateCommand(CreateCommandResponse {
                command,
            }))
        }
        redesmyn_protocol::client::RequestPayload::GetCommand(req) => {
            match control_plane.commands().get_command(req.command_id).await? {
                Some(command) => Ok(ResponseResult::GetCommand(GetCommandResponse { command })),
                None => Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Command not found.").with_detail(
                        ErrorDetail::from([("command_id".to_string(), req.command_id.to_string())]),
                    ),
                )),
            }
        }
        redesmyn_protocol::client::RequestPayload::WaitForCommand(req) => {
            let span = tracing::debug_span!(
                "client.api.wait_for_command",
                request_id = %envelope.msg_id,
                command_id = %req.command_id,
            );
            let _enter = span.enter();

            let timeout = if req.timeout_ms == 0 {
                default_timeout()
            } else {
                Duration::from_millis(req.timeout_ms)
            };

            match control_plane
                .commands()
                .wait_for_command(req.command_id, &req.terminal_states, timeout)
                .await
            {
                Ok(command) => Ok(ResponseResult::WaitForCommand(WaitForCommandResponse {
                    command,
                })),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::WaitForEvent(req) => {
            let timeout = if req.timeout_ms == 0 {
                default_timeout()
            } else {
                Duration::from_millis(req.timeout_ms)
            };

            let scope = match storage_scope_from_envelope(envelope) {
                Ok(scope) => scope,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            match wait_for_event_log(control_plane, scope, &req.filter, timeout).await {
                Ok(event_log) => Ok(ResponseResult::WaitForEvent(WaitForEventResponse {
                    event_log,
                })),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::WaitForIdle(req) => {
            let timeout = if req.timeout_ms == 0 {
                default_timeout()
            } else {
                Duration::from_millis(req.timeout_ms)
            };
            let quiescence = if req.quiescence_ms == 0 {
                DEFAULT_IDLE_QUIESCENCE
            } else {
                Duration::from_millis(req.quiescence_ms)
            };

            let scope = req.scope.or(envelope.scope);
            let event_scope = match storage_scope_from_scope(scope) {
                Ok(scope) => scope,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };
            let command_scope = match command_scope_from_scope(scope) {
                Ok(scope) => scope,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            match wait_for_idle(
                control_plane,
                command_scope,
                event_scope,
                timeout,
                quiescence,
            )
            .await
            {
                Ok(()) => Ok(ResponseResult::WaitForIdle(WaitForIdleResponse {})),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::ListTaskSessions(get) => {
            const MAX_LIMIT: u32 = 512;
            if get.limit == 0 {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "ListTaskSessions.limit must be > 0.",
                )));
            }
            if get.limit > MAX_LIMIT {
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "ListTaskSessions.limit is too large.",
                    )
                    .with_detail(ErrorDetail::from([(
                        "max_limit".to_string(),
                        MAX_LIMIT.to_string(),
                    )])),
                ));
            }

            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let exists = redesmyn_storage::sessions::task_exists_in_repo(
                control_plane.pool(),
                workspace_id,
                repo_id,
                get.task_id,
            )
            .await?;
            if !exists {
                let detail = ErrorDetail::from([("task_id".to_string(), get.task_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Task not found.")
                        .with_detail(detail),
                ));
            }

            let sessions = redesmyn_storage::sessions::list_task_sessions(
                control_plane.pool(),
                get.task_id,
                get.limit,
            )
            .await?;

            let active_session_id = sessions
                .iter()
                .find(|session| session.ended_at_ms.is_none())
                .map(|session| session.session_id);

            let sessions = sessions
                .into_iter()
                .map(agent_session_summary_from_record)
                .collect();

            Ok(ResponseResult::ListTaskSessions(ListTaskSessionsResponse {
                sessions,
                active_session_id,
            }))
        }
        redesmyn_protocol::client::RequestPayload::CreateChatSession(create) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            // TODO: Plumb agent kind / interface mode from the request (or client identity).
            // For now we default to Codex + app-server so chat sessions can stream structured
            // events back into the durable session log.
            let session_id = redesmyn_storage::sessions::create_chat_session(
                control_plane.pool(),
                workspace_id,
                repo_id,
                StorageAgentKind::Codex,
                StorageAgentInterfaceMode::AppServer,
                create.title.as_deref(),
            )
            .await?;

            Ok(ResponseResult::CreateChatSession(
                CreateChatSessionResponse { session_id },
            ))
        }
        redesmyn_protocol::client::RequestPayload::CloseChatSession(close) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let session = redesmyn_storage::sessions::get_agent_session(
                control_plane.pool(),
                close.session_id,
            )
            .await?;
            let Some(session) = session else {
                let detail =
                    ErrorDetail::from([("session_id".to_string(), close.session_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Session not found.")
                        .with_detail(detail),
                ));
            };

            if session.scope_workspace_id != workspace_id || session.scope_repo_id != repo_id {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::NotFound,
                    "Session not found in repo scope.",
                )));
            }
            if session.scope_kind != StorageAgentSessionScopeKind::Chat {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Session is not a chat session.",
                )));
            }

            redesmyn_storage::sessions::close_chat_session(control_plane.pool(), close.session_id)
                .await?;

            Ok(ResponseResult::CloseChatSession(
                CloseChatSessionResponse {},
            ))
        }
        redesmyn_protocol::client::RequestPayload::ListChatSessions(list) => {
            const MAX_LIMIT: u32 = 512;
            if list.limit == 0 {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "ListChatSessions.limit must be > 0.",
                )));
            }
            if list.limit > MAX_LIMIT {
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "ListChatSessions.limit is too large.",
                    )
                    .with_detail(ErrorDetail::from([(
                        "max_limit".to_string(),
                        MAX_LIMIT.to_string(),
                    )])),
                ));
            }

            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let sessions = redesmyn_storage::sessions::list_chat_sessions(
                control_plane.pool(),
                workspace_id,
                repo_id,
                list.include_closed,
                list.limit,
            )
            .await?;

            let sessions = sessions
                .into_iter()
                .map(agent_session_summary_from_record)
                .collect();

            Ok(ResponseResult::ListChatSessions(ListChatSessionsResponse {
                sessions,
            }))
        }
        redesmyn_protocol::client::RequestPayload::PinChatSessionToEpic(pin) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let epic_exists = redesmyn_storage::sessions::epic_exists_in_repo(
                control_plane.pool(),
                workspace_id,
                repo_id,
                pin.epic_id,
            )
            .await?;
            if !epic_exists {
                let detail = ErrorDetail::from([("epic_id".to_string(), pin.epic_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Epic not found.")
                        .with_detail(detail),
                ));
            }

            let session =
                redesmyn_storage::sessions::get_agent_session(control_plane.pool(), pin.session_id)
                    .await?;
            let Some(session) = session else {
                let detail =
                    ErrorDetail::from([("session_id".to_string(), pin.session_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Session not found.")
                        .with_detail(detail),
                ));
            };

            if session.scope_workspace_id != workspace_id || session.scope_repo_id != repo_id {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::NotFound,
                    "Session not found in repo scope.",
                )));
            }
            if session.scope_kind != StorageAgentSessionScopeKind::Chat {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Only chat sessions can be pinned.",
                )));
            }

            redesmyn_storage::sessions::pin_chat_session_to_epic(
                control_plane.pool(),
                pin.epic_id,
                pin.session_id,
            )
            .await?;

            Ok(ResponseResult::PinChatSessionToEpic(
                PinChatSessionToEpicResponse {},
            ))
        }
        redesmyn_protocol::client::RequestPayload::UnpinChatSessionFromEpic(unpin) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let epic_exists = redesmyn_storage::sessions::epic_exists_in_repo(
                control_plane.pool(),
                workspace_id,
                repo_id,
                unpin.epic_id,
            )
            .await?;
            if !epic_exists {
                let detail =
                    ErrorDetail::from([("epic_id".to_string(), unpin.epic_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Epic not found.")
                        .with_detail(detail),
                ));
            }

            redesmyn_storage::sessions::unpin_chat_session_from_epic(
                control_plane.pool(),
                unpin.epic_id,
            )
            .await?;

            Ok(ResponseResult::UnpinChatSessionFromEpic(
                UnpinChatSessionFromEpicResponse {},
            ))
        }
    }
}

fn require_repo_scope_ids(
    envelope: &ProtocolEnvelope,
) -> Result<(WorkspaceId, RepoId), ErrorEnvelope> {
    match envelope.scope {
        Some(Scope::Repo { repo }) => Ok((repo.workspace_id, repo.repo_id)),
        Some(Scope::Unknown) => Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Unknown scope kind.",
        )),
        Some(_) => Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Unsupported scope kind.",
        )),
        None => Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Repo scope is required.",
        )),
    }
}

fn conflict_error(
    session_id: SessionId,
    code: &'static str,
    message: &'static str,
) -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCategory::Conflict, message).with_detail(ErrorDetail::from([
        ("session_id".to_string(), session_id.to_string()),
        (CONFLICT_CODE_KEY.to_string(), code.to_string()),
    ]))
}

fn session_message_preview(text: &str) -> String {
    const MAX_PREVIEW_CHARS: usize = 140;
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");

    normalized.chars().take(MAX_PREVIEW_CHARS).collect()
}

fn agent_session_summary_from_record(record: AgentSessionRecord) -> AgentSessionSummary {
    let scope_kind = match record.scope_kind {
        StorageAgentSessionScopeKind::Task => AgentSessionScopeKind::Task,
        StorageAgentSessionScopeKind::Chat => AgentSessionScopeKind::Chat,
    };

    let agent_kind = protocol_agent_kind_from_storage(record.agent_kind);
    let interface_mode = protocol_interface_mode_from_storage(record.interface_mode);

    let status = match record.status {
        StorageAgentSessionStatus::Running => AgentSessionStatus::Running,
        StorageAgentSessionStatus::Blocked => AgentSessionStatus::Blocked,
        StorageAgentSessionStatus::Stopped => AgentSessionStatus::Stopped,
        StorageAgentSessionStatus::Error => AgentSessionStatus::Error,
    };

    AgentSessionSummary {
        session_id: record.session_id,
        scope_kind,
        task_id: record.task_id,
        agent_kind,
        interface_mode,
        status,
        title: record.title,
        closed_at: record.closed_at_ms.map(timestamp_from_ms),
        created_at: Some(timestamp_from_ms(record.created_at_ms)),
        updated_at: Some(timestamp_from_ms(record.updated_at_ms)),
        ended_at: record.ended_at_ms.map(timestamp_from_ms),
    }
}

fn protocol_agent_kind_from_storage(kind: StorageAgentKind) -> AgentKind {
    match kind {
        StorageAgentKind::Codex => AgentKind::Codex,
        StorageAgentKind::ClaudeCode => AgentKind::ClaudeCode,
        StorageAgentKind::Shell => AgentKind::Shell,
    }
}

fn protocol_interface_mode_from_storage(mode: StorageAgentInterfaceMode) -> AgentInterfaceMode {
    match mode {
        StorageAgentInterfaceMode::ShellTmux => AgentInterfaceMode::ShellTmux,
        StorageAgentInterfaceMode::StructuredExec => AgentInterfaceMode::StructuredExec,
        StorageAgentInterfaceMode::AppServer => AgentInterfaceMode::AppServer,
    }
}

fn parse_external_session_ref(raw: &str) -> Option<ExternalSessionRef> {
    serde_json::from_str(raw).ok()
}

fn external_ref_matches_agent_kind(external_session_ref: &ExternalSessionRef, agent_kind: AgentKind) -> bool {
    matches!(
        (agent_kind, external_session_ref),
        (AgentKind::Codex, ExternalSessionRef::CodexThread { .. })
            | (AgentKind::Codex, ExternalSessionRef::CodexSession { .. })
            | (AgentKind::ClaudeCode, ExternalSessionRef::ClaudeSession { .. })
    )
}

fn encode_agent_command_payload<T: serde::Serialize>(payload: &T) -> Result<Vec<u8>, ErrorEnvelope> {
    serde_json::to_vec(payload).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to encode agent command payload.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })
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
    storage_scope_from_scope(envelope.scope)
}

fn storage_scope_from_scope(
    scope: Option<Scope>,
) -> Result<redesmyn_storage::events::EventScope, ErrorEnvelope> {
    match scope {
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

fn command_scope_from_envelope(
    envelope: &ProtocolEnvelope,
) -> Result<redesmyn_storage::commands::CommandScope, ErrorEnvelope> {
    command_scope_from_scope(envelope.scope)
}

fn command_scope_from_scope(
    scope: Option<Scope>,
) -> Result<redesmyn_storage::commands::CommandScope, ErrorEnvelope> {
    match scope {
        None => Ok(redesmyn_storage::commands::CommandScope::None),
        Some(Scope::Repo { repo }) => Ok(redesmyn_storage::commands::CommandScope::Repo {
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

async fn wait_for_event_log(
    control_plane: &ControlPlane,
    scope: redesmyn_storage::events::EventScope,
    filter: &EventWaitFilter,
    timeout: Duration,
) -> Result<EventLogEvent, ErrorEnvelope> {
    let mut sub = control_plane
        .event_log()
        .subscribe(scope, filter.after_event_id);

    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());

        let item = match tokio::time::timeout(remaining, sub.recv()).await {
            Ok(Some(item)) => item,
            Ok(None) => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "Event log subscription closed.",
                ));
            }
            Err(_) => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "Timed out waiting for event.",
                ));
            }
        };

        match item {
            crate::event_log::EventLogSubscriptionItem::Event(record) => {
                if filter.event_type_prefix.is_empty()
                    || record.kind.starts_with(&filter.event_type_prefix)
                {
                    return Ok(EventLogEvent {
                        event_id: record.id,
                        occurred_at: timestamp_from_ms(record.created_at_ms),
                        event_type: record.kind,
                        json_payload: record.payload,
                    });
                }
            }
            crate::event_log::EventLogSubscriptionItem::ResyncRequired(resync) => {
                return Err(resync_error_envelope(resync));
            }
        }
    }
}

async fn wait_for_idle(
    control_plane: &ControlPlane,
    command_scope: redesmyn_storage::commands::CommandScope,
    event_scope: redesmyn_storage::events::EventScope,
    timeout: Duration,
    quiescence: Duration,
) -> Result<(), ErrorEnvelope> {
    let mut sub = control_plane.event_log().subscribe(event_scope, None);
    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Timed out waiting for idle.",
            ));
        }

        let inflight = control_plane
            .commands()
            .inflight_count(command_scope)
            .await
            .map_err(ErrorEnvelope::from)?;

        if inflight == 0 {
            let quiescence_deadline = tokio::time::Instant::now() + quiescence;

            tokio::select! {
                _ = tokio::time::sleep_until(quiescence_deadline) => {
                    let inflight = control_plane
                        .commands()
                        .inflight_count(command_scope)
                        .await
                        .map_err(ErrorEnvelope::from)?;

                    if inflight == 0 {
                        return Ok(());
                    }
                }
                item = sub.recv() => {
                    let Some(item) = item else {
                        return Ok(());
                    };

                    match item {
                        crate::event_log::EventLogSubscriptionItem::Event(_) => {}
                        crate::event_log::EventLogSubscriptionItem::ResyncRequired(_) => {}
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "Timed out waiting for idle.",
                    ));
                }
            }
        } else {
            tokio::select! {
                item = sub.recv() => {
                    let Some(item) = item else {
                        return Ok(());
                    };

                    match item {
                        crate::event_log::EventLogSubscriptionItem::Event(_) => {}
                        crate::event_log::EventLogSubscriptionItem::ResyncRequired(_) => {}
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "Timed out waiting for idle.",
                    ));
                }
            }
        }
    }
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
        redesmyn_storage::schema::CommandState::Queued => CommandState::Queued,
        redesmyn_storage::schema::CommandState::Accepted => CommandState::Accepted,
        redesmyn_storage::schema::CommandState::Running => CommandState::Running,
        redesmyn_storage::schema::CommandState::Blocked => CommandState::Blocked,
        redesmyn_storage::schema::CommandState::Resumable => CommandState::Resumable,
        redesmyn_storage::schema::CommandState::Succeeded => CommandState::Succeeded,
        redesmyn_storage::schema::CommandState::Failed => CommandState::Failed,
        redesmyn_storage::schema::CommandState::Canceled => CommandState::Canceled,
    }
}
