use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
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
    AGENT_LIST_MODELS, ListAgentModelsCommand, ListSessionModelsCommand,
    RespondPermissionRequestCommand, ResumeByIdTaskAgentTurnCommand, SESSION_AGENT_LIST_MODELS,
    SESSION_AGENT_RESPOND_PERMISSION_REQUEST, SESSION_AGENT_RESUME_BY_ID_TURN,
    SESSION_AGENT_SET_CODEX_APPROVAL_POLICY, SESSION_AGENT_SET_CODEX_SANDBOX_POLICY,
    SESSION_AGENT_SET_MODEL, SESSION_AGENT_SET_PERMISSIONS_MODE, SESSION_AGENT_START,
    SetSessionCodexApprovalPolicyCommand, SetSessionCodexSandboxPolicyCommand,
    SetSessionModelCommand, SetSessionPermissionsModeCommand, StartAgentSessionCommand,
};
use redesmyn_protocol::client::{
    AgentKind, AgentMessageConflictAction, AgentSessionScopeKind, AgentSessionStatus,
    AgentSessionSummary, ArchiveChatSessionResponse, ClientFrame, ClientMessage, CommandState,
    CommandSummary, CommandUpdateSummary, CreateChatSessionResponse, CreateCommandResponse,
    DaemonPresenceSummary, EpicGraph, EpicSummary, Event, EventLogEvent, EventWaitFilter,
    GetCommandResponse, GetEpicGraphResponse, GetEpicPinnedChatSessionResponse,
    GetLatestTaskSessionResponse, GetSessionEventsResponse, HealthResponse,
    ListAgentModelsResponse, ListChatSessionsResponse, ListEpicsResponse,
    ListSessionModelsResponse, ListTaskSessionsResponse, MergeReadiness,
    PinChatSessionToEpicResponse, RegenerateChatSessionTitleResponse,
    RespondPermissionRequestResponse, Response, ResponseResult,
    SEND_SESSION_MESSAGE_MAX_IMAGE_ATTACHMENT_BYTES, SEND_SESSION_MESSAGE_MAX_IMAGE_ATTACHMENTS,
    SEND_SESSION_MESSAGE_MAX_IMAGE_TOTAL_BYTES, SendSessionMessageResponse, SessionModelSelection,
    SessionSummary, SetSessionCodexApprovalPolicyResponse, SetSessionCodexSandboxPolicyResponse,
    SetSessionModelResponse, SetSessionPermissionsModeResponse, StatusResponse, Subscribed,
    SubscriptionEvent, TaskState, UnpinChatSessionFromEpicResponse, WaitForCommandResponse,
    WaitForEventResponse, WaitForIdleResponse,
};
use redesmyn_protocol::{
    ArtifactKind, CodexApprovalPolicy, CodexSandboxPolicy, ErrorCategory, ErrorDetail,
    ErrorEnvelope, ExternalSessionRef, PermissionDecision, PermissionsMode, ProtocolEnvelope,
    ProtocolVersion, Scope, SessionEvent, SessionEventKind, SessionScope, Timestamp, UserMessage,
};
use redesmyn_transport::client::codec::{Codec, JsonCodec, ProtobufCodec};
use redesmyn_transport::client::framed::FramedEndpoint;
use redesmyn_transport::client::{ClientConnection, ClientTransportError};

use redesmyn_storage::schema::{
    AgentKind as StorageAgentKind, AgentSessionScopeKind as StorageAgentSessionScopeKind,
    AgentSessionStatus as StorageAgentSessionStatus, CommandState as StorageCommandState,
};
use redesmyn_storage::sessions::AgentSessionRecord;

const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_IDLE_QUIESCENCE: Duration = Duration::from_millis(200);
const OPENAI_CHAT_TITLE_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
const OPENAI_CHAT_TITLE_MODEL: &str = "gpt-4o-mini";
const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const REDESMYN_OPENAI_API_KEY_ENV: &str = "REDESMYN_RUST__CONTROL_PLANE__OPENAI_API_KEY";
const CHAT_TITLE_SOURCE_MAX_CHARS: usize = 2_000;
const CHAT_TITLE_MAX_CHARS: usize = 96;
static OPENAI_API_KEY_OVERRIDE: OnceLock<RwLock<Option<String>>> = OnceLock::new();

fn openai_api_key_override_store() -> &'static RwLock<Option<String>> {
    OPENAI_API_KEY_OVERRIDE.get_or_init(|| RwLock::new(None))
}

pub fn set_openai_api_key_override(api_key: Option<String>) {
    let normalized = api_key
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let mut guard = openai_api_key_override_store()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = normalized;
}

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
                let (stream, _addr) = match accept {
                    Ok(value) => value,
                    Err(err) => {
                        tracing::warn!(error = %err, "client API UDS accept failed; continuing");
                        continue;
                    }
                };
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
            let has_text = !trimmed.is_empty();
            if !has_text && send.image_attachments.is_empty() {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Message text or image attachment is required.",
                )));
            }

            const MAX_TEXT_CHARS: usize = 20_000;
            if has_text && trimmed.chars().count() > MAX_TEXT_CHARS {
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::InvalidRequest, "Message text is too long.")
                        .with_detail(ErrorDetail::from([(
                            "max_chars".to_string(),
                            MAX_TEXT_CHARS.to_string(),
                        )])),
                ));
            }
            if let Err(err) = validate_image_attachments(&send.image_attachments) {
                return Ok(ResponseResult::Error(err));
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

            let agent_kind = protocol_agent_kind_from_storage(session.agent_kind);
            if !is_structured_agent_kind(agent_kind) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Interactive sessions are not supported.",
                )));
            }

            let (session_id, scope) = match (session.scope_kind, session.task_id) {
                (StorageAgentSessionScopeKind::Chat, _) => {
                    if send.on_conflict == AgentMessageConflictAction::StopSessionAndStartNew {
                        return Ok(ResponseResult::Error(ErrorEnvelope::new(
                            ErrorCategory::InvalidRequest,
                            "stop_session_and_start_new is not supported for chat sessions. Create a new chat session instead.",
                        )));
                    }
                    if session.ended_at_ms.is_none() {
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

                    (session.session_id, SessionScope::Chat)
                }
                (StorageAgentSessionScopeKind::Task, Some(task_id)) => {
                    (session.session_id, SessionScope::Task { task_id })
                }
                (StorageAgentSessionScopeKind::Task, None) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Internal,
                        "Task session is missing task_id.",
                    )));
                }
            };

            let mut stop_session_ids = Vec::new();

            let (session_id, scope) = if let SessionScope::Task { task_id } = scope {
                let task_id = task_id;

                let turn_in_progress = if session.ended_at_ms.is_none() {
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
                            None,
                        )
                        .await?;
                    }
                }

                (session_id, SessionScope::Task { task_id })
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
                    preview: session_message_preview(trimmed, send.image_attachments.len()),
                    full_text_artifact: None,
                    image_attachments: send.image_attachments.clone(),
                }),
            };

            control_plane
                .session_events()
                .append_session_event(&event)
                .await?;

            maybe_spawn_chat_title_generation(
                control_plane.pool().clone(),
                &session,
                text,
                has_text,
            );

            let current =
                redesmyn_storage::sessions::get_agent_session(control_plane.pool(), session_id)
                    .await?;
            let Some(current) = current else {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "Session disappeared while sending message.",
                )));
            };

            let agent_kind = protocol_agent_kind_from_storage(current.agent_kind);
            let task_id = current.task_id;

            let turn_in_progress = match crate::turn_state::structured_turn_in_progress(
                control_plane.pool(),
                session_id,
            )
            .await
            {
                Ok(turn_in_progress) => turn_in_progress,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let external_session_ref = parse_external_session_ref(&current.external_session_ref)
                .filter(|r| external_ref_matches_agent_kind(r, agent_kind));

            let interrupt_turn =
                send.on_conflict == AgentMessageConflictAction::InterruptTurn && turn_in_progress;

            let policy_snapshot = Some(
                crate::policy_snapshot::load_session_policy_snapshot(
                    control_plane.session_events(),
                    session_id,
                )
                .await?,
            );

            let (command_kind, json_payload) = if let Some(external_session_ref) =
                external_session_ref
            {
                let json_payload =
                    match encode_agent_command_payload(&ResumeByIdTaskAgentTurnCommand {
                        session_id,
                        task_id,
                        prompt: text.to_string(),
                        image_attachments: send.image_attachments.clone(),
                        external_session_ref,
                        policy_snapshot: policy_snapshot.clone(),
                        interrupt_turn,
                    }) {
                        Ok(payload) => payload,
                        Err(err) => return Ok(ResponseResult::Error(err)),
                    };

                (SESSION_AGENT_RESUME_BY_ID_TURN.to_string(), json_payload)
            } else {
                let task_branch_name = if let Some(task_id) = task_id {
                    let branch_name: Option<String> = match sqlx::query_scalar(
                        r#"
                        SELECT branch_name
                        FROM tasks
                        WHERE id = ?1
                        LIMIT 1
                        "#,
                    )
                    .bind(task_id)
                    .fetch_optional(control_plane.pool())
                    .await
                    {
                        Ok(row) => row.flatten(),
                        Err(err) => {
                            return Ok(ResponseResult::Error(
                                ErrorEnvelope::new(
                                    ErrorCategory::Internal,
                                    "Failed to load task branch.",
                                )
                                .with_detail(ErrorDetail::from([
                                    ("error".to_string(), err.to_string()),
                                ])),
                            ));
                        }
                    };

                    match branch_name
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty())
                    {
                        Some(branch_name) => Some(branch_name),
                        None => {
                            let detail =
                                ErrorDetail::from([("task_id".to_string(), task_id.to_string())]);
                            return Ok(ResponseResult::Error(
                                ErrorEnvelope::new(
                                    ErrorCategory::InvalidRequest,
                                    "Task branch is required to start a task session.",
                                )
                                .with_detail(detail),
                            ));
                        }
                    }
                } else {
                    None
                };

                let json_payload = match encode_agent_command_payload(&StartAgentSessionCommand {
                    session_id,
                    task_id,
                    task_branch_name,
                    agent_kind,
                    initial_prompt: has_text.then(|| text.to_string()),
                    image_attachments: send.image_attachments.clone(),
                    policy_snapshot,
                    stop_session_ids,
                }) {
                    Ok(payload) => payload,
                    Err(err) => return Ok(ResponseResult::Error(err)),
                };

                (SESSION_AGENT_START.to_string(), json_payload)
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

            let agent_kind = protocol_agent_kind_from_storage(session.agent_kind);
            if !is_structured_agent_kind(agent_kind) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Permissions mode is only supported for structured sessions.",
                )));
            }

            let json_payload =
                match encode_agent_command_payload(&SetSessionPermissionsModeCommand {
                    session_id: req.session_id,
                    task_id: session.task_id,
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
        redesmyn_protocol::client::RequestPayload::SetSessionCodexApprovalPolicy(req) => {
            let span = tracing::info_span!(
                "control_plane.client_api.set_session_codex_approval_policy",
                session_id = %req.session_id,
                approval_policy = ?req.approval_policy,
            );
            let _guard = span.enter();

            if matches!(req.approval_policy, Some(CodexApprovalPolicy::Unknown)) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Unknown Codex approval policy.",
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

            let agent_kind = protocol_agent_kind_from_storage(session.agent_kind);
            if !is_structured_agent_kind(agent_kind) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Codex approval policy is only supported for structured sessions.",
                )));
            }

            let json_payload =
                match encode_agent_command_payload(&SetSessionCodexApprovalPolicyCommand {
                    session_id: req.session_id,
                    task_id: session.task_id,
                    approval_policy: req.approval_policy,
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
                    SESSION_AGENT_SET_CODEX_APPROVAL_POLICY.to_string(),
                    session.task_id,
                    None,
                    None,
                    json_payload,
                )
                .await?;

            Ok(ResponseResult::SetSessionCodexApprovalPolicy(
                SetSessionCodexApprovalPolicyResponse {
                    command: Some(command),
                },
            ))
        }
        redesmyn_protocol::client::RequestPayload::SetSessionCodexSandboxPolicy(req) => {
            let span = tracing::info_span!(
                "control_plane.client_api.set_session_codex_sandbox_policy",
                session_id = %req.session_id,
                sandbox_policy = ?req.sandbox_policy,
            );
            let _guard = span.enter();

            if matches!(req.sandbox_policy, Some(CodexSandboxPolicy::Unknown)) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Unknown Codex sandbox policy.",
                )));
            }

            if let Some(CodexSandboxPolicy::WorkspaceWrite { writable_roots, .. }) =
                req.sandbox_policy.as_ref()
                && !writable_roots.is_empty()
            {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "writable_roots is not supported via the control plane.",
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

            let agent_kind = protocol_agent_kind_from_storage(session.agent_kind);
            if !is_structured_agent_kind(agent_kind) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Codex sandbox policy is only supported for structured sessions.",
                )));
            }

            let json_payload =
                match encode_agent_command_payload(&SetSessionCodexSandboxPolicyCommand {
                    session_id: req.session_id,
                    task_id: session.task_id,
                    sandbox_policy: req.sandbox_policy,
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
                    SESSION_AGENT_SET_CODEX_SANDBOX_POLICY.to_string(),
                    session.task_id,
                    None,
                    None,
                    json_payload,
                )
                .await?;

            Ok(ResponseResult::SetSessionCodexSandboxPolicy(
                SetSessionCodexSandboxPolicyResponse {
                    command: Some(command),
                },
            ))
        }
        redesmyn_protocol::client::RequestPayload::ListAgentModels(req) => {
            let span = tracing::info_span!(
                "control_plane.client_api.list_agent_models",
                agent_kind = ?req.agent_kind,
            );
            let _guard = span.enter();

            let (workspace_id, repo_id) = match envelope.scope {
                Some(Scope::Repo { repo }) => (repo.workspace_id, repo.repo_id),
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
                None => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "Repo scope is required to list agent models.",
                    )));
                }
            };

            let json_payload = match encode_agent_command_payload(&ListAgentModelsCommand {
                agent_kind: req.agent_kind,
            }) {
                Ok(payload) => payload,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let command = control_plane
                .issue_command(
                    redesmyn_storage::commands::CommandScope::Repo {
                        workspace_id,
                        repo_id,
                    },
                    AGENT_LIST_MODELS.to_string(),
                    None,
                    None,
                    None,
                    json_payload,
                )
                .await?;

            let command = match control_plane
                .commands()
                .wait_for_command(
                    command.command_id,
                    &[
                        CommandState::Succeeded,
                        CommandState::Failed,
                        CommandState::Canceled,
                    ],
                    default_timeout(),
                )
                .await
            {
                Ok(command) => command,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            if command.state != CommandState::Succeeded {
                let message = command
                    .last_update
                    .as_ref()
                    .and_then(|update| update.message.clone())
                    .unwrap_or_else(|| "Failed to list agent models.".to_string());
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    message,
                )));
            }

            let last_update = redesmyn_storage::commands::get_command_last_update_for_state(
                control_plane.pool(),
                command.command_id,
                StorageCommandState::Succeeded,
            )
            .await?
            .or(redesmyn_storage::commands::get_command_last_update(
                control_plane.pool(),
                command.command_id,
            )
            .await?);

            let Some(last_update) = last_update else {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "Missing command update payload for model list.",
                )));
            };

            let Some(detail_bytes) = last_update.detail else {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "Missing model list payload.",
                )));
            };

            match decode_list_session_models_from_command_detail(&detail_bytes) {
                Ok(resp) => Ok(ResponseResult::ListAgentModels(ListAgentModelsResponse {
                    options: resp.options,
                })),
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::ListSessionModels(req) => {
            let span = tracing::info_span!(
                "control_plane.client_api.list_session_models",
                session_id = %req.session_id,
            );
            let _guard = span.enter();

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

            let agent_kind = protocol_agent_kind_from_storage(session.agent_kind);
            if !is_structured_agent_kind(agent_kind) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Session model selection is only supported for structured sessions.",
                )));
            }

            let snapshot = crate::policy_snapshot::load_session_policy_snapshot(
                control_plane.session_events(),
                req.session_id,
            )
            .await?;
            let durable_selection_override =
                if snapshot.model_id.is_some() || snapshot.model_reasoning_effort.is_some() {
                    Some(SessionModelSelection {
                        model_id: snapshot.model_id,
                        reasoning_effort: snapshot.model_reasoning_effort,
                    })
                } else {
                    None
                };

            let json_payload = match encode_agent_command_payload(&ListSessionModelsCommand {
                session_id: req.session_id,
                task_id: session.task_id,
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
                    SESSION_AGENT_LIST_MODELS.to_string(),
                    session.task_id,
                    None,
                    None,
                    json_payload,
                )
                .await?;

            let command = match control_plane
                .commands()
                .wait_for_command(
                    command.command_id,
                    &[
                        CommandState::Succeeded,
                        CommandState::Failed,
                        CommandState::Canceled,
                    ],
                    default_timeout(),
                )
                .await
            {
                Ok(command) => command,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            if command.state != CommandState::Succeeded {
                let message = command
                    .last_update
                    .as_ref()
                    .and_then(|update| update.message.clone())
                    .unwrap_or_else(|| "Failed to list session models.".to_string());
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    message,
                )));
            }

            let last_update = redesmyn_storage::commands::get_command_last_update_for_state(
                control_plane.pool(),
                command.command_id,
                StorageCommandState::Succeeded,
            )
            .await?
            .or(redesmyn_storage::commands::get_command_last_update(
                control_plane.pool(),
                command.command_id,
            )
            .await?);

            let Some(last_update) = last_update else {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "Missing command update payload for model list.",
                )));
            };

            let Some(detail_bytes) = last_update.detail else {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "Missing model list payload.",
                )));
            };

            match decode_list_session_models_from_command_detail(&detail_bytes) {
                Ok(mut resp) => {
                    if let Some(selection) = durable_selection_override {
                        resp.selection = selection;
                    }
                    Ok(ResponseResult::ListSessionModels(resp))
                }
                Err(err) => Ok(ResponseResult::Error(err)),
            }
        }
        redesmyn_protocol::client::RequestPayload::SetSessionModel(req) => {
            let span = tracing::info_span!(
                "control_plane.client_api.set_session_model",
                session_id = %req.session_id,
                model_id = ?req.selection.model_id,
                reasoning_effort = ?req.selection.reasoning_effort,
            );
            let _guard = span.enter();

            if matches!(
                req.selection.reasoning_effort,
                Some(redesmyn_protocol::client::ModelReasoningEffort::Unknown)
            ) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Unknown model reasoning effort.",
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

            let agent_kind = protocol_agent_kind_from_storage(session.agent_kind);
            if !is_structured_agent_kind(agent_kind) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Session model selection is only supported for structured sessions.",
                )));
            }

            let model_id = req.selection.model_id.as_ref().and_then(|value| {
                let trimmed = value.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_owned())
            });

            let json_payload = match encode_agent_command_payload(&SetSessionModelCommand {
                session_id: req.session_id,
                task_id: session.task_id,
                model_id,
                reasoning_effort: req.selection.reasoning_effort,
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
                    SESSION_AGENT_SET_MODEL.to_string(),
                    session.task_id,
                    None,
                    None,
                    json_payload,
                )
                .await?;

            let command = match control_plane
                .commands()
                .wait_for_command(
                    command.command_id,
                    &[
                        CommandState::Succeeded,
                        CommandState::Failed,
                        CommandState::Canceled,
                    ],
                    default_timeout(),
                )
                .await
            {
                Ok(command) => command,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            if command.state != CommandState::Succeeded {
                let message = command
                    .last_update
                    .as_ref()
                    .and_then(|update| update.message.clone())
                    .unwrap_or_else(|| "Failed to update session model.".to_string());
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    message,
                )));
            }

            Ok(ResponseResult::SetSessionModel(SetSessionModelResponse {
                command: Some(command),
            }))
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

            let agent_kind = protocol_agent_kind_from_storage(session.agent_kind);
            if !is_structured_agent_kind(agent_kind) {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Permission requests are only supported for structured sessions.",
                )));
            }

            let json_payload =
                match encode_agent_command_payload(&RespondPermissionRequestCommand {
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

            if let Some(epic_id) = create.epic_id {
                let epic_exists = redesmyn_storage::sessions::epic_exists_in_repo(
                    control_plane.pool(),
                    workspace_id,
                    repo_id,
                    epic_id,
                )
                .await?;
                if !epic_exists {
                    let detail = ErrorDetail::from([("epic_id".to_string(), epic_id.to_string())]);
                    return Ok(ResponseResult::Error(
                        ErrorEnvelope::new(ErrorCategory::NotFound, "Epic not found.")
                            .with_detail(detail),
                    ));
                }
            }

            // TODO: Plumb agent kind from the request (or client identity).
            // For now we default to Codex so chat sessions can stream structured events
            // back into the durable session log.
            let session_id = redesmyn_storage::sessions::create_chat_session(
                control_plane.pool(),
                workspace_id,
                repo_id,
                create.epic_id,
                StorageAgentKind::Codex,
                create.title.as_deref(),
            )
            .await?;

            Ok(ResponseResult::CreateChatSession(
                CreateChatSessionResponse { session_id },
            ))
        }
        redesmyn_protocol::client::RequestPayload::ArchiveChatSession(archive) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

            let session = redesmyn_storage::sessions::get_agent_session(
                control_plane.pool(),
                archive.session_id,
            )
            .await?;
            let Some(session) = session else {
                let detail =
                    ErrorDetail::from([("session_id".to_string(), archive.session_id.to_string())]);
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

            redesmyn_storage::sessions::archive_chat_session(
                control_plane.pool(),
                archive.session_id,
            )
            .await?;

            Ok(ResponseResult::ArchiveChatSession(
                ArchiveChatSessionResponse {},
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

            if let Some(epic_id) = list.epic_id {
                let epic_exists = redesmyn_storage::sessions::epic_exists_in_repo(
                    control_plane.pool(),
                    workspace_id,
                    repo_id,
                    epic_id,
                )
                .await?;
                if !epic_exists {
                    let detail = ErrorDetail::from([("epic_id".to_string(), epic_id.to_string())]);
                    return Ok(ResponseResult::Error(
                        ErrorEnvelope::new(ErrorCategory::NotFound, "Epic not found.")
                            .with_detail(detail),
                    ));
                }
            }

            let sessions = redesmyn_storage::sessions::list_chat_sessions(
                control_plane.pool(),
                workspace_id,
                repo_id,
                list.epic_id,
                list.include_archived,
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
        redesmyn_protocol::client::RequestPayload::RegenerateChatSessionTitle(req) => {
            let (workspace_id, repo_id) = match require_repo_scope_ids(envelope) {
                Ok(ids) => ids,
                Err(err) => return Ok(ResponseResult::Error(err)),
            };

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

            let Some(api_key) = openai_api_key_from_env() else {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "OpenAI API key is not configured.",
                )));
            };

            let prompt = redesmyn_storage::sessions::latest_session_event_preview_by_kind(
                control_plane.pool(),
                req.session_id,
                "user_message",
            )
            .await?;
            let Some(prompt) = prompt
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty())
            else {
                return Ok(ResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Cannot regenerate title: this session has no user messages yet.",
                )));
            };
            let prompt: String = prompt.chars().take(CHAT_TITLE_SOURCE_MAX_CHARS).collect();

            let generated = match generate_chat_title_via_openai(&api_key, &prompt).await {
                Ok(Some(title)) => title,
                Ok(None) => {
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "OpenAI did not return a usable chat title.",
                    )));
                }
                Err(error) => {
                    tracing::warn!(
                        session_id = %req.session_id,
                        error = %error,
                        "failed to regenerate chat title"
                    );
                    return Ok(ResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "Failed to regenerate chat title.",
                    )));
                }
            };

            let updated = redesmyn_storage::sessions::set_session_title(
                control_plane.pool(),
                req.session_id,
                &generated,
            )
            .await?;
            if !updated {
                let detail =
                    ErrorDetail::from([("session_id".to_string(), req.session_id.to_string())]);
                return Ok(ResponseResult::Error(
                    ErrorEnvelope::new(ErrorCategory::NotFound, "Session not found.")
                        .with_detail(detail),
                ));
            }

            Ok(ResponseResult::RegenerateChatSessionTitle(
                RegenerateChatSessionTitleResponse { title: generated },
            ))
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

fn session_message_preview(text: &str, image_count: usize) -> String {
    const MAX_PREVIEW_CHARS: usize = 140;
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");

    if !normalized.is_empty() {
        return normalized.chars().take(MAX_PREVIEW_CHARS).collect();
    }

    match image_count {
        0 => String::new(),
        1 => "[image]".to_string(),
        count => format!("[{count} images]"),
    }
}

fn validate_image_attachments(
    image_attachments: &[redesmyn_protocol::session::ImageAttachment],
) -> Result<(), ErrorEnvelope> {
    if image_attachments.len() > SEND_SESSION_MESSAGE_MAX_IMAGE_ATTACHMENTS {
        return Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Too many image attachments.",
        )
        .with_detail(ErrorDetail::from([
            (
                "max_image_attachments".to_string(),
                SEND_SESSION_MESSAGE_MAX_IMAGE_ATTACHMENTS.to_string(),
            ),
            (
                "actual_image_attachments".to_string(),
                image_attachments.len().to_string(),
            ),
        ])));
    }

    let mut total_bytes = 0_u64;
    for (index, attachment) in image_attachments.iter().enumerate() {
        if attachment.artifact.kind != ArtifactKind::Image {
            return Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Image attachments must reference image artifacts.",
            )
            .with_detail(ErrorDetail::from([
                ("index".to_string(), index.to_string()),
                (
                    "artifact_kind".to_string(),
                    format!("{:?}", attachment.artifact.kind),
                ),
            ])));
        }

        let Some(byte_len) = attachment.artifact.byte_len else {
            return Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Image attachment byte length is required.",
            )
            .with_detail(ErrorDetail::from([(
                "index".to_string(),
                index.to_string(),
            )])));
        };

        if byte_len > SEND_SESSION_MESSAGE_MAX_IMAGE_ATTACHMENT_BYTES {
            return Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Image attachment is too large.",
            )
            .with_detail(ErrorDetail::from([
                ("index".to_string(), index.to_string()),
                (
                    "max_bytes".to_string(),
                    SEND_SESSION_MESSAGE_MAX_IMAGE_ATTACHMENT_BYTES.to_string(),
                ),
                ("actual_bytes".to_string(), byte_len.to_string()),
            ])));
        }

        total_bytes = total_bytes.saturating_add(byte_len);
        if total_bytes > SEND_SESSION_MESSAGE_MAX_IMAGE_TOTAL_BYTES {
            return Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Total image attachment payload is too large.",
            )
            .with_detail(ErrorDetail::from([
                (
                    "max_total_bytes".to_string(),
                    SEND_SESSION_MESSAGE_MAX_IMAGE_TOTAL_BYTES.to_string(),
                ),
                ("actual_total_bytes".to_string(), total_bytes.to_string()),
            ])));
        }

        if let Some(mime) = attachment.artifact.mime.as_deref() {
            if !mime.starts_with("image/") {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Image attachment mime type is invalid.",
                )
                .with_detail(ErrorDetail::from([
                    ("index".to_string(), index.to_string()),
                    ("mime".to_string(), mime.to_string()),
                ])));
            }
        }
    }

    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct OpenAiChatTitleRequest {
    model: &'static str,
    temperature: f32,
    max_tokens: u16,
    messages: Vec<OpenAiChatTitleMessage>,
}

#[derive(Debug, serde::Serialize)]
struct OpenAiChatTitleMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChatTitleResponse {
    #[serde(default)]
    choices: Vec<OpenAiChatTitleChoice>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChatTitleChoice {
    message: OpenAiChatTitleAssistantMessage,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiChatTitleAssistantMessage {
    content: Option<String>,
}

fn maybe_spawn_chat_title_generation(
    pool: sqlx::SqlitePool,
    session: &AgentSessionRecord,
    text: &str,
    has_text: bool,
) {
    if session.scope_kind != StorageAgentSessionScopeKind::Chat || !has_text {
        return;
    }

    if session
        .title
        .as_deref()
        .is_some_and(|title| !title.trim().is_empty())
    {
        return;
    }

    let Some(api_key) = openai_api_key_from_env() else {
        return;
    };

    let prompt = text.trim();
    if prompt.is_empty() {
        return;
    }

    let prompt: String = prompt.chars().take(CHAT_TITLE_SOURCE_MAX_CHARS).collect();
    let session_id = session.session_id;

    tokio::spawn(async move {
        let generated = match generate_chat_title_via_openai(&api_key, &prompt).await {
            Ok(title) => title,
            Err(error) => {
                tracing::warn!(
                    session_id = %session_id,
                    error = %error,
                    "failed to generate chat title"
                );
                return;
            }
        };

        let Some(title) = generated else {
            return;
        };

        match redesmyn_storage::sessions::set_session_title_if_missing(&pool, session_id, &title)
            .await
        {
            Ok(false) => {}
            Ok(true) => {
                tracing::debug!(session_id = %session_id, "generated chat title");
            }
            Err(error) => {
                tracing::warn!(
                    session_id = %session_id,
                    error = %error,
                    "failed to persist generated chat title"
                );
            }
        }
    });
}

fn openai_api_key_from_env() -> Option<String> {
    fn non_empty_env(key: &str) -> Option<String> {
        std::env::var(key)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    let override_value = openai_api_key_override_store()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();

    override_value
        .or_else(|| non_empty_env(REDESMYN_OPENAI_API_KEY_ENV))
        .or_else(|| non_empty_env(OPENAI_API_KEY_ENV))
}

async fn generate_chat_title_via_openai(
    api_key: &str,
    prompt: &str,
) -> Result<Option<String>, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(|error| format!("failed to initialize HTTP client: {error}"))?;

    let request = OpenAiChatTitleRequest {
        model: OPENAI_CHAT_TITLE_MODEL,
        temperature: 0.2,
        max_tokens: 20,
        messages: vec![
            OpenAiChatTitleMessage {
                role: "system",
                content: "Generate a concise chat session title (max 8 words). Return only the title, with no quotes or prefixes.".to_string(),
            },
            OpenAiChatTitleMessage {
                role: "user",
                content: prompt.to_string(),
            },
        ],
    };

    let response = client
        .post(OPENAI_CHAT_TITLE_ENDPOINT)
        .bearer_auth(api_key)
        .json(&request)
        .send()
        .await
        .map_err(|error| format!("request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let body_preview: String = body.chars().take(240).collect();
        return Err(format!("OpenAI returned {status}: {body_preview}"));
    }

    let payload: OpenAiChatTitleResponse = response
        .json()
        .await
        .map_err(|error| format!("invalid response payload: {error}"))?;

    Ok(payload
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .and_then(|content| normalize_chat_title_candidate(&content)))
}

fn normalize_chat_title_candidate(raw: &str) -> Option<String> {
    let mut title = raw
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .trim()
        .trim_matches(|ch| matches!(ch, '"' | '\'' | '`'))
        .to_string();

    let lower = title.to_ascii_lowercase();
    for prefix in ["title:", "session title:", "chat title:"] {
        if lower.starts_with(prefix) {
            title = title[prefix.len()..].trim().to_string();
            break;
        }
    }

    title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    title = title
        .trim_end_matches(|ch: char| ch.is_ascii_punctuation())
        .to_string();

    if title.is_empty() {
        return None;
    }

    if title.chars().count() > CHAT_TITLE_MAX_CHARS {
        title = title.chars().take(CHAT_TITLE_MAX_CHARS).collect();
        title = title.trim().to_string();
    }

    if title.is_empty() { None } else { Some(title) }
}

fn agent_session_summary_from_record(record: AgentSessionRecord) -> AgentSessionSummary {
    let scope_kind = match record.scope_kind {
        StorageAgentSessionScopeKind::Task => AgentSessionScopeKind::Task,
        StorageAgentSessionScopeKind::Chat => AgentSessionScopeKind::Chat,
    };

    let agent_kind = protocol_agent_kind_from_storage(record.agent_kind);

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
        epic_id: record.epic_id,
        agent_kind,
        status,
        title: record.title,
        archived_at: record.archived_at_ms.map(timestamp_from_ms),
        repo_name: record.repo_name,
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

fn is_structured_agent_kind(agent_kind: AgentKind) -> bool {
    matches!(agent_kind, AgentKind::Codex | AgentKind::ClaudeCode)
}

fn parse_external_session_ref(raw: &str) -> Option<ExternalSessionRef> {
    serde_json::from_str(raw).ok()
}

fn external_ref_matches_agent_kind(
    external_session_ref: &ExternalSessionRef,
    agent_kind: AgentKind,
) -> bool {
    matches!(
        (agent_kind, external_session_ref),
        (AgentKind::Codex, ExternalSessionRef::CodexThread { .. })
            | (AgentKind::Codex, ExternalSessionRef::CodexSession { .. })
            | (
                AgentKind::ClaudeCode,
                ExternalSessionRef::ClaudeSession { .. }
            )
    )
}

fn encode_agent_command_payload<T: serde::Serialize>(
    payload: &T,
) -> Result<Vec<u8>, ErrorEnvelope> {
    serde_json::to_vec(payload).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to encode agent command payload.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })
}

fn decode_list_session_models_from_command_detail(
    detail_bytes: &[u8],
) -> Result<ListSessionModelsResponse, ErrorEnvelope> {
    #[derive(serde::Deserialize)]
    struct UpdateDetailEnvelope {
        detail: Option<ErrorDetail>,
        error: Option<ErrorEnvelope>,
    }

    let detail_envelope: UpdateDetailEnvelope =
        serde_json::from_slice(detail_bytes).map_err(|err| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                "Failed to decode model-list command detail envelope.",
            )
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;

    if let Some(error) = detail_envelope.error {
        return Err(error);
    }

    let models_json = detail_envelope
        .detail
        .and_then(|mut detail| detail.remove("models_json"))
        .ok_or_else(|| {
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Model list response did not include models_json payload.",
            )
        })?;

    serde_json::from_str::<ListSessionModelsResponse>(&models_json).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to decode model list payload.",
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
        repo_slug: Some(graph.epic.repo_slug),
        repo_title: Some(graph.epic.repo_title),
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
