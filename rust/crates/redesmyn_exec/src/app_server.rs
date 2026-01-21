use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use redesmyn_domain::agent::AppServerTurnIntent;
use redesmyn_ids::{SessionEventId, SessionId, TaskId};
use redesmyn_logging::tracing;
use redesmyn_protocol::artifacts::{ArtifactKind, ArtifactRef};
use redesmyn_protocol::daemon::{DaemonFrame, DaemonMessage, SessionEventBatch};
use redesmyn_protocol::session::{
    ArtifactEmitted, AssistantMessage, InterfaceMode, SessionEnded, SessionEvent, SessionEventKind,
    SessionScope, SessionStarted, StatusUpdate, ToolInvocation, ToolResult, TurnCompleted,
    TurnStarted, TurnState, UserMessage,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, ProtocolEnvelope, Timestamp};
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::artifact_store::{ArtifactStoreError, LocalArtifactStore};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct AppServerSupervisorConfig {
    pub max_message_chars: usize,
    pub max_preview_chars: usize,
    pub max_tool_io_chars: usize,
    pub max_tool_preview_chars: usize,
}

impl Default for AppServerSupervisorConfig {
    fn default() -> Self {
        Self {
            max_message_chars: 4_000,
            max_preview_chars: 240,
            max_tool_io_chars: 4_000,
            max_tool_preview_chars: 240,
        }
    }
}

#[derive(Clone)]
pub struct AppServerSessionSpec {
    pub scope: SessionScope,
    pub allow_concurrent_for_task: bool,
    pub process: Arc<dyn AppServerProcess>,
}

impl std::fmt::Debug for AppServerSessionSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppServerSessionSpec")
            .field("scope", &self.scope)
            .field("allow_concurrent_for_task", &self.allow_concurrent_for_task)
            .field("process", &"<dyn AppServerProcess>")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppServerRequest {
    SendMessage { intent: AppServerTurnIntent },
    Interrupt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppServerResponse {
    MessageAccepted,
    Interrupted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppServerEvent {
    UserMessage {
        text: String,
    },
    AssistantMessage {
        text: String,
    },
    ToolInvocation {
        tool_name: String,
        tool_call_id: Option<String>,
        input: String,
    },
    ToolResult {
        tool_name: String,
        tool_call_id: Option<String>,
        output: String,
        error: Option<ErrorEnvelope>,
    },
    ArtifactBytes {
        kind: ArtifactKind,
        mime: Option<String>,
        label: Option<String>,
        bytes: Vec<u8>,
    },
    ArtifactRef {
        artifact: ArtifactRef,
        label: Option<String>,
    },
}

pub struct AppServerConnection {
    pub client: Arc<dyn AppServerClient>,
    pub events: mpsc::Receiver<AppServerEvent>,
}

pub trait AppServerClient: Send + Sync + 'static {
    fn request(
        &self,
        request: AppServerRequest,
    ) -> BoxFuture<'_, Result<AppServerResponse, AppServerRequestError>>;
}

pub trait AppServerProcess: Send + Sync + 'static {
    fn start(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>>;
    fn connect(&self) -> BoxFuture<'_, Result<AppServerConnection, AppServerProcessError>>;
    fn shutdown(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>>;
}

#[derive(Debug, thiserror::Error)]
pub enum AppServerProcessError {
    #[error("start failed: {reason}")]
    StartFailed { reason: String },
    #[error("connect failed: {reason}")]
    ConnectFailed { reason: String },
    #[error("shutdown failed: {reason}")]
    ShutdownFailed { reason: String },
}

#[derive(Debug, thiserror::Error)]
pub enum AppServerRequestError {
    #[error("request failed: {reason}")]
    Failed { reason: String },
}

#[derive(Debug, thiserror::Error)]
pub enum StartSessionError {
    #[error("invalid app-server session scope: {reason}")]
    InvalidScope { reason: String },
    #[error("task already has an active {interface_mode:?} session: {session_id}")]
    TaskHasActiveSession {
        session_id: SessionId,
        interface_mode: InterfaceMode,
    },
    #[error("supervisor is shutting down")]
    ShuttingDown,
    #[error(transparent)]
    Artifacts(#[from] ArtifactStoreError),
    #[error(transparent)]
    Process(#[from] AppServerProcessError),
}

#[derive(Debug, thiserror::Error)]
pub enum SessionControlError {
    #[error("unknown session: {session_id}")]
    UnknownSession { session_id: SessionId },
    #[error("session control channel closed: {session_id}")]
    SessionClosed { session_id: SessionId },
}

#[derive(Debug, thiserror::Error)]
pub enum AppServerCallError {
    #[error(transparent)]
    Control(#[from] SessionControlError),
    #[error(transparent)]
    Request(#[from] AppServerRequestError),
    #[error(transparent)]
    Process(#[from] AppServerProcessError),
}

#[derive(Debug)]
pub struct AppServerSupervisor {
    config: AppServerSupervisorConfig,
    artifact_store: LocalArtifactStore,
    frames_tx: mpsc::Sender<DaemonFrame>,
    state: Arc<Mutex<SupervisorState>>,
    cleanup_tx: mpsc::Sender<SessionId>,
}

#[derive(Debug, Default)]
struct SupervisorState {
    is_shutting_down: bool,
    sessions: HashMap<SessionId, SessionEntry>,
    active_by_task: HashMap<(TaskId, InterfaceMode), SessionId>,
}

#[derive(Debug)]
struct SessionEntry {
    task_id: TaskId,
    interface_mode: InterfaceMode,
    control_tx: mpsc::Sender<SessionCommand>,
    join: tokio::task::JoinHandle<()>,
}

#[derive(Debug)]
enum SessionCommand {
    SendMessage {
        intent: AppServerTurnIntent,
        reply: oneshot::Sender<Result<AppServerResponse, AppServerRequestError>>,
    },
    Interrupt {
        reply: oneshot::Sender<Result<AppServerResponse, AppServerRequestError>>,
    },
    Reconnect {
        reply: oneshot::Sender<Result<(), AppServerProcessError>>,
    },
    Stop,
}

impl AppServerSupervisor {
    pub async fn new(
        config: AppServerSupervisorConfig,
        artifact_store: LocalArtifactStore,
        frames_tx: mpsc::Sender<DaemonFrame>,
    ) -> Result<Self, StartSessionError> {
        artifact_store.ensure_dirs().await?;
        let (cleanup_tx, mut cleanup_rx) = mpsc::channel::<SessionId>(256);

        let state = Arc::new(Mutex::new(SupervisorState::default()));
        let state_for_cleanup = Arc::clone(&state);
        tokio::spawn(async move {
            while let Some(session_id) = cleanup_rx.recv().await {
                let mut st = state_for_cleanup.lock().await;
                if let Some(entry) = st.sessions.remove(&session_id) {
                    st.active_by_task
                        .remove(&(entry.task_id, entry.interface_mode));
                }
            }
        });

        Ok(Self {
            config,
            artifact_store,
            frames_tx,
            state,
            cleanup_tx,
        })
    }

    pub async fn start_session(
        &self,
        task_id: TaskId,
        spec: AppServerSessionSpec,
    ) -> Result<SessionId, StartSessionError> {
        validate_task_scope(task_id, spec.scope)?;
        let interface_mode = InterfaceMode::Structured;
        let session_id = SessionId::new();

        let (control_tx, control_rx) = mpsc::channel::<SessionCommand>(16);
        let (ready_tx, ready_rx) = oneshot::channel::<Result<(), AppServerProcessError>>();

        let config = self.config.clone();
        let artifact_store = self.artifact_store.clone();
        let frames_tx = self.frames_tx.clone();
        let cleanup_tx = self.cleanup_tx.clone();
        let process = Arc::clone(&spec.process);
        let scope = spec.scope;

        {
            let mut st = self.state.lock().await;
            if st.is_shutting_down {
                return Err(StartSessionError::ShuttingDown);
            }

            if !spec.allow_concurrent_for_task {
                if let Some(existing) = st.active_by_task.get(&(task_id, interface_mode)) {
                    return Err(StartSessionError::TaskHasActiveSession {
                        session_id: *existing,
                        interface_mode,
                    });
                }
            }

            let join = tokio::spawn(async move {
                let span = redesmyn_logging::redesmyn_info_span!(
                    "app_server.session",
                    session_id = %session_id,
                );
                redesmyn_logging::span::record_task_id(&span, task_id);
                let _guard = span.enter();

                if let Err(err) = run_session(
                    config,
                    artifact_store,
                    frames_tx,
                    task_id,
                    session_id,
                    scope,
                    process,
                    control_rx,
                    ready_tx,
                )
                .await
                {
                    tracing::error!(error = %err, "app-server session crashed");
                }

                let _ = cleanup_tx.send(session_id).await;
            });

            st.sessions.insert(
                session_id,
                SessionEntry {
                    task_id,
                    interface_mode,
                    control_tx: control_tx.clone(),
                    join,
                },
            );
            st.active_by_task
                .insert((task_id, interface_mode), session_id);
        }

        match ready_rx.await {
            Ok(Ok(())) => Ok(session_id),
            Ok(Err(err)) => {
                self.cleanup_local(session_id).await;
                Err(err.into())
            }
            Err(_) => {
                self.cleanup_local(session_id).await;
                Err(StartSessionError::Process(
                    AppServerProcessError::StartFailed {
                        reason: "session task failed to signal readiness".to_owned(),
                    },
                ))
            }
        }
    }

    pub async fn send_message(
        &self,
        session_id: SessionId,
        intent: AppServerTurnIntent,
    ) -> Result<AppServerResponse, AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<AppServerResponse, AppServerRequestError>>();
        self.send_command(session_id, SessionCommand::SendMessage { intent, reply })
            .await?;
        rx.await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)
    }

    pub async fn interrupt_session(
        &self,
        session_id: SessionId,
    ) -> Result<AppServerResponse, AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<AppServerResponse, AppServerRequestError>>();
        self.send_command(session_id, SessionCommand::Interrupt { reply })
            .await?;
        rx.await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)
    }

    pub async fn reconnect_session(&self, session_id: SessionId) -> Result<(), AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<(), AppServerProcessError>>();
        self.send_command(session_id, SessionCommand::Reconnect { reply })
            .await?;
        match rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(AppServerCallError::Process(err)),
            Err(_) => Err(AppServerCallError::Control(
                SessionControlError::SessionClosed { session_id },
            )),
        }
    }

    pub async fn stop_session(&self, session_id: SessionId) -> Result<(), SessionControlError> {
        self.send_command(session_id, SessionCommand::Stop).await
    }

    pub async fn shutdown(&self) {
        let sessions = {
            let mut st = self.state.lock().await;
            st.is_shutting_down = true;
            st.active_by_task.clear();
            std::mem::take(&mut st.sessions)
        };

        for (session_id, entry) in &sessions {
            let _ = entry.control_tx.send(SessionCommand::Stop).await;
            tracing::info!(%session_id, "shutdown requested stop");
        }

        for (_session_id, entry) in sessions {
            let _ = entry.join.await;
        }
    }

    async fn cleanup_local(&self, session_id: SessionId) {
        let mut st = self.state.lock().await;
        if let Some(entry) = st.sessions.remove(&session_id) {
            st.active_by_task
                .remove(&(entry.task_id, entry.interface_mode));
        }
    }

    async fn send_command(
        &self,
        session_id: SessionId,
        cmd: SessionCommand,
    ) -> Result<(), SessionControlError> {
        let tx = {
            let st = self.state.lock().await;
            let entry = st
                .sessions
                .get(&session_id)
                .ok_or(SessionControlError::UnknownSession { session_id })?;
            entry.control_tx.clone()
        };

        tx.send(cmd)
            .await
            .map_err(|_| SessionControlError::SessionClosed { session_id })
    }
}

fn validate_task_scope(task_id: TaskId, scope: SessionScope) -> Result<(), StartSessionError> {
    match scope {
        SessionScope::Task {
            task_id: scope_task_id,
        } if scope_task_id == task_id => Ok(()),
        SessionScope::Task {
            task_id: scope_task_id,
        } => Err(StartSessionError::InvalidScope {
            reason: format!("scope.task_id must match task_id ({task_id}); got {scope_task_id}"),
        }),
        _ => Err(StartSessionError::InvalidScope {
            reason: "expected task scope".to_owned(),
        }),
    }
}

#[derive(Debug, thiserror::Error)]
enum RunSessionError {
    #[error(transparent)]
    Artifacts(#[from] ArtifactStoreError),
    #[error(transparent)]
    Emit(#[from] EmitEventError),
    #[error(transparent)]
    Process(#[from] AppServerProcessError),
    #[error(transparent)]
    Request(#[from] AppServerRequestError),
}

async fn run_session(
    config: AppServerSupervisorConfig,
    artifact_store: LocalArtifactStore,
    frames_tx: mpsc::Sender<DaemonFrame>,
    task_id: TaskId,
    session_id: SessionId,
    scope: SessionScope,
    process: Arc<dyn AppServerProcess>,
    mut control_rx: mpsc::Receiver<SessionCommand>,
    ready_tx: oneshot::Sender<Result<(), AppServerProcessError>>,
) -> Result<(), RunSessionError> {
    if let Err(err) = process.start().await {
        tracing::error!(error = %err, "app-server start failed");
        let _ = ready_tx.send(Err(err));
        return Ok(());
    }

    let AppServerConnection { mut client, events } = match process.connect().await {
        Ok(conn) => conn,
        Err(err) => {
            tracing::error!(error = %err, "app-server connect failed");
            let _ = ready_tx.send(Err(err));
            return Ok(());
        }
    };

    emit_event(
        &frames_tx,
        session_id,
        scope,
        SessionEventKind::SessionStarted(SessionStarted {}),
    )
    .await?;

    let mut event_forwarder = tokio::spawn(run_event_forwarder(
        frames_tx.clone(),
        artifact_store.clone(),
        config.clone(),
        session_id,
        scope,
        events,
    ));

    let _ = ready_tx.send(Ok(()));
    tracing::info!(%task_id, %session_id, "app-server session started");

    let interface_mode = InterfaceMode::Structured;
    while let Some(cmd) = control_rx.recv().await {
        match cmd {
            SessionCommand::SendMessage { intent, reply } => {
                emit_event(
                    &frames_tx,
                    session_id,
                    scope,
                    SessionEventKind::TurnStarted(TurnStarted {
                        interface_mode,
                        external_session_ref: None,
                        idempotency_key: None,
                        log_offset_bytes: None,
                    }),
                )
                .await?;

                let response = client
                    .request(AppServerRequest::SendMessage { intent })
                    .await;
                match &response {
                    Ok(_) => {
                        emit_event(
                            &frames_tx,
                            session_id,
                            scope,
                            SessionEventKind::TurnCompleted(TurnCompleted {
                                interface_mode,
                                external_session_ref: None,
                                exit_code: None,
                                error: None,
                            }),
                        )
                        .await?;
                    }
                    Err(err) => {
                        emit_event(
                            &frames_tx,
                            session_id,
                            scope,
                            SessionEventKind::TurnCompleted(TurnCompleted {
                                interface_mode,
                                external_session_ref: None,
                                exit_code: None,
                                error: Some(ErrorEnvelope::new(
                                    ErrorCategory::Unavailable,
                                    format!("app-server request failed: {err}"),
                                )),
                            }),
                        )
                        .await?;
                    }
                }

                let _ = reply.send(response);
            }
            SessionCommand::Interrupt { reply } => {
                let response = client.request(AppServerRequest::Interrupt).await;
                let _ = reply.send(response);
            }
            SessionCommand::Reconnect { reply } => {
                event_forwarder.abort();
                match process.connect().await {
                    Ok(AppServerConnection {
                        client: new_client,
                        events: new_events,
                    }) => {
                        client = new_client;
                        event_forwarder = tokio::spawn(run_event_forwarder(
                            frames_tx.clone(),
                            artifact_store.clone(),
                            config.clone(),
                            session_id,
                            scope,
                            new_events,
                        ));
                        let _ = emit_event(
                            &frames_tx,
                            session_id,
                            scope,
                            SessionEventKind::StatusUpdate(StatusUpdate {
                                turn_state: TurnState::Running,
                                blocking: None,
                                progress_percent: None,
                                message: Some("app_server_reconnected".to_owned()),
                            }),
                        )
                        .await;
                        let _ = reply.send(Ok(()));
                    }
                    Err(err) => {
                        let _ = reply.send(Err(err));
                    }
                }
            }
            SessionCommand::Stop => break,
        }
    }

    event_forwarder.abort();

    if let Err(err) = process.shutdown().await {
        tracing::warn!(error = %err, "app-server shutdown failed");
    }

    emit_event(
        &frames_tx,
        session_id,
        scope,
        SessionEventKind::SessionEnded(SessionEnded {}),
    )
    .await?;

    Ok(())
}

async fn run_event_forwarder(
    frames_tx: mpsc::Sender<DaemonFrame>,
    artifact_store: LocalArtifactStore,
    config: AppServerSupervisorConfig,
    session_id: SessionId,
    scope: SessionScope,
    mut rx: mpsc::Receiver<AppServerEvent>,
) -> Result<(), RunSessionError> {
    while let Some(event) = rx.recv().await {
        emit_app_server_event(
            &frames_tx,
            &artifact_store,
            &config,
            session_id,
            scope,
            event,
        )
        .await?;
    }
    Ok(())
}

async fn emit_app_server_event(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    artifact_store: &LocalArtifactStore,
    config: &AppServerSupervisorConfig,
    session_id: SessionId,
    scope: SessionScope,
    event: AppServerEvent,
) -> Result<(), RunSessionError> {
    let kind = match event {
        AppServerEvent::UserMessage { text } => {
            let mut ev = UserMessage {
                text,
                preview: String::new(),
                full_text_artifact: None,
            };
            limit_message_event(
                artifact_store,
                config.max_message_chars,
                config.max_preview_chars,
                &mut ev.text,
                &mut ev.preview,
                &mut ev.full_text_artifact,
            )
            .await?;
            SessionEventKind::UserMessage(ev)
        }
        AppServerEvent::AssistantMessage { text } => {
            let mut ev = AssistantMessage {
                text,
                preview: String::new(),
                full_text_artifact: None,
            };
            limit_message_event(
                artifact_store,
                config.max_message_chars,
                config.max_preview_chars,
                &mut ev.text,
                &mut ev.preview,
                &mut ev.full_text_artifact,
            )
            .await?;
            SessionEventKind::AssistantMessage(ev)
        }
        AppServerEvent::ToolInvocation {
            tool_name,
            tool_call_id,
            input,
        } => {
            let (input_preview, input_artifact) = limit_tool_text(
                artifact_store,
                config.max_tool_io_chars,
                config.max_tool_preview_chars,
                &input,
            )
            .await?;
            SessionEventKind::ToolInvocation(ToolInvocation {
                tool_name,
                tool_call_id,
                input_preview,
                input_artifact,
            })
        }
        AppServerEvent::ToolResult {
            tool_name,
            tool_call_id,
            output,
            error,
        } => {
            let (output_preview, output_artifact) = limit_tool_text(
                artifact_store,
                config.max_tool_io_chars,
                config.max_tool_preview_chars,
                &output,
            )
            .await?;
            SessionEventKind::ToolResult(ToolResult {
                tool_name,
                tool_call_id,
                output_preview,
                output_artifact,
                error,
            })
        }
        AppServerEvent::ArtifactBytes {
            kind,
            mime,
            label,
            bytes,
        } => {
            let artifact = artifact_store.store_bytes(kind, mime, &bytes).await?;
            SessionEventKind::ArtifactEmitted(ArtifactEmitted { artifact, label })
        }
        AppServerEvent::ArtifactRef { artifact, label } => {
            SessionEventKind::ArtifactEmitted(ArtifactEmitted { artifact, label })
        }
    };

    emit_event(frames_tx, session_id, scope, kind).await?;
    Ok(())
}

async fn limit_message_event(
    artifact_store: &LocalArtifactStore,
    max_message_chars: usize,
    max_preview_chars: usize,
    text: &mut String,
    preview: &mut String,
    full_text_artifact: &mut Option<ArtifactRef>,
) -> Result<(), RunSessionError> {
    let normalized_preview = normalize_preview(text);
    *preview = truncate_chars(&normalized_preview, max_preview_chars);

    if text.chars().count() <= max_message_chars {
        return Ok(());
    }

    let artifact = artifact_store
        .store_bytes(
            ArtifactKind::Log,
            Some("text/plain".to_owned()),
            text.as_bytes(),
        )
        .await?;
    *full_text_artifact = Some(artifact);
    *text = truncate_chars(text, max_message_chars);
    Ok(())
}

async fn limit_tool_text(
    artifact_store: &LocalArtifactStore,
    max_chars: usize,
    max_preview_chars: usize,
    text: &str,
) -> Result<(String, Option<ArtifactRef>), RunSessionError> {
    let normalized_preview = normalize_preview(text);
    let preview = truncate_chars(&normalized_preview, max_preview_chars);

    if text.chars().count() <= max_chars {
        return Ok((preview, None));
    }

    let artifact = artifact_store
        .store_bytes(
            ArtifactKind::Log,
            Some("text/plain".to_owned()),
            text.as_bytes(),
        )
        .await?;
    Ok((preview, Some(artifact)))
}

fn normalize_preview(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let mut out = String::new();
    for (idx, ch) in text.chars().enumerate() {
        if idx >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

#[derive(Debug, thiserror::Error)]
enum EmitEventError {
    #[error("control plane session event stream closed")]
    StreamClosed,
}

async fn emit_event(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    session_id: SessionId,
    scope: SessionScope,
    kind: SessionEventKind,
) -> Result<(), EmitEventError> {
    let record = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope,
        session_id,
        turn_id: None,
        kind,
    };
    let batch = SessionEventBatch {
        events: vec![record],
    };
    let frame = DaemonFrame::new(
        ProtocolEnvelope::new(),
        DaemonMessage::SessionEventBatch(batch),
    );

    frames_tx
        .send(frame)
        .await
        .map_err(|_| EmitEventError::StreamClosed)
}
