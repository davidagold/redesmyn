use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use redesmyn_domain::agent::AppServerTurnIntent;
use redesmyn_ids::{SessionEventId, SessionId, TaskId};
use redesmyn_logging::tracing;
use redesmyn_protocol::agent_commands::SessionPolicySnapshot;
use redesmyn_protocol::artifacts::{ArtifactKind, ArtifactRef};
use redesmyn_protocol::client::{ModelReasoningEffort, SessionModelOption, SessionModelSelection};
use redesmyn_protocol::daemon::{
    DaemonFrame, DaemonMessage, SessionEventBatch, SessionLiveEventBatch,
};
use redesmyn_protocol::session::{
    ArtifactEmitted, AssistantMessage, AssistantReasoning, AssistantReasoningText,
    CodexApprovalPolicy, CodexApprovalPolicyChanged, CodexSandboxPolicy, CodexSandboxPolicyChanged,
    ExternalSessionRef, InterfaceMode, PermissionDecided, PermissionDecision, PermissionDecisionBy,
    PermissionRequest, PermissionRequested, PermissionsMode, PermissionsModeChanged, SessionEnded,
    SessionEvent, SessionEventKind, SessionModelChanged, SessionModelReasoningEffort, SessionScope,
    SessionStarted, StatusUpdate, ToolInvocation, ToolResult, TurnCompleted, TurnStarted,
    TurnState,
};
use redesmyn_protocol::session_live::{
    AssistantMessageDelta, AssistantReasoningRawDelta, AssistantReasoningSummaryDelta,
    AssistantReasoningSummaryPartAdded, SessionLiveEvent, SessionLiveEventKind, ToolOutputDelta,
};
use redesmyn_protocol::{ErrorEnvelope, ProtocolEnvelope, Timestamp};
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::active_sessions::ActiveSessionsByTask;
use crate::artifact_store::{ArtifactStoreError, LocalArtifactStore};
use crate::text_limits::{normalize_preview, truncate_chars};

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
    SendMessage {
        intent: AppServerTurnIntent,
    },
    Interrupt,
    ListModels,
    SetModel {
        selection: SessionModelSelection,
    },
    SetPermissionsMode {
        mode: PermissionsMode,
    },
    SetCodexApprovalPolicy {
        approval_policy: Option<CodexApprovalPolicy>,
    },
    SetCodexSandboxPolicy {
        sandbox_policy: Option<CodexSandboxPolicy>,
    },
    RespondPermissionRequest {
        request_id: String,
        decision: PermissionDecision,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppServerResponse {
    MessageAccepted,
    Interrupted,
    ModelsListed {
        options: Vec<SessionModelOption>,
        selection: SessionModelSelection,
    },
    ModelSet,
    PermissionsModeSet,
    CodexApprovalPolicySet,
    CodexSandboxPolicySet,
    PermissionRequestResponded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppServerEvent {
    TurnStarted {
        turn_id: Option<String>,
        external_session_ref: Option<ExternalSessionRef>,
    },
    TurnCompleted {
        turn_id: Option<String>,
        external_session_ref: Option<ExternalSessionRef>,
        error: Option<ErrorEnvelope>,
    },
    UserMessage {
        text: String,
    },
    AssistantMessage {
        text: String,
    },
    AssistantMessageDelta {
        turn_id: Option<String>,
        item_id: Option<String>,
        delta: String,
    },
    AssistantReasoningSummaryPartAdded {
        turn_id: Option<String>,
        item_id: Option<String>,
        summary_index: i64,
    },
    AssistantReasoningSummaryDelta {
        turn_id: Option<String>,
        item_id: Option<String>,
        summary_index: i64,
        delta: String,
    },
    AssistantReasoningRawDelta {
        turn_id: Option<String>,
        item_id: Option<String>,
        content_index: i64,
        delta: String,
    },
    AssistantReasoning {
        turn_id: Option<String>,
        item_id: Option<String>,
        summary: String,
        raw: Option<String>,
        signature: Option<String>,
    },
    ToolOutputDelta {
        turn_id: Option<String>,
        tool_name: String,
        tool_call_id: Option<String>,
        delta: String,
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
    PermissionRequested {
        request_id: String,
        summary: String,
        request: PermissionRequest,
    },
    PermissionDecided {
        request_id: String,
        decision: PermissionDecision,
        decided_by: PermissionDecisionBy,
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
    active_by_task: ActiveSessionsByTask,
}

#[derive(Debug)]
struct SessionEntry {
    task_id: Option<TaskId>,
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
    ListModels {
        reply: oneshot::Sender<Result<AppServerResponse, AppServerRequestError>>,
    },
    SetModel {
        selection: SessionModelSelection,
        reply: oneshot::Sender<Result<AppServerResponse, AppServerRequestError>>,
    },
    SetPermissionsMode {
        mode: PermissionsMode,
        reply: oneshot::Sender<Result<AppServerResponse, AppServerRequestError>>,
    },
    SetCodexApprovalPolicy {
        approval_policy: Option<CodexApprovalPolicy>,
        reply: oneshot::Sender<Result<AppServerResponse, AppServerRequestError>>,
    },
    SetCodexSandboxPolicy {
        sandbox_policy: Option<CodexSandboxPolicy>,
        reply: oneshot::Sender<Result<AppServerResponse, AppServerRequestError>>,
    },
    RespondPermissionRequest {
        request_id: String,
        decision: PermissionDecision,
        reply: oneshot::Sender<Result<AppServerResponse, AppServerRequestError>>,
    },
    HydratePolicies {
        snapshot: SessionPolicySnapshot,
        reply: oneshot::Sender<Result<(), AppServerRequestError>>,
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
                    if let Some(task_id) = entry.task_id {
                        st.active_by_task
                            .remove(task_id, entry.interface_mode, session_id);
                    }
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
        session_id: SessionId,
        task_id: Option<TaskId>,
        spec: AppServerSessionSpec,
    ) -> Result<SessionId, StartSessionError> {
        validate_scope(task_id, spec.scope)?;
        let interface_mode = InterfaceMode::Structured;

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

            if st.sessions.contains_key(&session_id) {
                return Ok(session_id);
            }

            if let Some(task_id) = task_id {
                if !spec.allow_concurrent_for_task {
                    if let Some(existing) = st.active_by_task.any_session(task_id, interface_mode) {
                        return Err(StartSessionError::TaskHasActiveSession {
                            session_id: existing,
                            interface_mode,
                        });
                    }
                }
            }

            let join = tokio::spawn(async move {
                let span = redesmyn_logging::redesmyn_info_span!(
                    "app_server.session",
                    session_id = %session_id,
                );
                if let Some(task_id) = task_id {
                    redesmyn_logging::span::record_task_id(&span, task_id);
                }
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
            if let Some(task_id) = task_id {
                st.active_by_task
                    .insert(task_id, interface_mode, session_id);
            }
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

    pub async fn list_models(
        &self,
        session_id: SessionId,
    ) -> Result<(Vec<SessionModelOption>, SessionModelSelection), AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<AppServerResponse, AppServerRequestError>>();
        self.send_command(session_id, SessionCommand::ListModels { reply })
            .await?;
        match rx
            .await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)?
        {
            AppServerResponse::ModelsListed { options, selection } => Ok((options, selection)),
            other => Err(AppServerCallError::Request(AppServerRequestError::Failed {
                reason: format!("unexpected list models response: {other:?}"),
            })),
        }
    }

    pub async fn set_model(
        &self,
        session_id: SessionId,
        selection: SessionModelSelection,
    ) -> Result<(), AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<AppServerResponse, AppServerRequestError>>();
        self.send_command(session_id, SessionCommand::SetModel { selection, reply })
            .await?;
        rx.await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)?;
        Ok(())
    }

    pub async fn set_permissions_mode(
        &self,
        session_id: SessionId,
        mode: PermissionsMode,
    ) -> Result<(), AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<AppServerResponse, AppServerRequestError>>();
        self.send_command(
            session_id,
            SessionCommand::SetPermissionsMode { mode, reply },
        )
        .await?;
        rx.await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)?;
        Ok(())
    }

    pub async fn set_codex_approval_policy(
        &self,
        session_id: SessionId,
        approval_policy: Option<CodexApprovalPolicy>,
    ) -> Result<(), AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<AppServerResponse, AppServerRequestError>>();
        self.send_command(
            session_id,
            SessionCommand::SetCodexApprovalPolicy {
                approval_policy,
                reply,
            },
        )
        .await?;
        rx.await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)?;
        Ok(())
    }

    pub async fn set_codex_sandbox_policy(
        &self,
        session_id: SessionId,
        sandbox_policy: Option<CodexSandboxPolicy>,
    ) -> Result<(), AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<AppServerResponse, AppServerRequestError>>();
        self.send_command(
            session_id,
            SessionCommand::SetCodexSandboxPolicy {
                sandbox_policy,
                reply,
            },
        )
        .await?;
        rx.await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)?;
        Ok(())
    }

    pub async fn respond_permission_request(
        &self,
        session_id: SessionId,
        request_id: String,
        decision: PermissionDecision,
    ) -> Result<(), AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<AppServerResponse, AppServerRequestError>>();
        self.send_command(
            session_id,
            SessionCommand::RespondPermissionRequest {
                request_id,
                decision,
                reply,
            },
        )
        .await?;
        rx.await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)?;
        Ok(())
    }

    pub async fn hydrate_policies(
        &self,
        session_id: SessionId,
        snapshot: SessionPolicySnapshot,
    ) -> Result<(), AppServerCallError> {
        let (reply, rx) = oneshot::channel::<Result<(), AppServerRequestError>>();
        self.send_command(
            session_id,
            SessionCommand::HydratePolicies { snapshot, reply },
        )
        .await?;
        rx.await
            .map_err(|_| SessionControlError::SessionClosed { session_id })?
            .map_err(AppServerCallError::Request)?;
        Ok(())
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
            if let Some(task_id) = entry.task_id {
                st.active_by_task
                    .remove(task_id, entry.interface_mode, session_id);
            }
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

fn validate_scope(task_id: Option<TaskId>, scope: SessionScope) -> Result<(), StartSessionError> {
    match scope {
        SessionScope::Task {
            task_id: scope_task_id,
        } => match task_id {
            Some(task_id) if scope_task_id == task_id => Ok(()),
            Some(task_id) => Err(StartSessionError::InvalidScope {
                reason: format!(
                    "scope.task_id must match task_id ({task_id}); got {scope_task_id}"
                ),
            }),
            None => Err(StartSessionError::InvalidScope {
                reason: "missing task_id for task-scoped app-server session".to_owned(),
            }),
        },
        SessionScope::Chat => {
            if task_id.is_some() {
                return Err(StartSessionError::InvalidScope {
                    reason: "chat-scoped app-server sessions must not include task_id".to_owned(),
                });
            }
            Ok(())
        }
        _ => Err(StartSessionError::InvalidScope {
            reason: "unsupported session scope for app-server session".to_owned(),
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
    task_id: Option<TaskId>,
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
            if let Err(shutdown_err) = process.shutdown().await {
                tracing::warn!(
                    error = %shutdown_err,
                    "app-server shutdown after connect failure failed"
                );
            }
            let _ = ready_tx.send(Err(err));
            return Ok(());
        }
    };

    emit_event(
        &frames_tx,
        session_id,
        scope,
        None,
        SessionEventKind::SessionStarted(SessionStarted {}),
    )
    .await?;

    let mut event_forwarder = spawn_event_forwarder(
        frames_tx.clone(),
        artifact_store.clone(),
        config.clone(),
        session_id,
        scope,
        events,
    );

    let _ = ready_tx.send(Ok(()));
    tracing::info!(task_id = ?task_id, %session_id, "app-server session started");

    while let Some(cmd) = control_rx.recv().await {
        match cmd {
            SessionCommand::SendMessage { intent, reply } => {
                let response = client
                    .request(AppServerRequest::SendMessage { intent })
                    .await;
                if let Err(err) = &response {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("app_server_request_failed: {err}"),
                    );
                }
                let _ = reply.send(response);
            }
            SessionCommand::Interrupt { reply } => {
                try_emit_status_update(
                    &frames_tx,
                    session_id,
                    scope,
                    "interrupt_requested".to_owned(),
                );
                let response = client.request(AppServerRequest::Interrupt).await;
                let _ = reply.send(response);
            }
            SessionCommand::ListModels { reply } => {
                let response = client.request(AppServerRequest::ListModels).await;
                if let Err(err) = &response {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("list_models_failed: {err}"),
                    );
                }
                let _ = reply.send(response);
            }
            SessionCommand::SetModel { selection, reply } => {
                let response = client
                    .request(AppServerRequest::SetModel {
                        selection: selection.clone(),
                    })
                    .await;
                if let Err(err) = &response {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("set_model_failed: {err}"),
                    );
                } else {
                    let _ = emit_event(
                        &frames_tx,
                        session_id,
                        scope,
                        None,
                        SessionEventKind::SessionModelChanged(SessionModelChanged {
                            model_id: selection.model_id.clone(),
                            reasoning_effort: selection
                                .reasoning_effort
                                .map(session_reasoning_effort_from_client),
                        }),
                    )
                    .await;
                    let model = selection.model_id.as_deref().unwrap_or("default");
                    let effort = match selection.reasoning_effort {
                        Some(ModelReasoningEffort::Minimal) => "minimal",
                        Some(ModelReasoningEffort::Low) => "low",
                        Some(ModelReasoningEffort::Medium) => "medium",
                        Some(ModelReasoningEffort::High) => "high",
                        Some(ModelReasoningEffort::Xhigh) => "xhigh",
                        Some(ModelReasoningEffort::Unknown) => "unknown",
                        None => "default",
                    };
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("model_configured:{model}:{effort}"),
                    );
                }
                let _ = reply.send(response);
            }
            SessionCommand::SetPermissionsMode { mode, reply } => {
                let response = client
                    .request(AppServerRequest::SetPermissionsMode { mode })
                    .await;

                if response.is_ok() {
                    emit_event(
                        &frames_tx,
                        session_id,
                        scope,
                        None,
                        SessionEventKind::PermissionsModeChanged(PermissionsModeChanged { mode }),
                    )
                    .await?;
                } else if let Err(err) = &response {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("set_permissions_mode_failed: {err}"),
                    );
                }

                let _ = reply.send(response);
            }
            SessionCommand::SetCodexApprovalPolicy {
                approval_policy,
                reply,
            } => {
                let response = client
                    .request(AppServerRequest::SetCodexApprovalPolicy { approval_policy })
                    .await;

                if response.is_ok() {
                    emit_event(
                        &frames_tx,
                        session_id,
                        scope,
                        None,
                        SessionEventKind::CodexApprovalPolicyChanged(CodexApprovalPolicyChanged {
                            approval_policy,
                        }),
                    )
                    .await?;
                } else if let Err(err) = &response {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("set_codex_approval_policy_failed: {err}"),
                    );
                }

                let _ = reply.send(response);
            }
            SessionCommand::SetCodexSandboxPolicy {
                sandbox_policy,
                reply,
            } => {
                let response = client
                    .request(AppServerRequest::SetCodexSandboxPolicy {
                        sandbox_policy: sandbox_policy.clone(),
                    })
                    .await;

                if response.is_ok() {
                    emit_event(
                        &frames_tx,
                        session_id,
                        scope,
                        None,
                        SessionEventKind::CodexSandboxPolicyChanged(CodexSandboxPolicyChanged {
                            sandbox_policy,
                        }),
                    )
                    .await?;
                } else if let Err(err) = &response {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("set_codex_sandbox_policy_failed: {err}"),
                    );
                }

                let _ = reply.send(response);
            }
            SessionCommand::RespondPermissionRequest {
                request_id,
                decision,
                reply,
            } => {
                let response = client
                    .request(AppServerRequest::RespondPermissionRequest {
                        request_id,
                        decision,
                    })
                    .await;
                if let Err(err) = &response {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("respond_permission_request_failed: {err}"),
                    );
                }
                let _ = reply.send(response);
            }
            SessionCommand::HydratePolicies { snapshot, reply } => {
                let response = async {
                    let _ = client
                        .request(AppServerRequest::SetPermissionsMode {
                            mode: snapshot.permissions_mode,
                        })
                        .await?;
                    let _ = client
                        .request(AppServerRequest::SetCodexApprovalPolicy {
                            approval_policy: snapshot.codex_approval_policy,
                        })
                        .await?;
                    let _ = client
                        .request(AppServerRequest::SetCodexSandboxPolicy {
                            sandbox_policy: snapshot.codex_sandbox_policy.clone(),
                        })
                        .await?;
                    let _ = client
                        .request(AppServerRequest::SetModel {
                            selection: SessionModelSelection {
                                model_id: snapshot.model_id.clone(),
                                reasoning_effort: snapshot.model_reasoning_effort,
                            },
                        })
                        .await?;
                    Ok(())
                }
                .await;

                if let Err(err) = &response {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("hydrate_policies_failed: {err}"),
                    );
                }

                let _ = reply.send(response);
            }
            SessionCommand::Reconnect { reply } => match process.connect().await {
                Ok(AppServerConnection {
                    client: new_client,
                    events: new_events,
                }) => {
                    event_forwarder.abort();
                    client = new_client;
                    event_forwarder = spawn_event_forwarder(
                        frames_tx.clone(),
                        artifact_store.clone(),
                        config.clone(),
                        session_id,
                        scope,
                        new_events,
                    );
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        "app_server_reconnected".to_owned(),
                    );
                    let _ = reply.send(Ok(()));
                }
                Err(err) => {
                    try_emit_status_update(
                        &frames_tx,
                        session_id,
                        scope,
                        format!("app_server_reconnect_failed: {err}"),
                    );
                    let _ = reply.send(Err(err));
                }
            },
            SessionCommand::Stop => {
                try_emit_status_update(&frames_tx, session_id, scope, "stop_requested".to_owned());
                break;
            }
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
        None,
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
    const LIVE_FLUSH_MAX_EVENTS: usize = 64;
    let mut pending_live = Vec::<SessionLiveEvent>::new();
    let mut flush_interval = tokio::time::interval(std::time::Duration::from_millis(50));
    flush_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            maybe_event = rx.recv() => {
                let Some(event) = maybe_event else { break };

                match event {
                    AppServerEvent::AssistantMessageDelta { turn_id, item_id, delta } => {
                        if delta.is_empty() {
                            continue;
                        }
                        pending_live.push(make_assistant_delta_live_event(session_id, turn_id, item_id, delta));
                        if pending_live.len() >= LIVE_FLUSH_MAX_EVENTS {
                            try_emit_session_live_event_batch(&frames_tx, std::mem::take(&mut pending_live));
                        }
                    }
                    AppServerEvent::AssistantReasoningSummaryPartAdded {
                        turn_id,
                        item_id,
                        summary_index,
                    } => {
                        pending_live.push(make_reasoning_summary_part_added_live_event(
                            session_id,
                            turn_id,
                            item_id,
                            summary_index,
                        ));
                        if pending_live.len() >= LIVE_FLUSH_MAX_EVENTS {
                            try_emit_session_live_event_batch(
                                &frames_tx,
                                std::mem::take(&mut pending_live),
                            );
                        }
                    }
                    AppServerEvent::AssistantReasoningSummaryDelta {
                        turn_id,
                        item_id,
                        summary_index,
                        delta,
                    } => {
                        if delta.is_empty() {
                            continue;
                        }
                        pending_live.push(make_reasoning_summary_delta_live_event(
                            session_id,
                            turn_id,
                            item_id,
                            summary_index,
                            delta,
                        ));
                        if pending_live.len() >= LIVE_FLUSH_MAX_EVENTS {
                            try_emit_session_live_event_batch(
                                &frames_tx,
                                std::mem::take(&mut pending_live),
                            );
                        }
                    }
                    AppServerEvent::AssistantReasoningRawDelta {
                        turn_id,
                        item_id,
                        content_index,
                        delta,
                    } => {
                        if delta.is_empty() {
                            continue;
                        }
                        pending_live.push(make_reasoning_raw_delta_live_event(
                            session_id,
                            turn_id,
                            item_id,
                            content_index,
                            delta,
                        ));
                        if pending_live.len() >= LIVE_FLUSH_MAX_EVENTS {
                            try_emit_session_live_event_batch(
                                &frames_tx,
                                std::mem::take(&mut pending_live),
                            );
                        }
                    }
                    AppServerEvent::ToolOutputDelta { turn_id, tool_name, tool_call_id, delta } => {
                        if delta.is_empty() {
                            continue;
                        }
                        pending_live.push(make_tool_output_delta_live_event(
                            session_id,
                            turn_id,
                            tool_call_id,
                            tool_name,
                            delta,
                        ));
                        if pending_live.len() >= LIVE_FLUSH_MAX_EVENTS {
                            try_emit_session_live_event_batch(&frames_tx, std::mem::take(&mut pending_live));
                        }
                    }
                    event => {
                        if !pending_live.is_empty() {
                            try_emit_session_live_event_batch(&frames_tx, std::mem::take(&mut pending_live));
                        }
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
                }
            }
            _ = flush_interval.tick() => {
                if !pending_live.is_empty() {
                    try_emit_session_live_event_batch(&frames_tx, std::mem::take(&mut pending_live));
                }
            }
        }
    }

    if !pending_live.is_empty() {
        try_emit_session_live_event_batch(&frames_tx, std::mem::take(&mut pending_live));
    }
    Ok(())
}

fn spawn_event_forwarder(
    frames_tx: mpsc::Sender<DaemonFrame>,
    artifact_store: LocalArtifactStore,
    config: AppServerSupervisorConfig,
    session_id: SessionId,
    scope: SessionScope,
    rx: mpsc::Receiver<AppServerEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        match run_event_forwarder(frames_tx, artifact_store, config, session_id, scope, rx).await {
            Ok(()) => {
                tracing::debug!(%session_id, "app-server event forwarder exited");
            }
            Err(err) => {
                tracing::warn!(%session_id, error = %err, "app-server event forwarder exited unexpectedly");
            }
        }
    })
}

async fn emit_app_server_event(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    artifact_store: &LocalArtifactStore,
    config: &AppServerSupervisorConfig,
    session_id: SessionId,
    scope: SessionScope,
    event: AppServerEvent,
) -> Result<(), RunSessionError> {
    let (turn_id, kind) = match event {
        AppServerEvent::TurnStarted {
            turn_id,
            external_session_ref,
        } => (
            turn_id.clone(),
            SessionEventKind::TurnStarted(TurnStarted {
                interface_mode: InterfaceMode::Structured,
                external_session_ref,
                idempotency_key: None,
                log_offset_bytes: None,
            }),
        ),
        AppServerEvent::TurnCompleted {
            turn_id,
            external_session_ref,
            error,
        } => (
            turn_id.clone(),
            SessionEventKind::TurnCompleted(TurnCompleted {
                interface_mode: InterfaceMode::Structured,
                external_session_ref,
                exit_code: None,
                error,
            }),
        ),
        AppServerEvent::UserMessage { .. } => {
            // The control plane persists UserMessage events before dispatching daemon commands.
            // Avoid emitting duplicate copies of the same user text from the app-server runner.
            return Ok(());
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
            (None, SessionEventKind::AssistantMessage(ev))
        }
        AppServerEvent::AssistantMessageDelta {
            turn_id,
            item_id,
            delta,
        } => {
            let event = make_assistant_delta_live_event(session_id, turn_id, item_id, delta);
            try_emit_session_live_event_batch(frames_tx, vec![event]);
            return Ok(());
        }
        AppServerEvent::AssistantReasoningSummaryPartAdded {
            turn_id,
            item_id,
            summary_index,
        } => {
            let event = make_reasoning_summary_part_added_live_event(
                session_id,
                turn_id,
                item_id,
                summary_index,
            );
            try_emit_session_live_event_batch(frames_tx, vec![event]);
            return Ok(());
        }
        AppServerEvent::AssistantReasoningSummaryDelta {
            turn_id,
            item_id,
            summary_index,
            delta,
        } => {
            let event = make_reasoning_summary_delta_live_event(
                session_id,
                turn_id,
                item_id,
                summary_index,
                delta,
            );
            try_emit_session_live_event_batch(frames_tx, vec![event]);
            return Ok(());
        }
        AppServerEvent::AssistantReasoningRawDelta {
            turn_id,
            item_id,
            content_index,
            delta,
        } => {
            let event = make_reasoning_raw_delta_live_event(
                session_id,
                turn_id,
                item_id,
                content_index,
                delta,
            );
            try_emit_session_live_event_batch(frames_tx, vec![event]);
            return Ok(());
        }
        AppServerEvent::AssistantReasoning {
            turn_id,
            item_id,
            summary,
            raw,
            signature,
        } => {
            let mut summary = AssistantReasoningText {
                text: summary,
                preview: String::new(),
                full_text_artifact: None,
            };
            limit_message_event(
                artifact_store,
                config.max_message_chars,
                config.max_preview_chars,
                &mut summary.text,
                &mut summary.preview,
                &mut summary.full_text_artifact,
            )
            .await?;

            let raw = match raw {
                Some(raw) if !raw.trim().is_empty() => {
                    let mut raw = AssistantReasoningText {
                        text: raw,
                        preview: String::new(),
                        full_text_artifact: None,
                    };
                    limit_message_event(
                        artifact_store,
                        config.max_message_chars,
                        config.max_preview_chars,
                        &mut raw.text,
                        &mut raw.preview,
                        &mut raw.full_text_artifact,
                    )
                    .await?;
                    Some(raw)
                }
                _ => None,
            };

            (
                turn_id,
                SessionEventKind::AssistantReasoning(AssistantReasoning {
                    item_id,
                    summary,
                    raw,
                    signature,
                }),
            )
        }
        AppServerEvent::ToolOutputDelta {
            turn_id,
            tool_name,
            tool_call_id,
            delta,
        } => {
            let event = make_tool_output_delta_live_event(
                session_id,
                turn_id,
                tool_call_id,
                tool_name,
                delta,
            );
            try_emit_session_live_event_batch(frames_tx, vec![event]);
            return Ok(());
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
            (
                None,
                SessionEventKind::ToolInvocation(ToolInvocation {
                    tool_name,
                    tool_call_id,
                    input_preview,
                    input_artifact,
                }),
            )
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
            (
                None,
                SessionEventKind::ToolResult(ToolResult {
                    tool_name,
                    tool_call_id,
                    output_preview,
                    output_artifact,
                    error,
                }),
            )
        }
        AppServerEvent::PermissionRequested {
            request_id,
            summary,
            request,
        } => (
            None,
            SessionEventKind::PermissionRequested(PermissionRequested {
                request_id,
                summary,
                request,
            }),
        ),
        AppServerEvent::PermissionDecided {
            request_id,
            decision,
            decided_by,
        } => (
            None,
            SessionEventKind::PermissionDecided(PermissionDecided {
                request_id,
                decision,
                decided_by,
            }),
        ),
        AppServerEvent::ArtifactBytes {
            kind,
            mime,
            label,
            bytes,
        } => {
            let artifact = artifact_store.store_bytes(kind, mime, &bytes).await?;
            (
                None,
                SessionEventKind::ArtifactEmitted(ArtifactEmitted { artifact, label }),
            )
        }
        AppServerEvent::ArtifactRef { artifact, label } => (
            None,
            SessionEventKind::ArtifactEmitted(ArtifactEmitted { artifact, label }),
        ),
    };

    emit_event(frames_tx, session_id, scope, turn_id, kind).await?;
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

#[derive(Debug, thiserror::Error)]
enum EmitEventError {
    #[error("control plane session event stream closed")]
    StreamClosed,
}

fn make_session_event_frame(
    session_id: SessionId,
    scope: SessionScope,
    turn_id: Option<String>,
    kind: SessionEventKind,
) -> DaemonFrame {
    let record = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope,
        session_id,
        turn_id,
        kind,
    };
    let batch = SessionEventBatch {
        events: vec![record],
    };
    DaemonFrame::new(
        ProtocolEnvelope::new(),
        DaemonMessage::SessionEventBatch(batch),
    )
}

fn make_assistant_delta_live_event(
    session_id: SessionId,
    turn_id: Option<String>,
    item_id: Option<String>,
    delta: String,
) -> SessionLiveEvent {
    SessionLiveEvent {
        created_at: Timestamp::now_utc(),
        session_id,
        turn_id,
        item_id,
        kind: SessionLiveEventKind::AssistantMessageDelta(AssistantMessageDelta { delta }),
    }
}

fn make_reasoning_summary_part_added_live_event(
    session_id: SessionId,
    turn_id: Option<String>,
    item_id: Option<String>,
    summary_index: i64,
) -> SessionLiveEvent {
    SessionLiveEvent {
        created_at: Timestamp::now_utc(),
        session_id,
        turn_id,
        item_id,
        kind: SessionLiveEventKind::AssistantReasoningSummaryPartAdded(
            AssistantReasoningSummaryPartAdded { summary_index },
        ),
    }
}

fn make_reasoning_summary_delta_live_event(
    session_id: SessionId,
    turn_id: Option<String>,
    item_id: Option<String>,
    summary_index: i64,
    delta: String,
) -> SessionLiveEvent {
    SessionLiveEvent {
        created_at: Timestamp::now_utc(),
        session_id,
        turn_id,
        item_id,
        kind: SessionLiveEventKind::AssistantReasoningSummaryDelta(
            AssistantReasoningSummaryDelta {
                summary_index,
                delta,
            },
        ),
    }
}

fn make_reasoning_raw_delta_live_event(
    session_id: SessionId,
    turn_id: Option<String>,
    item_id: Option<String>,
    content_index: i64,
    delta: String,
) -> SessionLiveEvent {
    SessionLiveEvent {
        created_at: Timestamp::now_utc(),
        session_id,
        turn_id,
        item_id,
        kind: SessionLiveEventKind::AssistantReasoningRawDelta(AssistantReasoningRawDelta {
            content_index,
            delta,
        }),
    }
}

fn make_tool_output_delta_live_event(
    session_id: SessionId,
    turn_id: Option<String>,
    tool_call_id: Option<String>,
    tool_name: String,
    delta: String,
) -> SessionLiveEvent {
    SessionLiveEvent {
        created_at: Timestamp::now_utc(),
        session_id,
        turn_id,
        item_id: tool_call_id,
        kind: SessionLiveEventKind::ToolOutputDelta(ToolOutputDelta { tool_name, delta }),
    }
}

fn try_emit_session_live_event_batch(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    events: Vec<SessionLiveEvent>,
) {
    if events.is_empty() {
        return;
    }
    let frame = DaemonFrame::new(
        ProtocolEnvelope::new(),
        DaemonMessage::SessionLiveEventBatch(SessionLiveEventBatch { events }),
    );
    let _ = frames_tx.try_send(frame);
}

fn try_emit_status_update(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    session_id: SessionId,
    scope: SessionScope,
    message: String,
) {
    let frame = make_session_event_frame(
        session_id,
        scope,
        None,
        SessionEventKind::StatusUpdate(StatusUpdate {
            turn_state: TurnState::Running,
            blocking: None,
            progress_percent: None,
            message: Some(message),
        }),
    );
    let _ = frames_tx.try_send(frame);
}

fn session_reasoning_effort_from_client(
    effort: ModelReasoningEffort,
) -> SessionModelReasoningEffort {
    match effort {
        ModelReasoningEffort::Minimal => SessionModelReasoningEffort::Minimal,
        ModelReasoningEffort::Low => SessionModelReasoningEffort::Low,
        ModelReasoningEffort::Medium => SessionModelReasoningEffort::Medium,
        ModelReasoningEffort::High => SessionModelReasoningEffort::High,
        ModelReasoningEffort::Xhigh => SessionModelReasoningEffort::Xhigh,
        ModelReasoningEffort::Unknown => SessionModelReasoningEffort::Unknown,
    }
}

async fn emit_event(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    session_id: SessionId,
    scope: SessionScope,
    turn_id: Option<String>,
    kind: SessionEventKind,
) -> Result<(), EmitEventError> {
    frames_tx
        .send(make_session_event_frame(session_id, scope, turn_id, kind))
        .await
        .map_err(|_| EmitEventError::StreamClosed)
}
