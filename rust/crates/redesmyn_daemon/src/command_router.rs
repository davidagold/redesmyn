use std::path::PathBuf;
use std::sync::Arc;

use redesmyn_config::DaemonConfig;
use redesmyn_domain::agent::{AppServerTurnIntent, ExternalSessionRef as DomainExternalSessionRef};
use redesmyn_exec::app_server::{AppServerSessionSpec, AppServerSupervisor, AppServerSupervisorConfig};
use redesmyn_exec::artifact_store::LocalArtifactStore;
use redesmyn_exec::codex_app_server::{CodexAppServerProcess, CodexAppServerProcessConfig};
use redesmyn_logging::tracing;
use redesmyn_protocol::agent_commands::{
    InterruptTaskAgentTurnCommand, RespondPermissionRequestCommand, ResumeByIdTaskAgentTurnCommand,
    SetSessionCodexApprovalPolicyCommand, SetSessionCodexSandboxPolicyCommand,
    SetSessionPermissionsModeCommand, StartAgentSessionCommand, StartTaskAgentSessionCommand,
    SESSION_AGENT_INTERRUPT_TURN, SESSION_AGENT_RESPOND_PERMISSION_REQUEST,
    SESSION_AGENT_RESUME_BY_ID_TURN, SESSION_AGENT_SEND_MESSAGE, SESSION_AGENT_SET_PERMISSIONS_MODE,
    SESSION_AGENT_SET_CODEX_APPROVAL_POLICY, SESSION_AGENT_SET_CODEX_SANDBOX_POLICY,
    SESSION_AGENT_START, TASK_AGENT_START,
};
use redesmyn_protocol::client::{AgentInterfaceMode, AgentKind};
use redesmyn_protocol::daemon::{
    CommandDispatch, CommandProgress, CommandState, CommandUpdate, DaemonFrame, DaemonMessage,
};
use redesmyn_protocol::session::{ExternalSessionRef, SessionScope};
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope, ProtocolEnvelope, RepoScope, Scope};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;

use crate::host_identity::HostIdentity;
use crate::repo::{RepoRegistry, RepoRegistryError};

#[derive(Clone)]
struct CommandRouter {
    repo_registry: Arc<dyn RepoRegistry>,
    app_server: Arc<AppServerSupervisor>,
}

pub async fn run_command_router(
    identity: HostIdentity,
    daemon: DaemonConfig,
    repo_registry: Arc<dyn RepoRegistry>,
    mut rx: mpsc::Receiver<CommandDispatch>,
    frames_tx: mpsc::Sender<DaemonFrame>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let span = redesmyn_logging::redesmyn_info_span!("daemon.command_router");
    redesmyn_logging::span::record_host_id(&span, identity.host_id);
    redesmyn_logging::span::record_host_instance_id(&span, identity.host_instance_id);
    let _enter = span.enter();

    let artifact_root = daemon_state_dir(&daemon);
    let artifact_store = LocalArtifactStore::new(artifact_root);

    let app_server = match AppServerSupervisor::new(
        AppServerSupervisorConfig::default(),
        artifact_store,
        frames_tx.clone(),
    )
    .await
    {
        Ok(supervisor) => Arc::new(supervisor),
        Err(err) => {
            tracing::error!(error = %err, "failed to initialize app-server supervisor; session exec disabled");
            return;
        }
    };

    let router = CommandRouter {
        repo_registry,
        app_server,
    };

    let mut tasks = JoinSet::new();

    loop {
        if *shutdown_rx.borrow() {
            return;
        }

        tokio::select! {
            _ = shutdown_rx.changed() => {}
            Some(join_result) = tasks.join_next() => {
                if let Err(err) = join_result {
                    tracing::warn!(error = %err, "command handler task panicked");
                }
            }
            dispatch = rx.recv() => {
                let Some(dispatch) = dispatch else { return; };
                let router = router.clone();
                let frames_tx = frames_tx.clone();
                tasks.spawn(async move {
                    handle_dispatch(router, frames_tx, dispatch).await;
                });
            }
        }
    }
}

async fn handle_dispatch(router: CommandRouter, frames_tx: mpsc::Sender<DaemonFrame>, dispatch: CommandDispatch) {
    let span = tracing::info_span!(
        "daemon.command_router.dispatch",
        command_id = %dispatch.command_id,
        command_kind = %dispatch.command_kind,
    );
    let _enter = span.enter();

    match dispatch.command_kind.as_str() {
        SESSION_AGENT_START => handle_session_agent_start(router, &frames_tx, dispatch).await,
        TASK_AGENT_START => handle_task_agent_start(router, &frames_tx, dispatch).await,
        SESSION_AGENT_RESUME_BY_ID_TURN => {
            handle_session_agent_resume_by_id_turn(router, &frames_tx, dispatch).await
        }
        SESSION_AGENT_INTERRUPT_TURN => {
            handle_session_agent_interrupt_turn(router, &frames_tx, dispatch).await
        }
        SESSION_AGENT_SET_PERMISSIONS_MODE => {
            handle_session_agent_set_permissions_mode(router, &frames_tx, dispatch).await
        }
        SESSION_AGENT_SET_CODEX_APPROVAL_POLICY => {
            handle_session_agent_set_codex_approval_policy(router, &frames_tx, dispatch).await
        }
        SESSION_AGENT_SET_CODEX_SANDBOX_POLICY => {
            handle_session_agent_set_codex_sandbox_policy(router, &frames_tx, dispatch).await
        }
        SESSION_AGENT_RESPOND_PERMISSION_REQUEST => {
            handle_session_agent_respond_permission_request(router, &frames_tx, dispatch).await
        }
        SESSION_AGENT_SEND_MESSAGE => reject_command(
            &frames_tx,
            dispatch,
            ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "session.agent.send_message is not supported by this daemon yet.",
            ),
        )
        .await,
        _ => reject_command(
            &frames_tx,
            dispatch,
            ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Unknown command kind.",
            ),
        )
        .await,
    }
}

async fn handle_session_agent_start(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: StartAgentSessionCommand = match decode_payload(&dispatch) {
        Ok(cmd) => cmd,
        Err(err) => {
            reject_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    handle_agent_start(router, frames_tx, dispatch, cmd).await;
}

async fn handle_task_agent_start(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: StartTaskAgentSessionCommand = match decode_payload(&dispatch) {
        Ok(cmd) => cmd,
        Err(err) => {
            reject_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    let mapped = StartAgentSessionCommand {
        session_id: cmd.session_id,
        task_id: Some(cmd.task_id),
        agent_kind: cmd.agent_kind,
        interface_mode: cmd.interface_mode,
        initial_prompt: cmd.initial_prompt,
        policy_snapshot: cmd.policy_snapshot,
        stop_session_ids: cmd.stop_session_ids,
    };

    handle_agent_start(router, frames_tx, dispatch, mapped).await;
}

async fn handle_agent_start(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
    cmd: StartAgentSessionCommand,
) {
    if cmd.agent_kind != AgentKind::Codex {
        reject_command(
            frames_tx,
            dispatch,
            ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                format!("Unsupported agent kind: {:?}", cmd.agent_kind),
            ),
        )
        .await;
        return;
    }

    match cmd.interface_mode {
        AgentInterfaceMode::AppServer => {}
        AgentInterfaceMode::StructuredExec => {
            tracing::info!("structured_exec sessions use the app-server runner in this daemon");
        }
        AgentInterfaceMode::ShellTmux => {
            reject_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "shell_tmux sessions are not supported by this daemon yet.",
                ),
            )
            .await;
            return;
        }
    }

    if let Err(err) = send_command_update(
        frames_tx,
        &dispatch,
        CommandState::Accepted,
        Some("accepted".to_owned()),
        None,
        None,
        None,
    )
    .await
    {
        tracing::warn!(error = ?err, "failed to send accepted command update");
    }

    for stop_session_id in cmd.stop_session_ids {
        if let Err(err) = router.app_server.stop_session(stop_session_id).await {
            tracing::debug!(
                session_id = %stop_session_id,
                error = %err,
                "failed to stop session"
            );
        }
    }

    let scope = session_scope(cmd.task_id);
    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    let process = Arc::new(CodexAppServerProcess::new(CodexAppServerProcessConfig::codex_default(
        repo_root,
    )));

    let spec = AppServerSessionSpec {
        scope,
        allow_concurrent_for_task: false,
        process,
    };

    if let Err(err) = router
        .app_server
        .start_session(cmd.session_id, cmd.task_id, spec)
        .await
    {
        fail_command(
            frames_tx,
            dispatch,
            ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()),
        )
        .await;
        return;
    }

    if let Some(snapshot) = cmd.policy_snapshot.clone() {
        if let Err(err) = router
            .app_server
            .hydrate_policies(cmd.session_id, snapshot)
            .await
        {
            fail_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()),
            )
            .await;
            return;
        }
    }

    if let Some(prompt) = cmd.initial_prompt {
        let intent = AppServerTurnIntent::StartNew { prompt };
        if let Err(err) = router.app_server.send_message(cmd.session_id, intent).await {
            fail_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()),
            )
            .await;
            return;
        }
    }

    if let Err(err) = send_command_update(
        frames_tx,
        &dispatch,
        CommandState::Succeeded,
        Some("dispatched".to_owned()),
        None,
        None,
        None,
    )
    .await
    {
        tracing::warn!(error = ?err, "failed to send succeeded command update");
    }
}

async fn handle_session_agent_resume_by_id_turn(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: ResumeByIdTaskAgentTurnCommand = match decode_payload(&dispatch) {
        Ok(cmd) => cmd,
        Err(err) => {
            reject_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    if let Err(err) = send_command_update(frames_tx, &dispatch, CommandState::Accepted, Some("accepted".to_owned()), None, None, None).await {
        tracing::warn!(error = ?err, "failed to send accepted command update");
    }

    let scope = session_scope(cmd.task_id);
    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    let process = Arc::new(CodexAppServerProcess::new(CodexAppServerProcessConfig::codex_default(
        repo_root,
    )));

    let spec = AppServerSessionSpec {
        scope,
        allow_concurrent_for_task: false,
        process,
    };

    if let Err(err) = router
        .app_server
        .start_session(cmd.session_id, cmd.task_id, spec)
        .await
    {
        fail_command(frames_tx, dispatch, ErrorEnvelope::new(ErrorCategory::Internal, err.to_string())).await;
        return;
    }

    if let Some(snapshot) = cmd.policy_snapshot.clone() {
        if let Err(err) = router
            .app_server
            .hydrate_policies(cmd.session_id, snapshot)
            .await
        {
            fail_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()),
            )
            .await;
            return;
        }
    }

    if cmd.interrupt_turn {
        if let Err(err) = router.app_server.interrupt_session(cmd.session_id).await {
            tracing::warn!(session_id = %cmd.session_id, error = %err, "failed to interrupt session before resume");
        }
    }

    let external = match domain_external_session_ref(&cmd.external_session_ref) {
        Ok(external) => external,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    let intent = AppServerTurnIntent::Resume {
        external,
        prompt: cmd.prompt,
    };

    if let Err(err) = router.app_server.send_message(cmd.session_id, intent).await {
        fail_command(frames_tx, dispatch, ErrorEnvelope::new(ErrorCategory::Internal, err.to_string())).await;
        return;
    }

    if let Err(err) = send_command_update(frames_tx, &dispatch, CommandState::Succeeded, Some("dispatched".to_owned()), None, None, None).await {
        tracing::warn!(error = ?err, "failed to send succeeded command update");
    }
}

async fn handle_session_agent_interrupt_turn(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: InterruptTaskAgentTurnCommand = match decode_payload(&dispatch) {
        Ok(cmd) => cmd,
        Err(err) => {
            reject_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    if let Err(err) = send_command_update(frames_tx, &dispatch, CommandState::Accepted, Some("accepted".to_owned()), None, None, None).await {
        tracing::warn!(error = ?err, "failed to send accepted command update");
    }

    match router.app_server.interrupt_session(cmd.session_id).await {
        Ok(_) => {
            let _ = send_command_update(frames_tx, &dispatch, CommandState::Succeeded, Some("interrupted".to_owned()), None, None, None).await;
        }
        Err(err) => {
            fail_command(frames_tx, dispatch, ErrorEnvelope::new(ErrorCategory::Internal, err.to_string())).await;
        }
    }
}

async fn handle_session_agent_set_permissions_mode(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: SetSessionPermissionsModeCommand = match decode_payload(&dispatch) {
        Ok(cmd) => cmd,
        Err(err) => {
            reject_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    if let Err(err) = send_command_update(
        frames_tx,
        &dispatch,
        CommandState::Accepted,
        Some("accepted".to_owned()),
        None,
        None,
        None,
    )
    .await
    {
        tracing::warn!(error = ?err, "failed to send accepted command update");
    }

    if let Err(err) = ensure_app_server_session_started(
        &router,
        dispatch.scope,
        cmd.session_id,
        cmd.task_id,
    )
    .await
    {
        fail_command(frames_tx, dispatch, err).await;
        return;
    }

    match router
        .app_server
        .set_permissions_mode(cmd.session_id, cmd.mode)
        .await
    {
        Ok(()) => {
            let _ = send_command_update(
                frames_tx,
                &dispatch,
                CommandState::Succeeded,
                Some("updated".to_owned()),
                None,
                None,
                None,
            )
            .await;
        }
        Err(err) => {
            fail_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()),
            )
            .await;
        }
    }
}

async fn handle_session_agent_set_codex_approval_policy(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: SetSessionCodexApprovalPolicyCommand = match decode_payload(&dispatch) {
        Ok(cmd) => cmd,
        Err(err) => {
            reject_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    if let Err(err) = send_command_update(
        frames_tx,
        &dispatch,
        CommandState::Accepted,
        Some("accepted".to_owned()),
        None,
        None,
        None,
    )
    .await
    {
        tracing::warn!(error = ?err, "failed to send accepted command update");
    }

    if let Err(err) = ensure_app_server_session_started(
        &router,
        dispatch.scope,
        cmd.session_id,
        cmd.task_id,
    )
    .await
    {
        fail_command(frames_tx, dispatch, err).await;
        return;
    }

    match router
        .app_server
        .set_codex_approval_policy(cmd.session_id, cmd.approval_policy)
        .await
    {
        Ok(()) => {
            let _ = send_command_update(
                frames_tx,
                &dispatch,
                CommandState::Succeeded,
                Some("updated".to_owned()),
                None,
                None,
                None,
            )
            .await;
        }
        Err(err) => {
            fail_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()),
            )
            .await;
        }
    }
}

async fn handle_session_agent_set_codex_sandbox_policy(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: SetSessionCodexSandboxPolicyCommand = match decode_payload(&dispatch) {
        Ok(cmd) => cmd,
        Err(err) => {
            reject_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    if let Err(err) = send_command_update(
        frames_tx,
        &dispatch,
        CommandState::Accepted,
        Some("accepted".to_owned()),
        None,
        None,
        None,
    )
    .await
    {
        tracing::warn!(error = ?err, "failed to send accepted command update");
    }

    if let Err(err) = ensure_app_server_session_started(
        &router,
        dispatch.scope,
        cmd.session_id,
        cmd.task_id,
    )
    .await
    {
        fail_command(frames_tx, dispatch, err).await;
        return;
    }

    match router
        .app_server
        .set_codex_sandbox_policy(cmd.session_id, cmd.sandbox_policy)
        .await
    {
        Ok(()) => {
            let _ = send_command_update(
                frames_tx,
                &dispatch,
                CommandState::Succeeded,
                Some("updated".to_owned()),
                None,
                None,
                None,
            )
            .await;
        }
        Err(err) => {
            fail_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()),
            )
            .await;
        }
    }
}

async fn handle_session_agent_respond_permission_request(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: RespondPermissionRequestCommand = match decode_payload(&dispatch) {
        Ok(cmd) => cmd,
        Err(err) => {
            reject_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    if let Err(err) = send_command_update(
        frames_tx,
        &dispatch,
        CommandState::Accepted,
        Some("accepted".to_owned()),
        None,
        None,
        None,
    )
    .await
    {
        tracing::warn!(error = ?err, "failed to send accepted command update");
    }

    match router
        .app_server
        .respond_permission_request(cmd.session_id, cmd.request_id, cmd.decision)
        .await
    {
        Ok(()) => {
            let _ = send_command_update(
                frames_tx,
                &dispatch,
                CommandState::Succeeded,
                Some("responded".to_owned()),
                None,
                None,
                None,
            )
            .await;
        }
        Err(err) => {
            fail_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()),
            )
            .await;
        }
    }
}

fn session_scope(task_id: Option<redesmyn_ids::TaskId>) -> SessionScope {
    match task_id {
        Some(task_id) => SessionScope::Task { task_id },
        None => SessionScope::Chat,
    }
}

async fn ensure_app_server_session_started(
    router: &CommandRouter,
    repo_scope: RepoScope,
    session_id: redesmyn_ids::SessionId,
    task_id: Option<redesmyn_ids::TaskId>,
) -> Result<(), ErrorEnvelope> {
    let scope = session_scope(task_id);
    let repo_root = resolve_repo_root(&router.repo_registry, repo_scope)?;
    let process = Arc::new(CodexAppServerProcess::new(CodexAppServerProcessConfig::codex_default(
        repo_root,
    )));
    let spec = AppServerSessionSpec {
        scope,
        allow_concurrent_for_task: false,
        process,
    };

    router
        .app_server
        .start_session(session_id, task_id, spec)
        .await
        .map(|_| ())
        .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))
}

fn daemon_state_dir(daemon: &DaemonConfig) -> PathBuf {
    daemon
        .repo_registry_dir
        .parent()
        .unwrap_or(&daemon.repo_registry_dir)
        .to_path_buf()
}

fn resolve_repo_root(
    registry: &Arc<dyn RepoRegistry>,
    scope: RepoScope,
) -> Result<PathBuf, ErrorEnvelope> {
    match registry.resolve_repo_root(scope) {
        Ok(path) => Ok(path),
        Err(RepoRegistryError::Unconfigured) => std::env::current_dir().map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Unavailable, "Repo registry is not configured.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        }),
    }
}

fn domain_external_session_ref(
    external: &ExternalSessionRef,
) -> Result<DomainExternalSessionRef, ErrorEnvelope> {
    match external {
        ExternalSessionRef::None => Ok(DomainExternalSessionRef::None),
        ExternalSessionRef::CodexThread { thread_id, turn_id } => Ok(
            DomainExternalSessionRef::CodexThread {
                thread_id: thread_id.clone(),
                turn_id: turn_id.clone(),
            },
        ),
        ExternalSessionRef::CodexSession { session_id, turn_id } => Ok(
            DomainExternalSessionRef::CodexSession {
                session_id: session_id.clone(),
                turn_id: turn_id.clone(),
            },
        ),
        ExternalSessionRef::ClaudeSession { session_id } => Ok(DomainExternalSessionRef::ClaudeSession {
            session_id: session_id.clone(),
        }),
        ExternalSessionRef::Unknown {
            unknown_type,
            json_payload,
        } => {
            let raw = serde_json::from_slice(json_payload).unwrap_or(serde_json::Value::Null);
            Ok(DomainExternalSessionRef::Unknown {
                type_: unknown_type.clone(),
                raw,
            })
        }
    }
}

fn decode_payload<T>(dispatch: &CommandDispatch) -> Result<T, ErrorEnvelope>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_slice(&dispatch.json_payload).map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::InvalidRequest, "Invalid command payload.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })
}

async fn reject_command(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
    error: ErrorEnvelope,
) {
    let _ = send_command_update(
        frames_tx,
        &dispatch,
        CommandState::Rejected,
        Some(error.message.clone()),
        None,
        None,
        Some(error),
    )
    .await;
}

async fn fail_command(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
    error: ErrorEnvelope,
) {
    let _ = send_command_update(
        frames_tx,
        &dispatch,
        CommandState::Failed,
        Some(error.message.clone()),
        None,
        None,
        Some(error),
    )
    .await;
}

async fn send_command_update(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: &CommandDispatch,
    state: CommandState,
    message: Option<String>,
    progress: Option<CommandProgress>,
    detail: Option<ErrorDetail>,
    error: Option<ErrorEnvelope>,
) -> Result<(), ()> {
    let update = CommandUpdate {
        command_id: dispatch.command_id,
        state,
        message,
        progress,
        detail,
        error,
    };

    let mut envelope = ProtocolEnvelope::new();
    envelope.scope = Some(Scope::Repo { repo: dispatch.scope });

    frames_tx
        .send(DaemonFrame::new(
            envelope,
            DaemonMessage::CommandUpdate(update),
        ))
        .await
        .map_err(|_| ())
}
