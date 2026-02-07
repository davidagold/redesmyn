use std::path::PathBuf;
use std::sync::Arc;

use redesmyn_config::DaemonConfig;
use redesmyn_domain::agent::ExternalSessionRef as DomainExternalSessionRef;
use redesmyn_exec::app_server::{
    AppServerProcess, AppServerRequest, AppServerResponse, AppServerSupervisor,
    AppServerSupervisorConfig,
};
use redesmyn_exec::artifact_store::LocalArtifactStore;
use redesmyn_exec::codex_app_server::{CodexAppServerProcess, CodexAppServerProcessConfig};
use redesmyn_git::{
    GitBackend, GitRefName, GitRevision, GitRunOptions, GitWorktreeAddOptions, GitWorktreeTarget,
};
use redesmyn_logging::tracing;
use redesmyn_protocol::agent_commands::{
    AGENT_LIST_MODELS, InterruptTaskAgentTurnCommand, ListAgentModelsCommand,
    ListSessionModelsCommand, RespondPermissionRequestCommand, ResumeByIdTaskAgentTurnCommand,
    SESSION_AGENT_INTERRUPT_TURN, SESSION_AGENT_LIST_MODELS,
    SESSION_AGENT_RESPOND_PERMISSION_REQUEST, SESSION_AGENT_RESUME_BY_ID_TURN,
    SESSION_AGENT_SET_CODEX_APPROVAL_POLICY, SESSION_AGENT_SET_CODEX_SANDBOX_POLICY,
    SESSION_AGENT_SET_MODEL, SESSION_AGENT_SET_PERMISSIONS_MODE, SESSION_AGENT_START,
    SetSessionCodexApprovalPolicyCommand, SetSessionCodexSandboxPolicyCommand,
    SetSessionModelCommand, SetSessionPermissionsModeCommand, StartAgentSessionCommand,
    StartTaskAgentSessionCommand, TASK_AGENT_START,
};
use redesmyn_protocol::client::AgentKind;
use redesmyn_protocol::daemon::{
    CommandDispatch, CommandProgress, CommandState, CommandUpdate, DaemonFrame, DaemonMessage,
};
use redesmyn_protocol::session::{ExternalSessionRef, SessionScope};
use redesmyn_protocol::{
    ErrorCategory, ErrorDetail, ErrorEnvelope, ProtocolEnvelope, RepoScope, Scope,
};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;

use crate::agent_driver::{AgentDriver, CodexDriver, ResumeByIdTurnSpec, StartSessionSpec};
use crate::host_identity::HostIdentity;
use crate::repo::{RepoRegistry, RepoRegistryError};

#[derive(Clone)]
struct CommandRouter {
    repo_registry: Arc<dyn RepoRegistry>,
    git_backend: Arc<dyn GitBackend>,
    worktree_root: PathBuf,
    codex_driver: Arc<dyn AgentDriver>,
}

pub async fn run_command_router(
    identity: HostIdentity,
    daemon: DaemonConfig,
    repo_registry: Arc<dyn RepoRegistry>,
    git_backend: Arc<dyn GitBackend>,
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

    let codex_driver = Arc::new(CodexDriver::new(app_server));

    let router = CommandRouter {
        repo_registry,
        git_backend,
        worktree_root: daemon.worktree_root.clone(),
        codex_driver,
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

async fn handle_dispatch(
    router: CommandRouter,
    frames_tx: mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
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
        SESSION_AGENT_SET_MODEL => {
            handle_session_agent_set_model(router, &frames_tx, dispatch).await
        }
        SESSION_AGENT_LIST_MODELS => {
            handle_session_agent_list_models(router, &frames_tx, dispatch).await
        }
        AGENT_LIST_MODELS => handle_agent_list_models(router, &frames_tx, dispatch).await,
        SESSION_AGENT_RESPOND_PERMISSION_REQUEST => {
            handle_session_agent_respond_permission_request(router, &frames_tx, dispatch).await
        }
        _ => {
            reject_command(
                &frames_tx,
                dispatch,
                ErrorEnvelope::new(ErrorCategory::InvalidRequest, "Unknown command kind."),
            )
            .await
        }
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
        task_branch_name: cmd.task_branch_name,
        agent_kind: cmd.agent_kind,
        initial_prompt: cmd.initial_prompt,
        image_attachments: Vec::new(),
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

    let scope = session_scope(cmd.task_id);
    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };
    let working_directory = if cmd.task_id.is_some() {
        let Some(branch_name) = cmd
            .task_branch_name
            .clone()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        else {
            fail_command(
                frames_tx,
                dispatch,
                ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Task branch is required to start task sessions.",
                ),
            )
            .await;
            return;
        };

        match ensure_task_worktree(
            &router.git_backend,
            &repo_root,
            &router.worktree_root,
            &branch_name,
        )
        .await
        {
            Ok(path) => path,
            Err(err) => {
                fail_command(frames_tx, dispatch, err).await;
                return;
            }
        }
    } else {
        repo_root.clone()
    };

    let spec = StartSessionSpec {
        session_id: cmd.session_id,
        task_id: cmd.task_id,
        scope,
        repo_root,
        working_directory,
        initial_prompt: cmd.initial_prompt,
        image_attachments: cmd.image_attachments,
        policy_snapshot: cmd.policy_snapshot,
        stop_session_ids: cmd.stop_session_ids,
    };

    if let Err(err) = router.codex_driver.start_session(spec).await {
        fail_command(frames_tx, dispatch, err).await;
        return;
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

    let scope = session_scope(cmd.task_id);
    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    let external = match domain_external_session_ref(&cmd.external_session_ref) {
        Ok(external) => external,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };
    let spec = ResumeByIdTurnSpec {
        session_id: cmd.session_id,
        task_id: cmd.task_id,
        scope,
        repo_root,
        external_session_ref: external,
        prompt: cmd.prompt,
        image_attachments: cmd.image_attachments,
        policy_snapshot: cmd.policy_snapshot,
        interrupt_turn: cmd.interrupt_turn,
    };

    if let Err(err) = router.codex_driver.resume_by_id_turn(spec).await {
        fail_command(frames_tx, dispatch, err).await;
        return;
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

    match router.codex_driver.interrupt_turn(cmd.session_id).await {
        Ok(()) => {
            let _ = send_command_update(
                frames_tx,
                &dispatch,
                CommandState::Succeeded,
                Some("interrupted".to_owned()),
                None,
                None,
                None,
            )
            .await;
        }
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
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

    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    match router
        .codex_driver
        .set_permissions_mode(repo_root, cmd.session_id, cmd.task_id, cmd.mode)
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
            fail_command(frames_tx, dispatch, err).await;
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

    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    match router
        .codex_driver
        .set_codex_approval_policy(repo_root, cmd.session_id, cmd.task_id, cmd.approval_policy)
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
            fail_command(frames_tx, dispatch, err).await;
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

    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    match router
        .codex_driver
        .set_codex_sandbox_policy(repo_root, cmd.session_id, cmd.task_id, cmd.sandbox_policy)
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
            fail_command(frames_tx, dispatch, err).await;
        }
    }
}

async fn handle_session_agent_set_model(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: SetSessionModelCommand = match decode_payload(&dispatch) {
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

    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    match router
        .codex_driver
        .set_model(
            repo_root,
            cmd.session_id,
            cmd.task_id,
            redesmyn_protocol::client::SessionModelSelection {
                model_id: cmd.model_id,
                reasoning_effort: cmd.reasoning_effort,
            },
        )
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
            fail_command(frames_tx, dispatch, err).await;
        }
    }
}

async fn handle_session_agent_list_models(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: ListSessionModelsCommand = match decode_payload(&dispatch) {
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

    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    match router
        .codex_driver
        .list_models(repo_root, cmd.session_id, cmd.task_id)
        .await
    {
        Ok((options, selection)) => {
            let detail =
                match serde_json::to_string(&redesmyn_protocol::client::ListSessionModelsResponse {
                    options,
                    selection,
                }) {
                    Ok(models_json) => Some(ErrorDetail::from([(
                        "models_json".to_string(),
                        models_json,
                    )])),
                    Err(err) => {
                        tracing::warn!(error = %err, "failed to encode list-models detail payload");
                        None
                    }
                };

            let _ = send_command_update(
                frames_tx,
                &dispatch,
                CommandState::Succeeded,
                Some("listed".to_owned()),
                None,
                detail,
                None,
            )
            .await;
        }
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
        }
    }
}

async fn handle_agent_list_models(
    router: CommandRouter,
    frames_tx: &mpsc::Sender<DaemonFrame>,
    dispatch: CommandDispatch,
) {
    let cmd: ListAgentModelsCommand = match decode_payload(&dispatch) {
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

    let repo_root = match resolve_repo_root(&router.repo_registry, dispatch.scope) {
        Ok(path) => path,
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
            return;
        }
    };

    let list_result = match cmd.agent_kind {
        AgentKind::Codex => list_codex_models_from_app_server(repo_root).await,
        AgentKind::ClaudeCode | AgentKind::Shell => Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!(
                "Unsupported agent kind for model listing: {:?}",
                cmd.agent_kind
            ),
        )),
    };

    match list_result {
        Ok((options, selection)) => {
            let detail =
                match serde_json::to_string(&redesmyn_protocol::client::ListSessionModelsResponse {
                    options,
                    selection,
                }) {
                    Ok(models_json) => Some(ErrorDetail::from([(
                        "models_json".to_string(),
                        models_json,
                    )])),
                    Err(err) => {
                        tracing::warn!(error = %err, "failed to encode list-models detail payload");
                        None
                    }
                };

            let _ = send_command_update(
                frames_tx,
                &dispatch,
                CommandState::Succeeded,
                Some("listed".to_owned()),
                None,
                detail,
                None,
            )
            .await;
        }
        Err(err) => {
            fail_command(frames_tx, dispatch, err).await;
        }
    }
}

async fn list_codex_models_from_app_server(
    repo_root: PathBuf,
) -> Result<
    (
        Vec<redesmyn_protocol::client::SessionModelOption>,
        redesmyn_protocol::client::SessionModelSelection,
    ),
    ErrorEnvelope,
> {
    let process = CodexAppServerProcess::new(CodexAppServerProcessConfig::codex_default(repo_root));
    process
        .start()
        .await
        .map_err(|err| ErrorEnvelope::new(ErrorCategory::Unavailable, err.to_string()))?;

    let connection = process
        .connect()
        .await
        .map_err(|err| ErrorEnvelope::new(ErrorCategory::Unavailable, err.to_string()));

    let result = match connection {
        Ok(connection) => {
            let response = connection
                .client
                .request(AppServerRequest::ListModels)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Unavailable, err.to_string()))?;
            match response {
                AppServerResponse::ModelsListed { options, selection } => Ok((options, selection)),
                other => Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("unexpected response for model catalog request: {other:?}"),
                )),
            }
        }
        Err(err) => Err(err),
    };

    if let Err(err) = process.shutdown().await {
        tracing::warn!(error = %err, "failed to shutdown codex app-server after list models");
    }

    result
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
        .codex_driver
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
            fail_command(frames_tx, dispatch, err).await;
        }
    }
}

fn session_scope(task_id: Option<redesmyn_ids::TaskId>) -> SessionScope {
    match task_id {
        Some(task_id) => SessionScope::Task { task_id },
        None => SessionScope::Chat,
    }
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
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Repo registry is not configured.",
            )
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        }),
    }
}

fn default_worktree_path(worktree_root: &std::path::Path, branch_name: &str) -> PathBuf {
    let mut out = worktree_root.to_path_buf();
    let mut has_part = false;

    for part in branch_name.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            continue;
        }
        out.push(part.replace(':', "_"));
        has_part = true;
    }

    if !has_part {
        out.push(branch_name.replace(':', "_"));
    }

    out
}

fn worktree_branch_matches(worktree_branch: &GitRefName, branch_name: &str) -> bool {
    if worktree_branch.as_str() == branch_name {
        return true;
    }
    if let Some(stripped) = worktree_branch.as_str().strip_prefix("refs/heads/") {
        return stripped == branch_name;
    }
    false
}

async fn ensure_task_worktree(
    git_backend: &Arc<dyn GitBackend>,
    repo_root: &std::path::Path,
    worktree_root: &std::path::Path,
    branch_name: &str,
) -> Result<PathBuf, ErrorEnvelope> {
    let _branch = GitRefName::new(branch_name.to_string()).map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::InvalidRequest, "Invalid task branch name.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    let worktrees = git_backend
        .worktree_list(repo_root, GitRunOptions::default())
        .await
        .map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Unavailable, "Failed to list git worktrees.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;

    if let Some(existing) = worktrees
        .iter()
        .find(|worktree| {
            worktree
                .branch
                .as_ref()
                .is_some_and(|branch| worktree_branch_matches(branch, branch_name))
        })
    {
        return Ok(existing.path.clone());
    }

    let worktree_path = default_worktree_path(worktree_root, branch_name);

    if worktree_path.exists() {
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "Worktree path already exists but is not attached to the task branch.",
        )
        .with_detail(ErrorDetail::from([
            ("branch_name".to_string(), branch_name.to_string()),
            (
                "worktree_path".to_string(),
                worktree_path.display().to_string(),
            ),
        ])));
    }

    if let Some(parent) = worktree_path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "Failed to prepare worktree root.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())])));
    }

    let branch_revision = GitRevision::new(branch_name.to_string()).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Invalid task branch revision.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    let create_from_existing = git_backend
        .worktree_add(
            repo_root,
            &worktree_path,
            &GitWorktreeTarget::Revision(branch_revision),
            GitWorktreeAddOptions::default(),
            GitRunOptions::default(),
        )
        .await;

    if let Err(primary_err) = create_from_existing {
        let fallback = git_backend
            .worktree_add(
                repo_root,
                &worktree_path,
                &GitWorktreeTarget::Head,
                GitWorktreeAddOptions {
                    new_branch: Some(branch_name.to_string()),
                    ..GitWorktreeAddOptions::default()
                },
                GitRunOptions::default(),
            )
            .await;

        if let Err(fallback_err) = fallback {
            return Err(ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Failed to create task worktree.",
            )
            .with_detail(ErrorDetail::from([
                ("branch_name".to_string(), branch_name.to_string()),
                (
                    "worktree_path".to_string(),
                    worktree_path.display().to_string(),
                ),
                ("primary_error".to_string(), primary_err.to_string()),
                ("fallback_error".to_string(), fallback_err.to_string()),
            ])));
        }
    }

    Ok(worktree_path)
}

fn domain_external_session_ref(
    external: &ExternalSessionRef,
) -> Result<DomainExternalSessionRef, ErrorEnvelope> {
    match external {
        ExternalSessionRef::None => Ok(DomainExternalSessionRef::None),
        ExternalSessionRef::CodexThread { thread_id, turn_id } => {
            Ok(DomainExternalSessionRef::CodexThread {
                thread_id: thread_id.clone(),
                turn_id: turn_id.clone(),
            })
        }
        ExternalSessionRef::CodexSession {
            session_id,
            turn_id,
        } => Ok(DomainExternalSessionRef::CodexSession {
            session_id: session_id.clone(),
            turn_id: turn_id.clone(),
        }),
        ExternalSessionRef::ClaudeSession { session_id } => {
            Ok(DomainExternalSessionRef::ClaudeSession {
                session_id: session_id.clone(),
            })
        }
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
    envelope.scope = Some(Scope::Repo {
        repo: dispatch.scope,
    });

    frames_tx
        .send(DaemonFrame::new(
            envelope,
            DaemonMessage::CommandUpdate(update),
        ))
        .await
        .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::worktree_branch_matches;
    use redesmyn_git::GitRefName;

    #[test]
    fn worktree_branch_matches_short_and_full_head_refs() {
        let short =
            GitRefName::new("rn/director-v0/T-1-director-run-semantics").expect("valid short ref");
        let full = GitRefName::new("refs/heads/rn/director-v0/T-1-director-run-semantics")
            .expect("valid full ref");

        assert!(worktree_branch_matches(
            &short,
            "rn/director-v0/T-1-director-run-semantics"
        ));
        assert!(worktree_branch_matches(
            &full,
            "rn/director-v0/T-1-director-run-semantics"
        ));
    }
}
