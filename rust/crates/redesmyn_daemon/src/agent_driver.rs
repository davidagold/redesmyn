use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use redesmyn_domain::agent::{AppServerTurnIntent, ExternalSessionRef as DomainExternalSessionRef};
use redesmyn_exec::app_server::{
    AppServerSessionSpec, AppServerSupervisor, SessionControlError, StartSessionError,
};
use redesmyn_exec::codex_app_server::{CodexAppServerProcess, CodexAppServerProcessConfig};
use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::agent_commands::SessionPolicySnapshot;
use redesmyn_protocol::client::{SessionModelOption, SessionModelSelection};
use redesmyn_protocol::session::{
    CodexApprovalPolicy, CodexSandboxPolicy, ImageAttachment, InterfaceMode, PermissionsMode,
    SessionScope,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope};

pub type DriverFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct StartSessionSpec {
    pub session_id: SessionId,
    pub task_id: Option<TaskId>,
    pub scope: SessionScope,
    pub repo_root: PathBuf,
    pub working_directory: PathBuf,
    pub initial_prompt: Option<String>,
    pub image_attachments: Vec<ImageAttachment>,
    pub policy_snapshot: Option<SessionPolicySnapshot>,
    pub stop_session_ids: Vec<SessionId>,
}

#[derive(Debug, Clone)]
pub struct ResumeByIdTurnSpec {
    pub session_id: SessionId,
    pub task_id: Option<TaskId>,
    pub scope: SessionScope,
    pub repo_root: PathBuf,
    pub external_session_ref: DomainExternalSessionRef,
    pub prompt: String,
    pub image_attachments: Vec<ImageAttachment>,
    pub policy_snapshot: Option<SessionPolicySnapshot>,
    pub interrupt_turn: bool,
}

pub trait AgentDriver: Send + Sync {
    fn start_session(&self, spec: StartSessionSpec) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
    fn stop_session(&self, session_id: SessionId) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
    fn stop_task_sessions(
        &self,
        task_id: TaskId,
    ) -> DriverFuture<'_, Result<Vec<SessionId>, ErrorEnvelope>>;
    fn resume_by_id_turn(
        &self,
        spec: ResumeByIdTurnSpec,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
    fn interrupt_turn(&self, session_id: SessionId) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
    fn set_permissions_mode(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        mode: PermissionsMode,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
    fn set_codex_approval_policy(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        approval_policy: Option<CodexApprovalPolicy>,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
    fn set_codex_sandbox_policy(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        sandbox_policy: Option<CodexSandboxPolicy>,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
    fn set_model(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        selection: SessionModelSelection,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
    fn list_models(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
    ) -> DriverFuture<'_, Result<(Vec<SessionModelOption>, SessionModelSelection), ErrorEnvelope>>;
    fn respond_permission_request(
        &self,
        session_id: SessionId,
        request_id: String,
        decision: redesmyn_protocol::session::PermissionDecision,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>>;
}

#[derive(Clone)]
pub struct CodexDriver {
    app_server: Arc<AppServerSupervisor>,
}

impl CodexDriver {
    pub fn new(app_server: Arc<AppServerSupervisor>) -> Self {
        Self { app_server }
    }

    async fn ensure_session_started(
        &self,
        working_directory: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        scope: SessionScope,
    ) -> Result<(), ErrorEnvelope> {
        let process = Arc::new(CodexAppServerProcess::new(
            CodexAppServerProcessConfig::codex_default(working_directory),
        ));
        let spec = AppServerSessionSpec {
            scope,
            allow_concurrent_for_task: false,
            process,
        };

        self.app_server
            .start_session(session_id, task_id, spec)
            .await
            .map(|_| ())
            .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))
    }

    async fn stop_session_and_wait(&self, session_id: SessionId) -> Result<(), ErrorEnvelope> {
        match self.app_server.stop_session(session_id).await {
            Ok(()) | Err(SessionControlError::SessionClosed { .. }) => {}
            Err(SessionControlError::UnknownSession { .. }) => return Ok(()),
        }

        const STOP_WAIT_INTERVAL: Duration = Duration::from_millis(25);
        const STOP_WAIT_ATTEMPTS: usize = 160;
        for _ in 0..STOP_WAIT_ATTEMPTS {
            if !self.app_server.has_session(session_id).await {
                return Ok(());
            }
            tokio::time::sleep(STOP_WAIT_INTERVAL).await;
        }

        Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("Timed out waiting for session {} to stop.", session_id),
        ))
    }
}

impl AgentDriver for CodexDriver {
    fn start_session(&self, spec: StartSessionSpec) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let driver = self.clone();
        Box::pin(async move {
            for stop_session_id in spec.stop_session_ids {
                if let Err(err) = driver.stop_session_and_wait(stop_session_id).await {
                    redesmyn_logging::tracing::debug!(
                        session_id = %stop_session_id,
                        error = %err.message,
                        "failed to stop session"
                    );
                }
            }

            let process = Arc::new(CodexAppServerProcess::new(
                CodexAppServerProcessConfig::codex_default(spec.working_directory),
            ));
            let session_spec = AppServerSessionSpec {
                scope: spec.scope,
                allow_concurrent_for_task: false,
                process,
            };

            const CONFLICT_RETRY_LIMIT: usize = 8;
            let mut attempts = 0_usize;
            loop {
                match driver
                    .app_server
                    .start_session(spec.session_id, spec.task_id, session_spec.clone())
                    .await
                {
                    Ok(_) => break,
                    Err(StartSessionError::TaskHasActiveSession {
                        session_id: active_session_id,
                        interface_mode,
                    }) if spec.task_id.is_some() && attempts < CONFLICT_RETRY_LIMIT => {
                        attempts += 1;
                        redesmyn_logging::tracing::warn!(
                            requested_session_id = %spec.session_id,
                            conflicting_session_id = %active_session_id,
                            ?interface_mode,
                            attempt = attempts,
                            "start-session conflict detected; stopping conflicting runtime session"
                        );

                        if let Err(err) = driver.stop_session_and_wait(active_session_id).await {
                            return Err(err);
                        }
                    }
                    Err(err) => {
                        return Err(ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()));
                    }
                }
            }

            if let Some(snapshot) = spec.policy_snapshot {
                driver
                    .app_server
                    .hydrate_policies(spec.session_id, snapshot)
                    .await
                    .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            }

            if let Some(prompt) = spec.initial_prompt {
                let intent = AppServerTurnIntent::StartNew { prompt };
                driver
                    .app_server
                    .send_message(spec.session_id, intent, spec.image_attachments)
                    .await
                    .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            }

            Ok(())
        })
    }

    fn stop_session(&self, session_id: SessionId) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let driver = self.clone();
        Box::pin(async move { driver.stop_session_and_wait(session_id).await })
    }

    fn stop_task_sessions(
        &self,
        task_id: TaskId,
    ) -> DriverFuture<'_, Result<Vec<SessionId>, ErrorEnvelope>> {
        let driver = self.clone();
        Box::pin(async move {
            let active = driver
                .app_server
                .active_task_sessions(task_id, InterfaceMode::Structured)
                .await;
            for session_id in &active {
                driver.stop_session_and_wait(*session_id).await?;
            }
            Ok(active)
        })
    }

    fn resume_by_id_turn(
        &self,
        spec: ResumeByIdTurnSpec,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let driver = self.clone();
        Box::pin(async move {
            driver
                .ensure_session_started(spec.repo_root, spec.session_id, spec.task_id, spec.scope)
                .await?;

            if let Some(snapshot) = spec.policy_snapshot {
                driver
                    .app_server
                    .hydrate_policies(spec.session_id, snapshot)
                    .await
                    .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            }

            if spec.interrupt_turn {
                if let Err(err) = driver.app_server.interrupt_session(spec.session_id).await {
                    redesmyn_logging::tracing::warn!(
                        session_id = %spec.session_id,
                        error = %err,
                        "failed to interrupt session before resume"
                    );
                }
            }

            let intent = AppServerTurnIntent::Resume {
                external: spec.external_session_ref,
                prompt: spec.prompt,
            };

            driver
                .app_server
                .send_message(spec.session_id, intent, spec.image_attachments)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;

            Ok(())
        })
    }

    fn interrupt_turn(&self, session_id: SessionId) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let app_server = Arc::clone(&self.app_server);
        Box::pin(async move {
            app_server
                .interrupt_session(session_id)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            Ok(())
        })
    }

    fn set_permissions_mode(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        mode: PermissionsMode,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let driver = self.clone();
        Box::pin(async move {
            driver
                .ensure_session_started(repo_root, session_id, task_id, session_scope(task_id))
                .await?;

            driver
                .app_server
                .set_permissions_mode(session_id, mode)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            Ok(())
        })
    }

    fn set_codex_approval_policy(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        approval_policy: Option<CodexApprovalPolicy>,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let driver = self.clone();
        Box::pin(async move {
            driver
                .ensure_session_started(repo_root, session_id, task_id, session_scope(task_id))
                .await?;

            driver
                .app_server
                .set_codex_approval_policy(session_id, approval_policy)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            Ok(())
        })
    }

    fn set_codex_sandbox_policy(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        sandbox_policy: Option<CodexSandboxPolicy>,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let driver = self.clone();
        Box::pin(async move {
            driver
                .ensure_session_started(repo_root, session_id, task_id, session_scope(task_id))
                .await?;

            driver
                .app_server
                .set_codex_sandbox_policy(session_id, sandbox_policy)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            Ok(())
        })
    }

    fn respond_permission_request(
        &self,
        session_id: SessionId,
        request_id: String,
        decision: redesmyn_protocol::session::PermissionDecision,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let app_server = Arc::clone(&self.app_server);
        Box::pin(async move {
            app_server
                .respond_permission_request(session_id, request_id, decision)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            Ok(())
        })
    }

    fn set_model(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        selection: SessionModelSelection,
    ) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let driver = self.clone();
        Box::pin(async move {
            driver
                .ensure_session_started(repo_root, session_id, task_id, session_scope(task_id))
                .await?;

            driver
                .app_server
                .set_model(session_id, selection)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            Ok(())
        })
    }

    fn list_models(
        &self,
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
    ) -> DriverFuture<'_, Result<(Vec<SessionModelOption>, SessionModelSelection), ErrorEnvelope>>
    {
        let driver = self.clone();
        Box::pin(async move {
            driver
                .ensure_session_started(repo_root, session_id, task_id, session_scope(task_id))
                .await?;

            driver
                .app_server
                .list_models(session_id)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))
        })
    }
}

fn session_scope(task_id: Option<TaskId>) -> SessionScope {
    match task_id {
        Some(task_id) => SessionScope::Task { task_id },
        None => SessionScope::Chat,
    }
}
