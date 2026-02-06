use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use redesmyn_domain::agent::{AppServerTurnIntent, ExternalSessionRef as DomainExternalSessionRef};
use redesmyn_exec::app_server::{AppServerSessionSpec, AppServerSupervisor};
use redesmyn_exec::codex_app_server::{CodexAppServerProcess, CodexAppServerProcessConfig};
use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::agent_commands::SessionPolicySnapshot;
use redesmyn_protocol::client::{SessionModelOption, SessionModelSelection};
use redesmyn_protocol::session::{
    CodexApprovalPolicy, CodexSandboxPolicy, ImageAttachment, PermissionsMode, SessionScope,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope};

pub type DriverFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct StartSessionSpec {
    pub session_id: SessionId,
    pub task_id: Option<TaskId>,
    pub scope: SessionScope,
    pub repo_root: PathBuf,
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
        repo_root: PathBuf,
        session_id: SessionId,
        task_id: Option<TaskId>,
        scope: SessionScope,
    ) -> Result<(), ErrorEnvelope> {
        let process = Arc::new(CodexAppServerProcess::new(
            CodexAppServerProcessConfig::codex_default(repo_root),
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
}

impl AgentDriver for CodexDriver {
    fn start_session(&self, spec: StartSessionSpec) -> DriverFuture<'_, Result<(), ErrorEnvelope>> {
        let app_server = Arc::clone(&self.app_server);
        Box::pin(async move {
            for stop_session_id in spec.stop_session_ids {
                if let Err(err) = app_server.stop_session(stop_session_id).await {
                    redesmyn_logging::tracing::debug!(
                        session_id = %stop_session_id,
                        error = %err,
                        "failed to stop session"
                    );
                }
            }

            let process = Arc::new(CodexAppServerProcess::new(
                CodexAppServerProcessConfig::codex_default(spec.repo_root),
            ));
            let session_spec = AppServerSessionSpec {
                scope: spec.scope,
                allow_concurrent_for_task: false,
                process,
            };

            app_server
                .start_session(spec.session_id, spec.task_id, session_spec)
                .await
                .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;

            if let Some(snapshot) = spec.policy_snapshot {
                app_server
                    .hydrate_policies(spec.session_id, snapshot)
                    .await
                    .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            }

            if let Some(prompt) = spec.initial_prompt {
                let intent = AppServerTurnIntent::StartNew { prompt };
                app_server
                    .send_message(spec.session_id, intent, spec.image_attachments)
                    .await
                    .map_err(|err| ErrorEnvelope::new(ErrorCategory::Internal, err.to_string()))?;
            }

            Ok(())
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
