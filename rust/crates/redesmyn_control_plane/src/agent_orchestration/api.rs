use redesmyn_ids::{RepoId, TaskId, WorkspaceId};
use redesmyn_logging::tracing;
use redesmyn_protocol::client::{
    AgentMessageConflictAction, AttachAgentSessionRequest, AttachAgentSessionResponse,
    RestartAgentRequest, RestartAgentResponse, SendTaskAgentMessageRequest,
    SendTaskAgentMessageResponse, StartAgentRequest, StartAgentResponse, StopAgentRequest,
    StopAgentResponse,
};
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope, RepoScope};

use crate::ControlPlane;

use super::{executor, planner, state};

impl ControlPlane {
    pub async fn start_agent(
        &self,
        workspace_id: WorkspaceId,
        repo_id: RepoId,
        req: StartAgentRequest,
    ) -> Result<StartAgentResponse, ErrorEnvelope> {
        let span = tracing::debug_span!(
            "control_plane.agent.start",
            task_id = %req.task_id,
            agent_kind = ?req.agent_kind,
        );
        let _enter = span.enter();

        ensure_task_exists(self.pool(), workspace_id, repo_id, req.task_id).await?;
        if !state::is_daemon_supported_agent_kind(req.agent_kind) {
            return Err(super::conflicts::invalid_request(
                "Unsupported agent kind for daemon execution.",
            ));
        }

        let active_sessions = state::load_active_task_session_ids(self.pool(), req.task_id).await?;
        let plan = planner::plan_start_agent(active_sessions, req.on_conflict)?;

        executor::execute_start_agent(
            self,
            RepoScope {
                workspace_id,
                repo_id,
            },
            req.task_id,
            req.agent_kind,
            req.initial_prompt,
            plan.stop_session_ids,
        )
        .await
    }

    pub async fn stop_agent(
        &self,
        workspace_id: WorkspaceId,
        repo_id: RepoId,
        req: StopAgentRequest,
    ) -> Result<StopAgentResponse, ErrorEnvelope> {
        let span = tracing::debug_span!("control_plane.agent.stop", task_id = %req.task_id);
        let _enter = span.enter();

        ensure_task_exists(self.pool(), workspace_id, repo_id, req.task_id).await?;

        let session_ids = state::load_active_task_session_ids(self.pool(), req.task_id).await?;
        let plan = planner::plan_stop_agent(session_ids);

        executor::execute_stop_agent(self, workspace_id, repo_id, req.task_id, plan.session_ids)
            .await
    }

    pub async fn restart_agent(
        &self,
        workspace_id: WorkspaceId,
        repo_id: RepoId,
        req: RestartAgentRequest,
    ) -> Result<RestartAgentResponse, ErrorEnvelope> {
        let span = tracing::debug_span!(
            "control_plane.agent.restart",
            task_id = %req.task_id,
            agent_kind = ?req.agent_kind,
        );
        let _enter = span.enter();

        let stop = StopAgentRequest {
            task_id: req.task_id,
        };
        let _ = self.stop_agent(workspace_id, repo_id, stop).await;

        let start = StartAgentRequest {
            task_id: req.task_id,
            agent_kind: req.agent_kind,
            initial_prompt: req.initial_prompt,
            on_conflict: AgentMessageConflictAction::StopSessionAndStartNew,
        };

        let started = self.start_agent(workspace_id, repo_id, start).await?;
        Ok(RestartAgentResponse {
            command: started.command,
            session_id: started.session_id,
        })
    }

    pub async fn attach_agent_session(
        &self,
        workspace_id: WorkspaceId,
        repo_id: RepoId,
        req: AttachAgentSessionRequest,
    ) -> Result<AttachAgentSessionResponse, ErrorEnvelope> {
        let span = tracing::debug_span!(
            "control_plane.agent.attach",
            session_id = %req.session_id
        );
        let _enter = span.enter();

        executor::execute_attach_agent_session(self, workspace_id, repo_id, req.session_id).await
    }

    pub async fn send_task_agent_message(
        &self,
        workspace_id: WorkspaceId,
        repo_id: RepoId,
        req: SendTaskAgentMessageRequest,
    ) -> Result<SendTaskAgentMessageResponse, ErrorEnvelope> {
        let span = tracing::debug_span!(
            "control_plane.agent.send_task_agent_message",
            task_id = %req.task_id,
            agent_kind = ?req.agent_kind,
            on_conflict = ?req.on_conflict,
            interrupt = ?req.interrupt,
        );
        let _enter = span.enter();

        let trimmed = req.message.trim();
        if trimmed.is_empty() {
            return Err(super::conflicts::invalid_request("Message is empty."));
        }
        if !state::is_daemon_supported_agent_kind(req.agent_kind) {
            return Err(super::conflicts::invalid_request(
                "Unsupported agent kind for daemon execution.",
            ));
        }

        ensure_task_exists(self.pool(), workspace_id, repo_id, req.task_id).await?;

        let effective_on_conflict = planner::effective_on_conflict(req.on_conflict, req.interrupt);

        let recent_sessions =
            state::load_recent_task_sessions(self.pool(), req.task_id, 50).await?;
        let resumable_structured =
            state::load_resumable_structured_session(self.pool(), &recent_sessions, req.agent_kind)
                .await?;

        let plan = planner::plan_send_task_agent_message(
            &recent_sessions,
            req.agent_kind,
            effective_on_conflict,
            resumable_structured.as_ref(),
        )?;

        match plan {
            planner::SendTaskAgentMessagePlan::StructuredResume(resume) => {
                executor::execute_structured_resume(
                    self,
                    workspace_id,
                    repo_id,
                    req.task_id,
                    trimmed,
                    resume,
                )
                .await
            }
            planner::SendTaskAgentMessagePlan::StructuredStart(plan) => {
                executor::execute_new_session_send(
                    self,
                    RepoScope {
                        workspace_id,
                        repo_id,
                    },
                    req.task_id,
                    req.agent_kind,
                    trimmed,
                    plan,
                )
                .await
            }
        }
    }
}

async fn ensure_task_exists(
    pool: &sqlx::SqlitePool,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
) -> Result<(), ErrorEnvelope> {
    let task_exists =
        redesmyn_storage::sessions::task_exists_in_repo(pool, workspace_id, repo_id, task_id)
            .await
            .map_err(|err| {
                ErrorEnvelope::new(ErrorCategory::Internal, "Failed to query task.")
                    .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
            })?;
    if !task_exists {
        return Err(
            ErrorEnvelope::new(ErrorCategory::NotFound, "Task not found.").with_detail(
                ErrorDetail::from([("task_id".to_string(), task_id.to_string())]),
            ),
        );
    }
    Ok(())
}
