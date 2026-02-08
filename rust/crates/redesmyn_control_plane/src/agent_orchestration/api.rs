use redesmyn_ids::{RepoId, TaskId, WorkspaceId};
use redesmyn_logging::tracing;
use redesmyn_protocol::client::{
    AgentMessageConflictAction, AttachAgentSessionRequest, AttachAgentSessionResponse,
    RestartAgentRequest, RestartAgentResponse, SendTaskAgentMessageRequest,
    SendTaskAgentMessageResponse, SessionModelSelection, StartAgentRequest, StartAgentResponse,
    StopAgentRequest, StopAgentResponse,
};
use redesmyn_protocol::{
    CodexApprovalPolicy, CodexSandboxPolicy, ErrorCategory, ErrorDetail, ErrorEnvelope, RepoScope,
};

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
        if matches!(
            req.session_model_selection
                .as_ref()
                .and_then(|selection| selection.reasoning_effort),
            Some(redesmyn_protocol::client::ModelReasoningEffort::Unknown)
        ) {
            return Err(super::conflicts::invalid_request(
                "Unknown model reasoning effort.",
            ));
        }
        if matches!(
            req.codex_approval_policy,
            Some(CodexApprovalPolicy::Unknown)
        ) {
            return Err(super::conflicts::invalid_request(
                "Unknown Codex approval policy.",
            ));
        }
        if matches!(
            req.codex_sandbox_policy.as_ref(),
            Some(CodexSandboxPolicy::Unknown)
        ) {
            return Err(super::conflicts::invalid_request(
                "Unknown Codex sandbox policy.",
            ));
        }
        if let Some(CodexSandboxPolicy::WorkspaceWrite { writable_roots, .. }) =
            req.codex_sandbox_policy.as_ref()
            && !writable_roots.is_empty()
        {
            return Err(super::conflicts::invalid_request(
                "writable_roots is not supported via the control plane.",
            ));
        }
        if req.agent_kind != redesmyn_protocol::client::AgentKind::Codex
            && (req.codex_approval_policy.is_some() || req.codex_sandbox_policy.is_some())
        {
            return Err(super::conflicts::invalid_request(
                "Codex session policy defaults are only supported for Codex sessions.",
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
            req.session_model_selection,
            req.codex_approval_policy,
            req.codex_sandbox_policy,
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

        let RestartAgentRequest {
            task_id,
            agent_kind,
            initial_prompt,
            session_model_selection,
            codex_approval_policy,
            codex_sandbox_policy,
        } = req;

        let request_fallback = RestartStickySettings {
            session_model_selection,
            codex_approval_policy,
            codex_sandbox_policy,
        };
        let sticky = load_restart_sticky_settings(self, task_id, request_fallback).await?;

        let stop = StopAgentRequest { task_id };
        self.stop_agent(workspace_id, repo_id, stop)
            .await
            .map_err(|err| {
                err.with_detail(ErrorDetail::from([(
                    "restart_phase".to_string(),
                    "stop".to_string(),
                )]))
            })?;

        let start = StartAgentRequest {
            task_id,
            agent_kind,
            initial_prompt,
            on_conflict: AgentMessageConflictAction::StopSessionAndStartNew,
            session_model_selection: sticky.session_model_selection,
            codex_approval_policy: sticky.codex_approval_policy,
            codex_sandbox_policy: sticky.codex_sandbox_policy,
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

#[derive(Debug, Clone, Default)]
struct RestartStickySettings {
    session_model_selection: Option<SessionModelSelection>,
    codex_approval_policy: Option<CodexApprovalPolicy>,
    codex_sandbox_policy: Option<CodexSandboxPolicy>,
}

impl RestartStickySettings {
    fn is_complete(&self) -> bool {
        self.session_model_selection.is_some()
            && self.codex_approval_policy.is_some()
            && self.codex_sandbox_policy.is_some()
    }

    fn merge_missing_from_policy_snapshot(
        &mut self,
        snapshot: redesmyn_protocol::agent_commands::SessionPolicySnapshot,
    ) {
        if self.session_model_selection.is_none()
            && (snapshot.model_id.is_some() || snapshot.model_reasoning_effort.is_some())
        {
            self.session_model_selection = Some(SessionModelSelection {
                model_id: snapshot.model_id,
                reasoning_effort: snapshot.model_reasoning_effort,
            });
        }
        if self.codex_approval_policy.is_none() {
            self.codex_approval_policy = snapshot.codex_approval_policy;
        }
        if self.codex_sandbox_policy.is_none() {
            self.codex_sandbox_policy = snapshot.codex_sandbox_policy;
        }
    }

    fn merge_missing_from_projection(
        &mut self,
        projection: crate::session_events_projection::PolicySnapshotProjection,
    ) {
        if self.session_model_selection.is_none()
            && (projection.model_id.is_some() || projection.model_reasoning_effort.is_some())
        {
            self.session_model_selection = Some(SessionModelSelection {
                model_id: projection.model_id,
                reasoning_effort: projection.model_reasoning_effort,
            });
        }
        if self.codex_approval_policy.is_none() {
            self.codex_approval_policy = projection.codex_approval_policy;
        }
        if self.codex_sandbox_policy.is_none() {
            self.codex_sandbox_policy = projection.codex_sandbox_policy;
        }
    }
}

async fn load_restart_sticky_settings(
    control_plane: &ControlPlane,
    task_id: TaskId,
    request_fallback: RestartStickySettings,
) -> Result<RestartStickySettings, ErrorEnvelope> {
    let mut sticky = request_fallback;

    if !sticky.is_complete() {
        let projected = crate::session_events_projection::load_task_last_seen_policy_projection(
            control_plane.pool(),
            task_id,
        )
        .await
        .map_err(|err| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                "Failed to load task policy projection.",
            )
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;
        sticky.merge_missing_from_projection(projected);
    }

    if !sticky.is_complete() {
        let recent_sessions =
            state::load_recent_task_sessions(control_plane.pool(), task_id, 50).await?;
        for session in recent_sessions {
            let snapshot = match crate::policy_snapshot::load_session_policy_snapshot(
                control_plane.session_events(),
                session.session_id,
            )
            .await
            {
                Ok(snapshot) => snapshot,
                Err(err) => {
                    tracing::warn!(
                        task_id = %task_id,
                        session_id = %session.session_id,
                        error = %err,
                        "failed to load task session policy snapshot while resolving restart settings"
                    );
                    continue;
                }
            };

            sticky.merge_missing_from_policy_snapshot(snapshot);
            if sticky.is_complete() {
                break;
            }
        }
    }

    Ok(sticky)
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
