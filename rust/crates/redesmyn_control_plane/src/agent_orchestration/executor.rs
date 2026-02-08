use std::time::{Duration, SystemTime, UNIX_EPOCH};

use redesmyn_ids::{CommandId, RepoId, SessionEventId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::agent_commands::{
    SESSION_AGENT_ATTACH_SESSION, SESSION_AGENT_RESUME_BY_ID_TURN, TASK_AGENT_START,
    TASK_AGENT_STOP,
};
use redesmyn_protocol::client::{
    AgentKind, AttachAgentSessionResponse, CommandState, CommandSummary, ModelReasoningEffort,
    SendTaskAgentMessageResponse, SessionModelSelection, StartAgentResponse, StopAgentResponse,
    TaskAgentMessageConversationContinuity, TaskAgentMessageDelivery,
};
use redesmyn_protocol::prelude::{
    PreludeTemplateContext, default_worktree_relative_path, render_prelude_template,
};
use redesmyn_protocol::session::{
    CodexApprovalPolicy, CodexApprovalPolicyChanged, CodexSandboxPolicy, CodexSandboxPolicyChanged,
    SessionEnded, SessionEventKind, SessionModelChanged, SessionModelReasoningEffort, SessionScope,
    SessionStarted,
};
use redesmyn_protocol::task_events::TASK_STATE_CHANGED_EVENT;
use redesmyn_protocol::{
    ErrorCategory, ErrorDetail, ErrorEnvelope, ExternalSessionRef, RepoScope, SessionEvent,
    Timestamp,
};
use sqlx::SqlitePool;

use crate::ControlPlane;

use super::payloads;
use super::planner::{NewSessionPlan, StructuredResumePlan};

type StorageAgentKind = redesmyn_storage::schema::AgentKind;
type StorageAgentSessionScopeKind = redesmyn_storage::schema::AgentSessionScopeKind;
type StorageAgentSessionStatus = redesmyn_storage::schema::AgentSessionStatus;

const START_COMMAND_WAIT_TIMEOUT: Duration = Duration::from_secs(15);

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

async fn insert_task_session(
    pool: &SqlitePool,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
    agent_kind: AgentKind,
) -> Result<SessionId, ErrorEnvelope> {
    use redesmyn_storage::sessions::{AgentSessionRecord, insert_agent_session};

    let session_id = SessionId::new();
    let now_ms = now_ms();

    let storage_kind = match agent_kind {
        AgentKind::Codex => StorageAgentKind::Codex,
        AgentKind::ClaudeCode => StorageAgentKind::ClaudeCode,
        AgentKind::Shell => StorageAgentKind::Shell,
    };

    let external_session_ref = serde_json::to_string(&ExternalSessionRef::None)
        .unwrap_or_else(|_| r#"{"type":"none"}"#.to_string());

    insert_agent_session(
        pool,
        &AgentSessionRecord {
            session_id,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: StorageAgentSessionScopeKind::Task,
            task_id: Some(task_id),
            agent_kind: storage_kind,
            status: StorageAgentSessionStatus::Running,
            external_session_ref,
            title: None,
            started_at_ms: Some(now_ms),
            ended_at_ms: None,
            closed_at_ms: None,
        },
    )
    .await
    .map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::Internal, "Failed to create agent session.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    Ok(session_id)
}

async fn load_task_branch_name(
    pool: &SqlitePool,
    task_id: TaskId,
) -> Result<String, ErrorEnvelope> {
    let branch_name: Option<String> = sqlx::query_scalar(
        r#"
        SELECT branch_name
        FROM tasks
        WHERE id = ?1
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await
    .map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::Internal, "Failed to load task branch.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?
    .flatten();

    let branch_name = branch_name
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Task branch is required to start an agent session.",
            )
            .with_detail(ErrorDetail::from([(
                "task_id".to_string(),
                task_id.to_string(),
            )]))
        })?;

    Ok(branch_name)
}

#[derive(Debug, Clone)]
struct TaskPreludeContextData {
    task_id: String,
    task_title: String,
    task_doc: String,
    epic_slug: String,
    epic_readme: String,
    branch: String,
    worktree: String,
}

async fn load_task_prelude_context(
    pool: &SqlitePool,
    task_id: TaskId,
) -> Result<TaskPreludeContextData, ErrorEnvelope> {
    let row: Option<(Option<String>, String, Option<String>, String)> = sqlx::query_as(
        r#"
        SELECT t.local_ref, t.title, t.branch_name, e.slug
        FROM tasks t
        JOIN epics e ON e.id = t.epic_id
        WHERE t.id = ?1
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await
    .map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to load task prelude context.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    let (local_ref, task_title, branch_name, epic_slug) = row.ok_or_else(|| {
        ErrorEnvelope::new(
            ErrorCategory::NotFound,
            "Task not found while preparing prelude.",
        )
        .with_detail(ErrorDetail::from([(
            "task_id".to_string(),
            task_id.to_string(),
        )]))
    })?;

    let task_id_value = local_ref
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| task_id.to_string());
    let branch = branch_name
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("rn/{epic_slug}/{task_id_value}"));
    let task_doc = format!("epics/{epic_slug}/tasks/{task_id_value}/README.md");
    let epic_readme = format!("epics/{epic_slug}/README.md");

    Ok(TaskPreludeContextData {
        task_id: task_id_value,
        task_title,
        task_doc,
        epic_slug,
        epic_readme,
        branch: branch.clone(),
        worktree: default_worktree_relative_path(&branch),
    })
}

async fn end_sessions(pool: &SqlitePool, session_ids: &[SessionId]) -> Result<(), ErrorEnvelope> {
    if session_ids.is_empty() {
        return Ok(());
    }

    let now_ms = now_ms();
    let mut query = sqlx::QueryBuilder::new("UPDATE agent_sessions SET updated_at_ms = ");
    query.push_bind(now_ms);
    query.push(", status = 'stopped', ended_at_ms = COALESCE(ended_at_ms, ");
    query.push_bind(now_ms);
    query.push(") WHERE session_id IN (");
    {
        let mut separated = query.separated(", ");
        for session_id in session_ids {
            separated.push_bind(*session_id);
        }
    }
    query.push(") AND ended_at_ms IS NULL");

    query.build().execute(pool).await.map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::Internal, "Failed to stop agent session.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    Ok(())
}

async fn append_session_started(
    control_plane: &ControlPlane,
    session_id: SessionId,
    task_id: TaskId,
) -> Result<(), ErrorEnvelope> {
    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Task { task_id },
        session_id,
        turn_id: None,
        kind: SessionEventKind::SessionStarted(SessionStarted {}),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Internal, "Failed to persist session event.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;
    Ok(())
}

async fn append_session_ended(
    control_plane: &ControlPlane,
    session_id: SessionId,
    task_id: TaskId,
) -> Result<(), ErrorEnvelope> {
    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Task { task_id },
        session_id,
        turn_id: None,
        kind: SessionEventKind::SessionEnded(SessionEnded {}),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Internal, "Failed to persist session event.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;
    Ok(())
}

fn message_preview(text: &str) -> String {
    const MAX: usize = 240;
    if text.chars().count() <= MAX {
        return text.to_string();
    }

    text.chars().take(MAX).collect()
}

async fn append_user_message(
    control_plane: &ControlPlane,
    session_id: SessionId,
    task_id: TaskId,
    text: &str,
) -> Result<(), ErrorEnvelope> {
    use redesmyn_protocol::session::UserMessage;

    let preview = message_preview(text);
    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Task { task_id },
        session_id,
        turn_id: None,
        kind: SessionEventKind::UserMessage(UserMessage {
            text: text.to_owned(),
            preview,
            full_text_artifact: None,
            image_attachments: Vec::new(),
        }),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Internal, "Failed to persist session event.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;
    Ok(())
}

async fn append_codex_approval_policy_changed(
    control_plane: &ControlPlane,
    session_id: SessionId,
    task_id: TaskId,
    approval_policy: CodexApprovalPolicy,
) -> Result<(), ErrorEnvelope> {
    if matches!(approval_policy, CodexApprovalPolicy::Unknown) {
        return Ok(());
    }

    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Task { task_id },
        session_id,
        turn_id: None,
        kind: SessionEventKind::CodexApprovalPolicyChanged(CodexApprovalPolicyChanged {
            approval_policy: Some(approval_policy),
        }),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Internal, "Failed to persist session event.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;
    Ok(())
}

async fn append_codex_sandbox_policy_changed(
    control_plane: &ControlPlane,
    session_id: SessionId,
    task_id: TaskId,
    sandbox_policy: &CodexSandboxPolicy,
) -> Result<(), ErrorEnvelope> {
    if matches!(sandbox_policy, CodexSandboxPolicy::Unknown) {
        return Ok(());
    }

    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Task { task_id },
        session_id,
        turn_id: None,
        kind: SessionEventKind::CodexSandboxPolicyChanged(CodexSandboxPolicyChanged {
            sandbox_policy: Some(sandbox_policy.clone()),
        }),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Internal, "Failed to persist session event.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;
    Ok(())
}

fn normalize_start_model_selection(
    selection: &SessionModelSelection,
) -> Option<SessionModelSelection> {
    let model_id = selection.model_id.as_ref().and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    });
    let reasoning_effort = selection
        .reasoning_effort
        .and_then(|effort| (!matches!(effort, ModelReasoningEffort::Unknown)).then_some(effort));
    (model_id.is_some() || reasoning_effort.is_some()).then_some(SessionModelSelection {
        model_id,
        reasoning_effort,
    })
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

async fn append_session_model_changed(
    control_plane: &ControlPlane,
    session_id: SessionId,
    task_id: TaskId,
    selection: &SessionModelSelection,
) -> Result<(), ErrorEnvelope> {
    let Some(selection) = normalize_start_model_selection(selection) else {
        return Ok(());
    };

    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Task { task_id },
        session_id,
        turn_id: None,
        kind: SessionEventKind::SessionModelChanged(SessionModelChanged {
            model_id: selection.model_id,
            reasoning_effort: selection
                .reasoning_effort
                .map(session_reasoning_effort_from_client),
        }),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Internal, "Failed to persist session event.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })?;
    Ok(())
}

async fn maybe_mark_task_in_progress(
    control_plane: &ControlPlane,
    repo: RepoScope,
    task_id: TaskId,
) -> Result<(), ErrorEnvelope> {
    let updated = sqlx::query(
        r#"
        UPDATE tasks
        SET state = 'in_progress',
            updated_at_ms = ?2
        WHERE id = ?1
          AND state = 'todo'
        "#,
    )
    .bind(task_id)
    .bind(now_ms())
    .execute(control_plane.pool())
    .await
    .map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::Internal, "Failed to update task state.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?
    .rows_affected()
        > 0;

    if !updated {
        return Ok(());
    }

    #[derive(serde::Serialize)]
    struct TaskStateChangedPayload {
        task_id: String,
        state: &'static str,
    }

    let payload = serde_json::to_vec(&TaskStateChangedPayload {
        task_id: task_id.to_string(),
        state: "in_progress",
    })
    .unwrap_or_default();

    let _ = control_plane
        .event_log()
        .append_event(
            redesmyn_storage::events::EventScope::Repo {
                workspace_id: repo.workspace_id,
                repo_id: repo.repo_id,
            },
            TASK_STATE_CHANGED_EVENT,
            payload,
        )
        .await;

    Ok(())
}

async fn issue_repo_command(
    control_plane: &ControlPlane,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    kind: String,
    target_task_id: Option<TaskId>,
    json_payload: Vec<u8>,
) -> Result<CommandSummary, ErrorEnvelope> {
    control_plane
        .issue_command(
            redesmyn_storage::commands::CommandScope::Repo {
                workspace_id,
                repo_id,
            },
            kind,
            target_task_id,
            None,
            None,
            json_payload,
        )
        .await
        .map_err(|err| {
            let envelope: ErrorEnvelope = err.into();
            envelope
        })
}

async fn wait_for_start_command_terminal(
    control_plane: &ControlPlane,
    command: CommandSummary,
) -> Result<CommandSummary, ErrorEnvelope> {
    wait_for_start_command_terminal_with_timeout(control_plane, command, START_COMMAND_WAIT_TIMEOUT)
        .await
}

async fn wait_for_start_command_terminal_with_timeout(
    control_plane: &ControlPlane,
    command: CommandSummary,
    timeout: Duration,
) -> Result<CommandSummary, ErrorEnvelope> {
    if matches!(
        command.state,
        CommandState::Succeeded | CommandState::Failed | CommandState::Canceled
    ) {
        return Ok(command);
    }

    control_plane
        .commands()
        .wait_for_command(
            command.command_id,
            &[
                CommandState::Succeeded,
                CommandState::Failed,
                CommandState::Canceled,
            ],
            timeout,
        )
        .await
}

async fn load_command_summary(
    control_plane: &ControlPlane,
    command_id: CommandId,
) -> Result<CommandSummary, ErrorEnvelope> {
    control_plane
        .commands()
        .get_command(command_id)
        .await
        .map_err(|err| {
            let envelope: ErrorEnvelope = err.into();
            envelope
        })?
        .ok_or_else(|| {
            ErrorEnvelope::new(ErrorCategory::NotFound, "Command not found.").with_detail(
                ErrorDetail::from([("command_id".to_string(), command_id.to_string())]),
            )
        })
}

async fn rollback_failed_start_session(
    control_plane: &ControlPlane,
    session_id: SessionId,
    task_id: TaskId,
) -> Option<String> {
    let mut cleanup_errors = Vec::new();

    if let Err(err) = end_sessions(control_plane.pool(), &[session_id]).await {
        cleanup_errors.push(format!("failed to stop session: {}", err.message));
    }

    if let Err(err) = append_session_ended(control_plane, session_id, task_id).await {
        cleanup_errors.push(format!(
            "failed to append session_ended event: {}",
            err.message
        ));
    }

    if cleanup_errors.is_empty() {
        None
    } else {
        Some(cleanup_errors.join("; "))
    }
}

pub(super) async fn execute_start_agent(
    control_plane: &ControlPlane,
    repo: RepoScope,
    task_id: TaskId,
    agent_kind: AgentKind,
    initial_prompt: Option<String>,
    session_model_selection: Option<SessionModelSelection>,
    codex_approval_policy: Option<CodexApprovalPolicy>,
    codex_sandbox_policy: Option<CodexSandboxPolicy>,
    stop_session_ids: Vec<SessionId>,
) -> Result<StartAgentResponse, ErrorEnvelope> {
    let RepoScope {
        workspace_id,
        repo_id,
    } = repo;

    if !stop_session_ids.is_empty() {
        end_sessions(control_plane.pool(), &stop_session_ids).await?;
        for session_id in &stop_session_ids {
            let _ = append_session_ended(control_plane, *session_id, task_id).await;
        }
    }

    let session_id = insert_task_session(
        control_plane.pool(),
        workspace_id,
        repo_id,
        task_id,
        agent_kind,
    )
    .await?;

    append_session_started(control_plane, session_id, task_id).await?;
    if let Some(policy) = codex_approval_policy {
        append_codex_approval_policy_changed(control_plane, session_id, task_id, policy).await?;
    }
    if let Some(policy) = codex_sandbox_policy.as_ref() {
        append_codex_sandbox_policy_changed(control_plane, session_id, task_id, policy).await?;
    }
    if let Some(selection) = session_model_selection.as_ref() {
        append_session_model_changed(control_plane, session_id, task_id, selection).await?;
    }
    let mut initial_prompt = initial_prompt
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(prompt) = initial_prompt.as_ref() {
        match load_task_prelude_context(control_plane.pool(), task_id).await {
            Ok(context) => {
                let rendered = render_prelude_template(
                    prompt,
                    PreludeTemplateContext {
                        task_id: &context.task_id,
                        task_title: &context.task_title,
                        task_doc: &context.task_doc,
                        epic_slug: &context.epic_slug,
                        epic_readme: &context.epic_readme,
                        branch: &context.branch,
                        worktree: &context.worktree,
                    },
                );
                let rendered = rendered.trim().to_string();
                if rendered.is_empty() {
                    initial_prompt = None;
                } else {
                    initial_prompt = Some(rendered);
                }
            }
            Err(err) => {
                redesmyn_logging::tracing::warn!(
                    task_id = %task_id,
                    error = ?err,
                    "unable to resolve task prelude context; using raw prompt"
                );
            }
        }
    }
    let policy_snapshot = crate::policy_snapshot::load_session_policy_snapshot(
        control_plane.session_events(),
        session_id,
    )
    .await
    .map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to load session policy snapshot.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;
    let task_branch_name = load_task_branch_name(control_plane.pool(), task_id).await?;

    let json_payload = payloads::start_task_session(
        session_id,
        task_id,
        Some(task_branch_name),
        agent_kind,
        initial_prompt.clone(),
        Some(policy_snapshot),
        stop_session_ids.clone(),
    )?;

    let command = issue_repo_command(
        control_plane,
        workspace_id,
        repo_id,
        TASK_AGENT_START.to_string(),
        Some(task_id),
        json_payload,
    )
    .await?;
    let command = match wait_for_start_command_terminal(control_plane, command.clone()).await {
        Ok(command) => command,
        Err(wait_error) => {
            let latest = load_command_summary(control_plane, command.command_id).await?;
            if matches!(
                latest.state,
                CommandState::Succeeded | CommandState::Failed | CommandState::Canceled
            ) {
                latest
            } else {
                let mut detail = ErrorDetail::from([
                    ("command_id".to_string(), command.command_id.to_string()),
                    ("session_id".to_string(), session_id.to_string()),
                    ("task_id".to_string(), task_id.to_string()),
                    ("wait_error".to_string(), wait_error.message),
                ]);

                if let Some(cleanup_error) =
                    rollback_failed_start_session(control_plane, session_id, task_id).await
                {
                    detail.insert("cleanup_error".to_string(), cleanup_error);
                }

                let stop_after_timeout = async {
                    let stop_payload = payloads::stop_task_sessions(task_id, vec![session_id])?;
                    issue_repo_command(
                        control_plane,
                        workspace_id,
                        repo_id,
                        TASK_AGENT_STOP.to_string(),
                        Some(task_id),
                        stop_payload,
                    )
                    .await
                    .map(|_| ())
                }
                .await;
                if let Err(stop_error) = stop_after_timeout {
                    detail.insert("stop_after_timeout_error".to_string(), stop_error.message);
                }

                let timed_out_message = "Timed out waiting for agent start command to finish.";
                return Err(
                    ErrorEnvelope::new(ErrorCategory::Unavailable, timed_out_message)
                        .with_detail(detail),
                );
            }
        }
    };

    if matches!(command.state, CommandState::Failed | CommandState::Canceled) {
        let mut detail = ErrorDetail::from([
            ("command_id".to_string(), command.command_id.to_string()),
            ("session_id".to_string(), session_id.to_string()),
            ("task_id".to_string(), task_id.to_string()),
        ]);

        if let Some(cleanup_error) =
            rollback_failed_start_session(control_plane, session_id, task_id).await
        {
            detail.insert("cleanup_error".to_string(), cleanup_error);
        }

        let message = command
            .last_update
            .as_ref()
            .and_then(|update| update.message.clone())
            .unwrap_or_else(|| "Failed to start agent.".to_string());

        return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, message).with_detail(detail));
    }

    if command.state == CommandState::Succeeded {
        if let Some(prompt) = initial_prompt.as_deref() {
            append_user_message(control_plane, session_id, task_id, prompt).await?;
        }
    }

    let _ = maybe_mark_task_in_progress(control_plane, repo, task_id).await;

    Ok(StartAgentResponse {
        command,
        session_id,
    })
}

pub(super) async fn execute_stop_agent(
    control_plane: &ControlPlane,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
    session_ids: Vec<SessionId>,
) -> Result<StopAgentResponse, ErrorEnvelope> {
    end_sessions(control_plane.pool(), &session_ids).await?;
    for session_id in &session_ids {
        let _ = append_session_ended(control_plane, *session_id, task_id).await;
    }

    let json_payload = payloads::stop_task_sessions(task_id, session_ids.clone())?;
    let command = issue_repo_command(
        control_plane,
        workspace_id,
        repo_id,
        TASK_AGENT_STOP.to_string(),
        Some(task_id),
        json_payload,
    )
    .await?;

    Ok(StopAgentResponse {
        command,
        ended_session_ids: session_ids,
    })
}

pub(super) async fn execute_attach_agent_session(
    control_plane: &ControlPlane,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    session_id: SessionId,
) -> Result<AttachAgentSessionResponse, ErrorEnvelope> {
    let json_payload = payloads::attach_session(session_id)?;
    let command = issue_repo_command(
        control_plane,
        workspace_id,
        repo_id,
        SESSION_AGENT_ATTACH_SESSION.to_string(),
        None,
        json_payload,
    )
    .await?;

    Ok(AttachAgentSessionResponse { command })
}

pub(super) async fn execute_structured_resume(
    control_plane: &ControlPlane,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
    message: &str,
    plan: StructuredResumePlan,
) -> Result<SendTaskAgentMessageResponse, ErrorEnvelope> {
    append_user_message(control_plane, plan.session_id, task_id, message).await?;

    let policy_snapshot = crate::policy_snapshot::load_session_policy_snapshot(
        control_plane.session_events(),
        plan.session_id,
    )
    .await
    .map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to load session policy snapshot.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    let json_payload = payloads::resume_by_id_turn(
        plan.session_id,
        Some(task_id),
        message.to_string(),
        plan.external_session_ref,
        Some(policy_snapshot),
        plan.interrupt_turn,
    )?;

    let command = issue_repo_command(
        control_plane,
        workspace_id,
        repo_id,
        SESSION_AGENT_RESUME_BY_ID_TURN.to_string(),
        Some(task_id),
        json_payload,
    )
    .await?;

    Ok(SendTaskAgentMessageResponse {
        command,
        session_id: plan.session_id,
        delivery: TaskAgentMessageDelivery::StructuredResumed,
        conversation_continuity: TaskAgentMessageConversationContinuity::Kept,
        warnings: Vec::new(),
    })
}

pub(super) async fn execute_new_session_send(
    control_plane: &ControlPlane,
    repo: RepoScope,
    task_id: TaskId,
    agent_kind: AgentKind,
    message: &str,
    plan: NewSessionPlan,
) -> Result<SendTaskAgentMessageResponse, ErrorEnvelope> {
    let RepoScope {
        workspace_id,
        repo_id,
    } = repo;

    if !plan.stop_session_ids.is_empty() {
        end_sessions(control_plane.pool(), &plan.stop_session_ids).await?;
        for session_id in &plan.stop_session_ids {
            let _ = append_session_ended(control_plane, *session_id, task_id).await;
        }
    }

    let session_id = insert_task_session(
        control_plane.pool(),
        workspace_id,
        repo_id,
        task_id,
        agent_kind,
    )
    .await?;

    append_session_started(control_plane, session_id, task_id).await?;
    append_user_message(control_plane, session_id, task_id, message).await?;

    let policy_snapshot = crate::policy_snapshot::load_session_policy_snapshot(
        control_plane.session_events(),
        session_id,
    )
    .await
    .map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to load session policy snapshot.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;
    let task_branch_name = load_task_branch_name(control_plane.pool(), task_id).await?;

    let json_payload = payloads::start_task_session(
        session_id,
        task_id,
        Some(task_branch_name),
        agent_kind,
        Some(message.to_string()),
        Some(policy_snapshot),
        plan.stop_session_ids.clone(),
    )?;

    let command = issue_repo_command(
        control_plane,
        workspace_id,
        repo_id,
        TASK_AGENT_START.to_string(),
        Some(task_id),
        json_payload,
    )
    .await?;

    Ok(SendTaskAgentMessageResponse {
        command,
        session_id,
        delivery: plan.delivery,
        conversation_continuity: plan.conversation_continuity,
        warnings: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::wait_for_start_command_terminal_with_timeout;
    use std::time::Duration;

    use crate::ControlPlane;
    use redesmyn_protocol::client::CommandState;
    use redesmyn_storage::commands::CommandScope;
    use redesmyn_storage::schema::CommandState as StorageCommandState;

    #[tokio::test]
    async fn wait_for_start_command_terminal_timeout_then_later_succeeds() {
        let control_plane = ControlPlane::open_test().await.expect("control plane");
        let created = control_plane
            .commands()
            .create_command(
                CommandScope::None,
                "task.agent.start".to_string(),
                None,
                None,
                None,
                Vec::new(),
            )
            .await
            .expect("create command");

        let timed_out = wait_for_start_command_terminal_with_timeout(
            &control_plane,
            created.command.clone(),
            Duration::from_millis(10),
        )
        .await
        .expect_err("expected timeout while command remains queued");
        assert_eq!(
            timed_out.category,
            redesmyn_protocol::ErrorCategory::Unavailable
        );

        control_plane
            .commands()
            .append_update(
                created.command.command_id,
                StorageCommandState::Succeeded,
                Some("dispatched".to_string()),
                None,
                None,
                None,
            )
            .await
            .expect("append succeeded update");

        let summary = wait_for_start_command_terminal_with_timeout(
            &control_plane,
            created.command,
            Duration::from_millis(10),
        )
        .await
        .expect("wait should return terminal command");
        assert_eq!(summary.state, CommandState::Succeeded);
    }

    #[tokio::test]
    async fn wait_for_start_command_terminal_timeout_then_later_fails() {
        let control_plane = ControlPlane::open_test().await.expect("control plane");
        let created = control_plane
            .commands()
            .create_command(
                CommandScope::None,
                "task.agent.start".to_string(),
                None,
                None,
                None,
                Vec::new(),
            )
            .await
            .expect("create command");

        let timed_out = wait_for_start_command_terminal_with_timeout(
            &control_plane,
            created.command.clone(),
            Duration::from_millis(10),
        )
        .await
        .expect_err("expected timeout while command remains queued");
        assert_eq!(
            timed_out.category,
            redesmyn_protocol::ErrorCategory::Unavailable
        );

        control_plane
            .commands()
            .append_update(
                created.command.command_id,
                StorageCommandState::Failed,
                Some("failed later".to_string()),
                None,
                None,
                None,
            )
            .await
            .expect("append failed update");

        let summary = wait_for_start_command_terminal_with_timeout(
            &control_plane,
            created.command,
            Duration::from_millis(10),
        )
        .await
        .expect("wait should return terminal command");
        assert_eq!(summary.state, CommandState::Failed);
    }
}
