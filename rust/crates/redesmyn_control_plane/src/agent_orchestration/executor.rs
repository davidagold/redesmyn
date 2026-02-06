use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_ids::{RepoId, SessionEventId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::agent_commands::{
    SESSION_AGENT_ATTACH_SESSION, SESSION_AGENT_RESUME_BY_ID_TURN, TASK_AGENT_START,
    TASK_AGENT_STOP,
};
use redesmyn_protocol::client::{
    AgentKind, AttachAgentSessionResponse, CommandState, CommandSummary,
    SendTaskAgentMessageResponse, StartAgentResponse, StopAgentResponse,
    TaskAgentMessageConversationContinuity, TaskAgentMessageDelivery,
};
use redesmyn_protocol::session::{SessionEnded, SessionEventKind, SessionScope, SessionStarted};
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
    use redesmyn_storage::sessions::{insert_agent_session, AgentSessionRecord};

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
    if let Some(prompt) = initial_prompt.as_deref() {
        let trimmed = prompt.trim();
        if !trimmed.is_empty() {
            append_user_message(control_plane, session_id, task_id, trimmed).await?;
        }
    }

    let initial_prompt = initial_prompt
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

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

    let json_payload = payloads::start_task_session(
        session_id,
        task_id,
        agent_kind,
        initial_prompt,
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

    if command.state == CommandState::Failed {
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

    let json_payload = payloads::start_task_session(
        session_id,
        task_id,
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
