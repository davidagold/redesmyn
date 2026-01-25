use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::client::{AgentInterfaceMode, AgentKind};
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope, ExternalSessionRef};
use sqlx::SqlitePool;

type StorageAgentInterfaceMode = redesmyn_storage::schema::AgentInterfaceMode;
type StorageAgentKind = redesmyn_storage::schema::AgentKind;
type StorageAgentSessionStatus = redesmyn_storage::schema::AgentSessionStatus;

pub(super) type StorageAgentSessionRecord = redesmyn_storage::sessions::AgentSessionRecord;

pub(super) fn protocol_agent_kind_from_storage(kind: StorageAgentKind) -> AgentKind {
    match kind {
        StorageAgentKind::Codex => AgentKind::Codex,
        StorageAgentKind::ClaudeCode => AgentKind::ClaudeCode,
        StorageAgentKind::Shell => AgentKind::Shell,
    }
}

pub(super) fn protocol_interface_mode_from_storage(
    mode: StorageAgentInterfaceMode,
) -> AgentInterfaceMode {
    match mode {
        StorageAgentInterfaceMode::ShellTmux => AgentInterfaceMode::ShellTmux,
        StorageAgentInterfaceMode::StructuredExec => AgentInterfaceMode::StructuredExec,
        StorageAgentInterfaceMode::AppServer => AgentInterfaceMode::AppServer,
    }
}

pub(super) fn is_active_task_session(row: &StorageAgentSessionRecord) -> bool {
    row.ended_at_ms.is_none()
        && matches!(
            row.status,
            StorageAgentSessionStatus::Running | StorageAgentSessionStatus::Blocked
        )
}

fn parse_external_session_ref(raw: &str) -> Option<ExternalSessionRef> {
    serde_json::from_str(raw).ok()
}

fn external_ref_matches_agent_kind(
    external_session_ref: &ExternalSessionRef,
    agent_kind: AgentKind,
) -> bool {
    matches!(
        (agent_kind, external_session_ref),
        (AgentKind::Codex, ExternalSessionRef::CodexThread { .. })
            | (
                AgentKind::ClaudeCode,
                ExternalSessionRef::ClaudeSession { .. }
            )
    )
}

pub(super) struct ResumableStructuredSession {
    pub session_id: SessionId,
    pub external_session_ref: ExternalSessionRef,
    pub turn_in_progress: bool,
}

pub(super) async fn load_active_task_session_ids(
    pool: &SqlitePool,
    task_id: TaskId,
) -> Result<Vec<SessionId>, ErrorEnvelope> {
    sqlx::query_scalar(
        r#"
        SELECT session_id
        FROM agent_sessions
        WHERE scope_kind = 'task' AND task_id = ?1 AND ended_at_ms IS NULL
        "#,
    )
    .bind(task_id)
    .fetch_all(pool)
    .await
    .map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::Internal, "Failed to query active sessions.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })
}

pub(super) async fn load_recent_task_sessions(
    pool: &SqlitePool,
    task_id: TaskId,
    limit: u32,
) -> Result<Vec<StorageAgentSessionRecord>, ErrorEnvelope> {
    redesmyn_storage::sessions::list_task_sessions(pool, task_id, limit)
        .await
        .map_err(|err| {
            ErrorEnvelope::new(ErrorCategory::Internal, "Failed to query sessions.")
                .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
        })
}

pub(super) async fn load_resumable_structured_session(
    pool: &SqlitePool,
    recent_sessions: &[StorageAgentSessionRecord],
    agent_kind: AgentKind,
    desired_interface_mode: AgentInterfaceMode,
) -> Result<Option<ResumableStructuredSession>, ErrorEnvelope> {
    if desired_interface_mode != AgentInterfaceMode::StructuredExec {
        return Ok(None);
    }

    for row in recent_sessions.iter().filter(|row| {
        is_active_task_session(row)
            && protocol_agent_kind_from_storage(row.agent_kind) == agent_kind
            && protocol_interface_mode_from_storage(row.interface_mode) == desired_interface_mode
    }) {
        let Some(parsed) = parse_external_session_ref(&row.external_session_ref) else {
            continue;
        };
        if !external_ref_matches_agent_kind(&parsed, agent_kind) {
            continue;
        }

        let turn_in_progress = crate::turn_state::structured_turn_in_progress(pool, row.session_id)
            .await?;
        return Ok(Some(ResumableStructuredSession {
            session_id: row.session_id,
            external_session_ref: parsed,
            turn_in_progress,
        }));
    }

    Ok(None)
}
