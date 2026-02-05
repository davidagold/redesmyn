use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_ids::{ArtifactId, EpicId, RepoId, SessionEventId, SessionId, TaskId, WorkspaceId};
use redesmyn_logging::tracing::{Instrument, debug_span};
use sqlx::{Executor, QueryBuilder, Sqlite};

use crate::StorageError;
use crate::schema::{AgentKind, AgentSessionScopeKind, AgentSessionStatus};

const MAX_SESSION_EVENT_PAYLOAD_BYTES: usize = 1024 * 1024;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

type AgentSessionRow = (
    SessionId,
    i64,
    i64,
    WorkspaceId,
    RepoId,
    String,
    Option<TaskId>,
    String,
    String,
    String,
    Option<String>,
    Option<i64>,
    Option<i64>,
    Option<i64>,
);

fn decode_agent_session_row(
    (
        session_id,
        created_at_ms,
        updated_at_ms,
        scope_workspace_id,
        scope_repo_id,
        scope_kind,
        task_id,
        agent_kind,
        status,
        external_session_ref,
        title,
        started_at_ms,
        ended_at_ms,
        closed_at_ms,
    ): AgentSessionRow,
) -> Result<AgentSessionRecord, StorageError> {
    let scope_kind = match scope_kind.as_str() {
        "task" => AgentSessionScopeKind::Task,
        "chat" => AgentSessionScopeKind::Chat,
        _ => {
            return Err(StorageError::InvalidData {
                message: format!("unknown agent_sessions.scope_kind={scope_kind}"),
            });
        }
    };

    let agent_kind = match agent_kind.as_str() {
        "codex" => AgentKind::Codex,
        "claude_code" => AgentKind::ClaudeCode,
        "shell" => AgentKind::Shell,
        _ => {
            return Err(StorageError::InvalidData {
                message: format!("unknown agent_sessions.agent_kind={agent_kind}"),
            });
        }
    };

    let status = match status.as_str() {
        "running" => AgentSessionStatus::Running,
        "blocked" => AgentSessionStatus::Blocked,
        "stopped" => AgentSessionStatus::Stopped,
        "error" => AgentSessionStatus::Error,
        _ => {
            return Err(StorageError::InvalidData {
                message: format!("unknown agent_sessions.status={status}"),
            });
        }
    };

    Ok(AgentSessionRecord {
        session_id,
        created_at_ms,
        updated_at_ms,
        scope_workspace_id,
        scope_repo_id,
        scope_kind,
        task_id,
        agent_kind,
        status,
        external_session_ref,
        title,
        started_at_ms,
        ended_at_ms,
        closed_at_ms,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSessionRecord {
    pub session_id: SessionId,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub scope_workspace_id: WorkspaceId,
    pub scope_repo_id: RepoId,
    pub scope_kind: AgentSessionScopeKind,
    pub task_id: Option<TaskId>,
    pub agent_kind: AgentKind,
    pub status: AgentSessionStatus,
    pub external_session_ref: String,
    pub title: Option<String>,
    pub started_at_ms: Option<i64>,
    pub ended_at_ms: Option<i64>,
    pub closed_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEventRecord {
    pub id: SessionEventId,
    pub session_id: SessionId,
    pub created_at_ms: i64,
    pub kind: String,
    pub turn_id: Option<String>,
    pub message_preview: Option<String>,
    pub artifact_id: Option<ArtifactId>,
    pub payload: Vec<u8>,
}

type SessionEventRow = (
    SessionEventId,
    SessionId,
    i64,
    String,
    Option<String>,
    Option<String>,
    Option<ArtifactId>,
    Vec<u8>,
);

fn decode_session_event_row(
    (id, session_id, created_at_ms, kind, turn_id, message_preview, artifact_id, payload): SessionEventRow,
) -> SessionEventRecord {
    SessionEventRecord {
        id,
        session_id,
        created_at_ms,
        kind,
        turn_id,
        message_preview,
        artifact_id,
        payload,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionEventCursor {
    pub created_at_ms: i64,
    pub id: SessionEventId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEventsQuery {
    pub limit: u32,
    pub after: Option<SessionEventCursor>,
    pub kinds: Vec<String>,
}

impl SessionEventsQuery {
    #[must_use]
    pub fn with_limit(limit: u32) -> Self {
        Self {
            limit,
            after: None,
            kinds: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewSessionEvent {
    pub kind: String,
    pub turn_id: Option<String>,
    pub message_preview: Option<String>,
    pub artifact_id: Option<ArtifactId>,
    pub payload: Vec<u8>,
    pub created_at_ms: Option<i64>,
}

pub async fn insert_agent_session<'e, E>(
    executor: E,
    session: &AgentSessionRecord,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO agent_sessions (
            session_id,
            created_at_ms,
            updated_at_ms,
            scope_workspace_id,
            scope_repo_id,
            scope_kind,
            task_id,
            agent_kind,
            status,
            external_session_ref,
            title,
            started_at_ms,
            ended_at_ms,
            closed_at_ms
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
        "#,
    )
    .bind(session.session_id)
    .bind(session.created_at_ms)
    .bind(session.updated_at_ms)
    .bind(session.scope_workspace_id)
    .bind(session.scope_repo_id)
    .bind(session.scope_kind.as_str())
    .bind(session.task_id)
    .bind(session.agent_kind.as_str())
    .bind(session.status.as_str())
    .bind(&session.external_session_ref)
    .bind(&session.title)
    .bind(session.started_at_ms)
    .bind(session.ended_at_ms)
    .bind(session.closed_at_ms)
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn get_agent_session<'e, E>(
    executor: E,
    session_id: SessionId,
) -> Result<Option<AgentSessionRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<AgentSessionRow> = sqlx::query_as(
        r#"
        SELECT
            session_id,
            created_at_ms,
            updated_at_ms,
            scope_workspace_id,
            scope_repo_id,
            scope_kind,
            task_id,
            agent_kind,
            status,
            external_session_ref,
            title,
            started_at_ms,
            ended_at_ms,
            closed_at_ms
        FROM agent_sessions
        WHERE session_id = ?1
        "#,
    )
    .bind(session_id)
    .fetch_optional(executor)
    .await?;

    row.map(decode_agent_session_row).transpose()
}

pub async fn create_chat_session<'e, E>(
    executor: E,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    agent_kind: AgentKind,
    title: Option<&str>,
) -> Result<SessionId, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let session_id = SessionId::new();
    let now_ms = now_ms();

    async {
        let session = AgentSessionRecord {
            session_id,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: AgentSessionScopeKind::Chat,
            task_id: None,
            agent_kind,
            status: AgentSessionStatus::Stopped,
            external_session_ref: r#"{"type":"none"}"#.to_owned(),
            title: title.map(ToOwned::to_owned),
            started_at_ms: None,
            ended_at_ms: None,
            closed_at_ms: None,
        };

        insert_agent_session(executor, &session).await?;
        Ok(session_id)
    }
    .instrument(debug_span!("storage.create_chat_session", session_id = %session_id))
    .await
}

pub async fn create_task_session<'e, E>(
    executor: E,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
    agent_kind: AgentKind,
    title: Option<&str>,
) -> Result<SessionId, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let session_id = SessionId::new();
    let now_ms = now_ms();

    async {
        let session = AgentSessionRecord {
            session_id,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: AgentSessionScopeKind::Task,
            task_id: Some(task_id),
            agent_kind,
            status: AgentSessionStatus::Stopped,
            external_session_ref: r#"{"type":"none"}"#.to_owned(),
            title: title.map(ToOwned::to_owned),
            started_at_ms: None,
            ended_at_ms: None,
            closed_at_ms: None,
        };

        insert_agent_session(executor, &session).await?;
        Ok(session_id)
    }
    .instrument(debug_span!(
        "storage.create_task_session",
        session_id = %session_id,
        task_id = %task_id
    ))
    .await
}

pub async fn list_task_sessions<'e, E>(
    executor: E,
    task_id: TaskId,
    limit: u32,
) -> Result<Vec<AgentSessionRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let rows: Vec<AgentSessionRow> = sqlx::query_as(
        r#"
        SELECT
            session_id,
            created_at_ms,
            updated_at_ms,
            scope_workspace_id,
            scope_repo_id,
            scope_kind,
            task_id,
            agent_kind,
            status,
            external_session_ref,
            title,
            started_at_ms,
            ended_at_ms,
            closed_at_ms
        FROM agent_sessions
        WHERE scope_kind = 'task' AND task_id = ?1
        ORDER BY created_at_ms DESC, session_id DESC
        LIMIT ?2
        "#,
    )
    .bind(task_id)
    .bind(i64::from(limit))
    .fetch_all(executor)
    .await?;

    rows.into_iter().map(decode_agent_session_row).collect()
}

pub async fn list_chat_sessions<'e, E>(
    executor: E,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    include_closed: bool,
    limit: u32,
) -> Result<Vec<AgentSessionRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let rows: Vec<AgentSessionRow> = if include_closed {
        sqlx::query_as(
            r#"
            SELECT
                session_id,
                created_at_ms,
                updated_at_ms,
                scope_workspace_id,
                scope_repo_id,
                scope_kind,
                task_id,
                agent_kind,
                status,
                external_session_ref,
                title,
                started_at_ms,
                ended_at_ms,
                closed_at_ms
            FROM agent_sessions
            WHERE
                scope_kind = 'chat'
                AND scope_workspace_id = ?1
                AND scope_repo_id = ?2
            ORDER BY created_at_ms DESC, session_id DESC
            LIMIT ?3
            "#,
        )
        .bind(workspace_id)
        .bind(repo_id)
        .bind(i64::from(limit))
        .fetch_all(executor)
        .await?
    } else {
        sqlx::query_as(
            r#"
            SELECT
                session_id,
                created_at_ms,
                updated_at_ms,
                scope_workspace_id,
                scope_repo_id,
                scope_kind,
                task_id,
                agent_kind,
                status,
                external_session_ref,
                title,
                started_at_ms,
                ended_at_ms,
                closed_at_ms
            FROM agent_sessions
            WHERE
                scope_kind = 'chat'
                AND scope_workspace_id = ?1
                AND scope_repo_id = ?2
                AND closed_at_ms IS NULL
            ORDER BY created_at_ms DESC, session_id DESC
            LIMIT ?3
            "#,
        )
        .bind(workspace_id)
        .bind(repo_id)
        .bind(i64::from(limit))
        .fetch_all(executor)
        .await?
    };

    rows.into_iter().map(decode_agent_session_row).collect()
}

pub async fn close_chat_session<'e, E>(
    executor: E,
    session_id: SessionId,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let now_ms = now_ms();
    async {
        let result = sqlx::query(
            r#"
            UPDATE agent_sessions
            SET
                updated_at_ms = ?1,
                status = 'stopped',
                closed_at_ms = COALESCE(closed_at_ms, ?1),
                ended_at_ms = COALESCE(ended_at_ms, ?1)
            WHERE session_id = ?2 AND scope_kind = 'chat'
            "#,
        )
        .bind(now_ms)
        .bind(session_id)
        .execute(executor)
        .await?;

        if result.rows_affected() == 0 {
            return Err(StorageError::InvalidData {
                message: format!("chat session not found: {session_id}"),
            });
        }

        Ok(())
    }
    .instrument(debug_span!("storage.close_chat_session", session_id = %session_id))
    .await
}

pub async fn update_agent_session_status<'e, E>(
    executor: E,
    session_id: SessionId,
    status: AgentSessionStatus,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let now_ms = now_ms();
    async {
        sqlx::query(
            r#"
            UPDATE agent_sessions
            SET updated_at_ms = ?1, status = ?2
            WHERE session_id = ?3
            "#,
        )
        .bind(now_ms)
        .bind(status.as_str())
        .bind(session_id)
        .execute(executor)
        .await?;
        Ok(())
    }
    .instrument(debug_span!(
        "storage.update_agent_session_status",
        session_id = %session_id,
        status = %status
    ))
    .await
}

pub async fn end_task_sessions<'e, E>(executor: E, task_id: TaskId) -> Result<u64, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let now_ms = now_ms();
    async {
        let result = sqlx::query(
            r#"
            UPDATE agent_sessions
            SET
                updated_at_ms = ?1,
                status = 'stopped',
                ended_at_ms = COALESCE(ended_at_ms, ?1)
            WHERE scope_kind = 'task' AND task_id = ?2 AND ended_at_ms IS NULL
            "#,
        )
        .bind(now_ms)
        .bind(task_id)
        .execute(executor)
        .await?;
        Ok(result.rows_affected())
    }
    .instrument(debug_span!("storage.end_task_sessions", task_id = %task_id))
    .await
}

pub async fn pin_chat_session_to_epic<'e, E>(
    executor: E,
    epic_id: EpicId,
    session_id: SessionId,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO session_pins (epic_id, session_id)
        VALUES (?1, ?2)
        ON CONFLICT(epic_id) DO UPDATE SET session_id = excluded.session_id
        "#,
    )
    .bind(epic_id)
    .bind(session_id)
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn unpin_chat_session_from_epic<'e, E>(
    executor: E,
    epic_id: EpicId,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query("DELETE FROM session_pins WHERE epic_id = ?1")
        .bind(epic_id)
        .execute(executor)
        .await?;
    Ok(())
}

pub async fn get_pinned_chat_session_for_epic<'e, E>(
    executor: E,
    epic_id: EpicId,
) -> Result<Option<SessionId>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(SessionId,)> =
        sqlx::query_as("SELECT session_id FROM session_pins WHERE epic_id = ?1")
            .bind(epic_id)
            .fetch_optional(executor)
            .await?;
    Ok(row.map(|(session_id,)| session_id))
}

pub async fn append_session_event<'e, E>(
    executor: E,
    session_id: SessionId,
    event: &NewSessionEvent,
) -> Result<SessionEventRecord, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    if event.payload.len() > MAX_SESSION_EVENT_PAYLOAD_BYTES {
        return Err(StorageError::InvalidData {
            message: format!(
                "session event payload too large: {} bytes (max {})",
                event.payload.len(),
                MAX_SESSION_EVENT_PAYLOAD_BYTES
            ),
        });
    }

    let id = SessionEventId::new();
    let created_at_ms = event.created_at_ms.unwrap_or_else(now_ms);

    async {
        let (workspace_id, repo_id, scope_kind, task_id): (
            WorkspaceId,
            RepoId,
            String,
            Option<TaskId>,
        ) = sqlx::query_as(
            r#"
                SELECT
                    scope_workspace_id,
                    scope_repo_id,
                    scope_kind,
                    task_id
                FROM agent_sessions
                WHERE session_id = ?1
                "#,
        )
        .bind(session_id)
        .fetch_optional(executor)
        .await?
        .ok_or_else(|| StorageError::InvalidData {
            message: format!("session not found: {session_id}"),
        })?;

        let (scope_kind, epic_id, task_id) = match scope_kind.as_str() {
            "chat" => ("repo", None::<EpicId>, None::<TaskId>),
            "task" => {
                let task_id = task_id.ok_or_else(|| StorageError::InvalidData {
                    message: format!("task session missing task_id: {session_id}"),
                })?;

                let (epic_id,): (EpicId,) =
                    sqlx::query_as("SELECT epic_id FROM tasks WHERE id = ?1")
                        .bind(task_id)
                        .fetch_one(executor)
                        .await?;

                ("task", Some(epic_id), Some(task_id))
            }
            _ => {
                return Err(StorageError::InvalidData {
                    message: format!("unknown agent_sessions.scope_kind={scope_kind}"),
                });
            }
        };

        sqlx::query(
            r#"
            INSERT INTO session_events (
                id,
                session_id,
                created_at_ms,
                scope_kind,
                scope_workspace_id,
                scope_repo_id,
                epic_id,
                task_id,
                kind,
                turn_id,
                message_preview,
                artifact_id,
                payload
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            "#,
        )
        .bind(id)
        .bind(session_id)
        .bind(created_at_ms)
        .bind(scope_kind)
        .bind(workspace_id)
        .bind(repo_id)
        .bind(epic_id)
        .bind(task_id)
        .bind(&event.kind)
        .bind(&event.turn_id)
        .bind(&event.message_preview)
        .bind(event.artifact_id)
        .bind(&event.payload)
        .execute(executor)
        .await?;

        Ok(SessionEventRecord {
            id,
            session_id,
            created_at_ms,
            kind: event.kind.clone(),
            turn_id: event.turn_id.clone(),
            message_preview: event.message_preview.clone(),
            artifact_id: event.artifact_id,
            payload: event.payload.clone(),
        })
    }
    .instrument(debug_span!(
        "storage.append_session_event",
        session_id = %session_id,
        session_event_id = %id,
        kind = %event.kind
    ))
    .await
}

pub async fn get_session_events<'e, E>(
    executor: E,
    session_id: SessionId,
    query: &SessionEventsQuery,
) -> Result<Vec<SessionEventRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
        r#"
        SELECT
            id,
            session_id,
            created_at_ms,
            kind,
            turn_id,
            message_preview,
            artifact_id,
            payload
        FROM session_events
        WHERE session_id =
        "#,
    );
    builder.push_bind(session_id);

    if let Some(after) = query.after {
        builder.push(" AND (created_at_ms > ");
        builder.push_bind(after.created_at_ms);
        builder.push(" OR (created_at_ms = ");
        builder.push_bind(after.created_at_ms);
        builder.push(" AND id > ");
        builder.push_bind(after.id);
        builder.push("))");
    }

    if !query.kinds.is_empty() {
        builder.push(" AND kind IN (");
        {
            let mut separated = builder.separated(", ");
            for kind in &query.kinds {
                separated.push_bind(kind);
            }
        }
        builder.push(")");
    }

    builder.push(" ORDER BY created_at_ms ASC, id ASC LIMIT ");
    builder.push_bind(i64::from(query.limit));

    let rows: Vec<SessionEventRow> = builder.build_query_as().fetch_all(executor).await?;
    Ok(rows.into_iter().map(decode_session_event_row).collect())
}

pub async fn get_session_event_cursor<'e, E>(
    executor: E,
    session_id: SessionId,
    after_event_id: SessionEventId,
) -> Result<Option<SessionEventCursor>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(i64,)> = sqlx::query_as(
        r#"
        SELECT created_at_ms
        FROM session_events
        WHERE session_id = ?1 AND id = ?2
        "#,
    )
    .bind(session_id)
    .bind(after_event_id)
    .fetch_optional(executor)
    .await?;

    Ok(row.map(|(created_at_ms,)| SessionEventCursor {
        created_at_ms,
        id: after_event_id,
    }))
}

pub async fn get_task_session_event_cursor<'e, E>(
    executor: E,
    task_id: TaskId,
    after_event_id: SessionEventId,
) -> Result<Option<SessionEventCursor>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(i64,)> = sqlx::query_as(
        r#"
        SELECT created_at_ms
        FROM session_events
        WHERE task_id = ?1 AND id = ?2
        "#,
    )
    .bind(task_id)
    .bind(after_event_id)
    .fetch_optional(executor)
    .await?;

    Ok(row.map(|(created_at_ms,)| SessionEventCursor {
        created_at_ms,
        id: after_event_id,
    }))
}

pub async fn get_task_session_events<'e, E>(
    executor: E,
    task_id: TaskId,
    query: &SessionEventsQuery,
) -> Result<Vec<SessionEventRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let mut builder: QueryBuilder<Sqlite> = QueryBuilder::new(
        r#"
        SELECT
            id,
            session_id,
            created_at_ms,
            kind,
            turn_id,
            message_preview,
            artifact_id,
            payload
        FROM session_events
        WHERE task_id =
        "#,
    );
    builder.push_bind(task_id);

    if let Some(after) = query.after {
        builder.push(" AND (created_at_ms > ");
        builder.push_bind(after.created_at_ms);
        builder.push(" OR (created_at_ms = ");
        builder.push_bind(after.created_at_ms);
        builder.push(" AND id > ");
        builder.push_bind(after.id);
        builder.push("))");
    }

    if !query.kinds.is_empty() {
        builder.push(" AND kind IN (");
        {
            let mut separated = builder.separated(", ");
            for kind in &query.kinds {
                separated.push_bind(kind);
            }
        }
        builder.push(")");
    }

    builder.push(" ORDER BY created_at_ms ASC, id ASC LIMIT ");
    builder.push_bind(i64::from(query.limit));

    let rows: Vec<SessionEventRow> = builder.build_query_as().fetch_all(executor).await?;
    Ok(rows.into_iter().map(decode_session_event_row).collect())
}

pub async fn task_exists_in_repo<'e, E>(
    executor: E,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
) -> Result<bool, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(i64,)> = sqlx::query_as(
        r#"
        SELECT 1
        FROM tasks
        JOIN epics ON tasks.epic_id = epics.id
        JOIN repositories ON epics.repo_id = repositories.id
        WHERE
            tasks.id = ?1
            AND repositories.workspace_id = ?2
            AND repositories.id = ?3
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .bind(workspace_id)
    .bind(repo_id)
    .fetch_optional(executor)
    .await?;

    Ok(row.is_some())
}

pub async fn epic_exists_in_repo<'e, E>(
    executor: E,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    epic_id: EpicId,
) -> Result<bool, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(i64,)> = sqlx::query_as(
        r#"
        SELECT 1
        FROM epics
        JOIN repositories ON epics.repo_id = repositories.id
        WHERE
            epics.id = ?1
            AND repositories.workspace_id = ?2
            AND repositories.id = ?3
        LIMIT 1
        "#,
    )
    .bind(epic_id)
    .bind(workspace_id)
    .bind(repo_id)
    .fetch_optional(executor)
    .await?;

    Ok(row.is_some())
}
