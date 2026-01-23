//! Epic graph query projection (T-21).
//!
//! This module defines a compact, query-oriented read model for the "epic graph"
//! view used by UI and CLI clients.

use std::collections::{HashMap, HashSet};

use redesmyn_ids::{
    CommandId, CommandUpdateId, EpicId, EventId, HostId, HostInstanceId, RepoId, SessionEventId,
    SessionId, TaskId, WorkspaceId,
};
use sqlx::{Executor, Sqlite};

use crate::StorageError;
use crate::schema::{CommandState, MergeReadiness, TaskState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepoScope {
    pub workspace_id: WorkspaceId,
    pub repo_id: RepoId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpicRecord {
    pub epic_id: EpicId,
    pub slug: String,
    pub title: String,
    pub scope: RepoScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRecord {
    pub task_id: TaskId,
    pub parent_task_id: Option<TaskId>,
    pub local_ref: Option<String>,
    pub title: String,
    pub state: TaskState,
    pub branch_name: Option<String>,
    pub merge_readiness: MergeReadiness,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandRecord {
    pub command_id: CommandId,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub kind: String,
    pub state: CommandState,
    pub target_task_id: Option<TaskId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandUpdateRecord {
    pub update_id: CommandUpdateId,
    pub command_id: CommandId,
    pub created_at_ms: i64,
    pub state: CommandState,
    pub message: Option<String>,
    pub progress_current: Option<i64>,
    pub progress_total: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonPresenceRecord {
    pub host_instance_id: HostInstanceId,
    pub host_id: HostId,
    pub hostname: Option<String>,
    pub connected_at_ms: i64,
    pub last_heartbeat_at_ms: i64,
    pub disconnected_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummaryRecord {
    pub session_id: SessionId,
    pub session_event_id: SessionEventId,
    pub task_id: TaskId,
    pub created_at_ms: i64,
    pub kind: String,
    pub turn_id: Option<String>,
    pub message_preview: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpicGraphData {
    pub epic: EpicRecord,
    pub tasks: Vec<TaskRecord>,
    pub commands: Vec<CommandRecord>,
    pub command_last_updates: HashMap<CommandId, CommandUpdateRecord>,
    pub daemon_presences: Vec<DaemonPresenceRecord>,
    pub session_summaries: Vec<SessionSummaryRecord>,
    pub as_of_event_id: Option<EventId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpicListItem {
    pub slug: String,
    pub title: String,
}

pub async fn list_epics<'e, E>(
    executor: E,
    scope: Option<RepoScope>,
) -> Result<Vec<EpicListItem>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    let rows: Vec<(String, String)> = match scope {
        Some(scope) => {
            sqlx::query_as(
                r#"
                SELECT e.slug, e.title
                FROM epics e
                JOIN repositories r ON r.id = e.repo_id
                WHERE r.id = ?1 AND r.workspace_id = ?2
                ORDER BY e.slug
                "#,
            )
            .bind(scope.repo_id)
            .bind(scope.workspace_id)
            .fetch_all(executor)
            .await?
        }
        None => {
            sqlx::query_as(
                r#"
                SELECT slug, title
                FROM epics
                ORDER BY slug
                "#,
            )
            .fetch_all(executor)
            .await?
        }
    };

    Ok(rows
        .into_iter()
        .map(|(slug, title)| EpicListItem { slug, title })
        .collect())
}

pub async fn load_epic_graph<'e, E>(
    executor: E,
    epic_slug: &str,
    scope: Option<RepoScope>,
) -> Result<Option<EpicGraphData>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    let epic = match load_epic(executor, epic_slug, scope).await? {
        Some(epic) => epic,
        None => return Ok(None),
    };

    let tasks = load_tasks(executor, epic.epic_id).await?;
    let task_ids: HashSet<TaskId> = tasks.iter().map(|task| task.task_id).collect();

    let mut commands = load_commands(executor, epic.scope).await?;
    commands.retain(|command| {
        command
            .target_task_id
            .map(|task_id| task_ids.contains(&task_id))
            .unwrap_or(true)
    });

    let command_last_updates = load_command_last_updates(executor, &commands).await?;
    let daemon_presences = load_daemon_presences(executor).await?;
    let session_summaries = load_session_summaries(executor, epic.epic_id).await?;
    let as_of_event_id = load_as_of_event_id(executor, epic.scope).await?;

    Ok(Some(EpicGraphData {
        epic,
        tasks,
        commands,
        command_last_updates,
        daemon_presences,
        session_summaries,
        as_of_event_id,
    }))
}

async fn load_epic<'e, E>(
    executor: E,
    epic_slug: &str,
    scope: Option<RepoScope>,
) -> Result<Option<EpicRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    let rows: Vec<(EpicId, String, String, RepoId, WorkspaceId)> = match scope {
        Some(scope) => {
            sqlx::query_as(
                r#"
                SELECT e.id, e.slug, e.title, r.id, r.workspace_id
                FROM epics e
                JOIN repositories r ON r.id = e.repo_id
                WHERE e.slug = ?1 AND r.id = ?2 AND r.workspace_id = ?3
                "#,
            )
            .bind(epic_slug)
            .bind(scope.repo_id)
            .bind(scope.workspace_id)
            .fetch_all(executor)
            .await?
        }
        None => {
            sqlx::query_as(
                r#"
                SELECT e.id, e.slug, e.title, r.id, r.workspace_id
                FROM epics e
                JOIN repositories r ON r.id = e.repo_id
                WHERE e.slug = ?1
                "#,
            )
            .bind(epic_slug)
            .fetch_all(executor)
            .await?
        }
    };

    match rows.as_slice() {
        [] => Ok(None),
        [(epic_id, slug, title, repo_id, workspace_id)] => Ok(Some(EpicRecord {
            epic_id: *epic_id,
            slug: slug.clone(),
            title: title.clone(),
            scope: RepoScope {
                workspace_id: *workspace_id,
                repo_id: *repo_id,
            },
        })),
        _ => Err(StorageError::InvalidData {
            message: format!("epic slug is not unique: {epic_slug}"),
        }),
    }
}

async fn load_tasks<'e, E>(executor: E, epic_id: EpicId) -> Result<Vec<TaskRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    let rows: Vec<(
        TaskId,
        Option<TaskId>,
        Option<String>,
        String,
        String,
        Option<String>,
        String,
    )> = sqlx::query_as(
        r#"
        SELECT
            id,
            parent_task_id,
            local_ref,
            title,
            state,
            branch_name,
            merge_readiness
        FROM tasks
        WHERE epic_id = ?1
        ORDER BY (local_ref IS NULL), local_ref, created_at_ms, id
        "#,
    )
    .bind(epic_id)
    .fetch_all(executor)
    .await?;

    rows.into_iter()
        .map(
            |(task_id, parent_task_id, local_ref, title, state, branch_name, merge_readiness)| {
                Ok(TaskRecord {
                    task_id,
                    parent_task_id,
                    local_ref,
                    title,
                    state: decode_task_state(&state)?,
                    branch_name,
                    merge_readiness: decode_merge_readiness(&merge_readiness)?,
                })
            },
        )
        .collect()
}

async fn load_commands<'e, E>(
    executor: E,
    scope: RepoScope,
) -> Result<Vec<CommandRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    const INFLIGHT_LIMIT: i64 = 50;
    const RECENT_LIMIT: i64 = 25;

    let inflight_rows: Vec<(CommandId, i64, i64, String, String, Option<TaskId>)> = sqlx::query_as(
        r#"
            SELECT
                id,
                created_at_ms,
                updated_at_ms,
                kind,
                state,
                target_task_id
            FROM commands
            WHERE
                scope_kind = 'repo'
                AND scope_workspace_id = ?1
                AND scope_repo_id = ?2
                AND state IN ('queued', 'accepted', 'running', 'blocked', 'resumable')
            ORDER BY created_at_ms DESC
            LIMIT ?3
            "#,
    )
    .bind(scope.workspace_id)
    .bind(scope.repo_id)
    .bind(INFLIGHT_LIMIT)
    .fetch_all(executor)
    .await?;

    let recent_rows: Vec<(CommandId, i64, i64, String, String, Option<TaskId>)> = sqlx::query_as(
        r#"
        SELECT
            id,
            created_at_ms,
            updated_at_ms,
            kind,
            state,
            target_task_id
        FROM commands
        WHERE
            scope_kind = 'repo'
            AND scope_workspace_id = ?1
            AND scope_repo_id = ?2
        ORDER BY created_at_ms DESC
        LIMIT ?3
        "#,
    )
    .bind(scope.workspace_id)
    .bind(scope.repo_id)
    .bind(RECENT_LIMIT)
    .fetch_all(executor)
    .await?;

    let mut seen: HashSet<CommandId> = HashSet::new();
    let mut combined = Vec::with_capacity(inflight_rows.len() + recent_rows.len());

    for row in inflight_rows.into_iter().chain(recent_rows.into_iter()) {
        let (command_id, created_at_ms, updated_at_ms, kind, state, target_task_id) = row;
        if !seen.insert(command_id) {
            continue;
        }

        combined.push(CommandRecord {
            command_id,
            created_at_ms,
            updated_at_ms,
            kind,
            state: decode_command_state(&state)?,
            target_task_id,
        });
    }

    combined.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
    Ok(combined)
}

async fn load_command_last_updates<'e, E>(
    executor: E,
    commands: &[CommandRecord],
) -> Result<HashMap<CommandId, CommandUpdateRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    if commands.is_empty() {
        return Ok(HashMap::new());
    }

    let command_ids: Vec<CommandId> = commands.iter().map(|cmd| cmd.command_id).collect();

    let mut builder = sqlx::QueryBuilder::<Sqlite>::new(
        r#"
        SELECT id, command_id, created_at_ms, state, message, progress_current, progress_total
        FROM (
            SELECT
                id,
                command_id,
                created_at_ms,
                state,
                message,
                progress_current,
                progress_total,
                ROW_NUMBER() OVER (PARTITION BY command_id ORDER BY created_at_ms DESC, id DESC) AS rn
            FROM command_updates
            WHERE command_id IN (
        "#,
    );

    {
        let mut ids = builder.separated(", ");
        for command_id in command_ids {
            ids.push_bind(command_id);
        }
    }

    builder.push(
        r#"
            )
        )
        WHERE rn = 1
        "#,
    );

    let rows: Vec<(
        CommandUpdateId,
        CommandId,
        i64,
        String,
        Option<String>,
        Option<i64>,
        Option<i64>,
    )> = builder.build_query_as().fetch_all(executor).await?;

    let mut out = HashMap::with_capacity(rows.len());
    for (update_id, command_id, created_at_ms, state, message, progress_current, progress_total) in
        rows
    {
        out.insert(
            command_id,
            CommandUpdateRecord {
                update_id,
                command_id,
                created_at_ms,
                state: decode_command_state(&state)?,
                message,
                progress_current,
                progress_total,
            },
        );
    }

    Ok(out)
}

async fn load_daemon_presences<'e, E>(
    executor: E,
) -> Result<Vec<DaemonPresenceRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    const PRESENCE_LIMIT: i64 = 10;

    let rows: Vec<(
        HostInstanceId,
        HostId,
        Option<String>,
        i64,
        i64,
        Option<i64>,
    )> = sqlx::query_as(
        r#"
            SELECT
                dp.host_instance_id,
                dp.host_id,
                h.hostname,
                dp.connected_at_ms,
                dp.last_heartbeat_at_ms,
                dp.disconnected_at_ms
            FROM daemon_presence dp
            JOIN hosts h ON h.id = dp.host_id
            ORDER BY (dp.disconnected_at_ms IS NULL) DESC, dp.last_heartbeat_at_ms DESC
            LIMIT ?1
            "#,
    )
    .bind(PRESENCE_LIMIT)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .into_iter()
        .map(
            |(
                host_instance_id,
                host_id,
                hostname,
                connected_at_ms,
                last_heartbeat_at_ms,
                disconnected_at_ms,
            )| DaemonPresenceRecord {
                host_instance_id,
                host_id,
                hostname,
                connected_at_ms,
                last_heartbeat_at_ms,
                disconnected_at_ms,
            },
        )
        .collect())
}

async fn load_session_summaries<'e, E>(
    executor: E,
    epic_id: EpicId,
) -> Result<Vec<SessionSummaryRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    let rows: Vec<(TaskId, SessionId, SessionEventId, i64, String, Option<String>, Option<String>)> =
        sqlx::query_as(
            r#"
            SELECT task_id, session_id, id, created_at_ms, kind, turn_id, message_preview
            FROM (
                SELECT
                    task_id,
                    session_id,
                    id,
                    created_at_ms,
                    kind,
                    turn_id,
                    message_preview,
                    ROW_NUMBER() OVER (PARTITION BY task_id ORDER BY created_at_ms DESC, id DESC) AS rn
                FROM session_events
                WHERE epic_id = ?1 AND task_id IS NOT NULL
            )
            WHERE rn = 1
            ORDER BY created_at_ms DESC
            "#,
        )
        .bind(epic_id)
        .fetch_all(executor)
        .await?;

    Ok(rows
        .into_iter()
        .map(
            |(
                task_id,
                session_id,
                session_event_id,
                created_at_ms,
                kind,
                turn_id,
                message_preview,
            )| {
                SessionSummaryRecord {
                    session_id,
                    session_event_id,
                    task_id,
                    created_at_ms,
                    kind,
                    turn_id,
                    message_preview,
                }
            },
        )
        .collect())
}

async fn load_as_of_event_id<'e, E>(
    executor: E,
    scope: RepoScope,
) -> Result<Option<EventId>, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    let row: Option<(EventId,)> = sqlx::query_as(
        r#"
        SELECT id
        FROM events
        WHERE scope_kind = 'repo' AND scope_workspace_id = ?1 AND scope_repo_id = ?2
        ORDER BY created_at_ms DESC
        LIMIT 1
        "#,
    )
    .bind(scope.workspace_id)
    .bind(scope.repo_id)
    .fetch_optional(executor)
    .await?;

    Ok(row.map(|(id,)| id))
}

fn decode_merge_readiness(value: &str) -> Result<MergeReadiness, StorageError> {
    match value {
        "unknown" => Ok(MergeReadiness::Unknown),
        "ready" => Ok(MergeReadiness::Ready),
        "blocked" => Ok(MergeReadiness::Blocked),
        _ => Err(StorageError::InvalidData {
            message: format!("unknown merge_readiness: {value}"),
        }),
    }
}

fn decode_task_state(value: &str) -> Result<TaskState, StorageError> {
    match value {
        "todo" => Ok(TaskState::Todo),
        "in_progress" => Ok(TaskState::InProgress),
        "blocked" => Ok(TaskState::Blocked),
        "done" => Ok(TaskState::Done),
        _ => Err(StorageError::InvalidData {
            message: format!("unknown task state: {value}"),
        }),
    }
}

fn decode_command_state(value: &str) -> Result<CommandState, StorageError> {
    match value {
        "queued" => Ok(CommandState::Queued),
        "accepted" => Ok(CommandState::Accepted),
        "running" => Ok(CommandState::Running),
        "blocked" => Ok(CommandState::Blocked),
        "resumable" => Ok(CommandState::Resumable),
        "succeeded" => Ok(CommandState::Succeeded),
        "failed" => Ok(CommandState::Failed),
        "canceled" => Ok(CommandState::Canceled),
        _ => Err(StorageError::InvalidData {
            message: format!("unknown command state: {value}"),
        }),
    }
}
