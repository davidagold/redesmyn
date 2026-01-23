use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_ids::{CommandId, CommandUpdateId, RepoId, TaskId, WorkspaceId};
use sqlx::{Executor, Sqlite};

use crate::StorageError;
use crate::schema::CommandState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandScope {
    None,
    Repo {
        workspace_id: WorkspaceId,
        repo_id: RepoId,
    },
}

impl CommandScope {
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Repo { .. } => "repo",
        }
    }

    #[must_use]
    pub fn workspace_id(&self) -> Option<WorkspaceId> {
        match self {
            Self::None => None,
            Self::Repo { workspace_id, .. } => Some(*workspace_id),
        }
    }

    #[must_use]
    pub fn repo_id(&self) -> Option<RepoId> {
        match self {
            Self::None => None,
            Self::Repo { repo_id, .. } => Some(*repo_id),
        }
    }
}

fn scope_from_columns(
    scope_kind: &str,
    scope_workspace_id: Option<WorkspaceId>,
    scope_repo_id: Option<RepoId>,
) -> Result<CommandScope, StorageError> {
    match (scope_kind, scope_workspace_id, scope_repo_id) {
        ("none", None, None) => Ok(CommandScope::None),
        ("repo", Some(workspace_id), Some(repo_id)) => Ok(CommandScope::Repo {
            workspace_id,
            repo_id,
        }),
        ("none" | "repo", _, _) => Err(StorageError::InvalidData {
            message: format!(
                "invalid commands scope columns: scope_kind={scope_kind} scope_workspace_id={scope_workspace_id:?} scope_repo_id={scope_repo_id:?}"
            ),
        }),
        _ => Err(StorageError::InvalidData {
            message: format!("unknown commands scope_kind={scope_kind}"),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandRecord {
    pub command_id: CommandId,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub scope: CommandScope,
    pub target_task_id: Option<TaskId>,
    pub kind: String,
    pub state: CommandState,
    pub idempotency_key: Option<String>,
    pub created_by: Option<String>,
    pub payload: Vec<u8>,
}

impl CommandRecord {
    #[must_use]
    pub fn new_now(
        command_id: CommandId,
        scope: CommandScope,
        kind: impl Into<String>,
        state: CommandState,
        target_task_id: Option<TaskId>,
        idempotency_key: Option<String>,
        created_by: Option<String>,
        payload: Vec<u8>,
    ) -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(i64::MAX);

        Self {
            command_id,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            scope,
            target_task_id,
            kind: kind.into(),
            state,
            idempotency_key,
            created_by,
            payload,
        }
    }
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
    pub detail: Option<Vec<u8>>,
}

impl CommandUpdateRecord {
    #[must_use]
    pub fn new_now(
        update_id: CommandUpdateId,
        command_id: CommandId,
        state: CommandState,
        message: Option<String>,
        progress_current: Option<i64>,
        progress_total: Option<i64>,
        detail: Option<Vec<u8>>,
    ) -> Self {
        let created_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(i64::MAX);

        Self {
            update_id,
            command_id,
            created_at_ms,
            state,
            message,
            progress_current,
            progress_total,
            detail,
        }
    }
}

pub async fn insert_command<'e, E>(executor: E, command: &CommandRecord) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO commands (
            id,
            created_at_ms,
            updated_at_ms,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            target_task_id,
            kind,
            state,
            idempotency_key,
            created_by,
            payload
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
        "#,
    )
    .bind(command.command_id)
    .bind(command.created_at_ms)
    .bind(command.updated_at_ms)
    .bind(command.scope.kind())
    .bind(command.scope.workspace_id())
    .bind(command.scope.repo_id())
    .bind(command.target_task_id)
    .bind(&command.kind)
    .bind(command.state.as_str())
    .bind(command.idempotency_key.as_deref())
    .bind(command.created_by.as_deref())
    .bind(&command.payload)
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn get_command<'e, E>(
    executor: E,
    command_id: CommandId,
) -> Result<Option<CommandRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(
        CommandId,
        i64,
        i64,
        String,
        Option<WorkspaceId>,
        Option<RepoId>,
        Option<TaskId>,
        String,
        String,
        Option<String>,
        Option<String>,
        Vec<u8>,
    )> = sqlx::query_as(
        r#"
        SELECT
            id,
            created_at_ms,
            updated_at_ms,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            target_task_id,
            kind,
            state,
            idempotency_key,
            created_by,
            payload
        FROM commands
        WHERE id = ?1
        "#,
    )
    .bind(command_id)
    .fetch_optional(executor)
    .await?;

    Ok(row
        .map(
            |(
                command_id,
                created_at_ms,
                updated_at_ms,
                scope_kind,
                scope_workspace_id,
                scope_repo_id,
                target_task_id,
                kind,
                state,
                idempotency_key,
                created_by,
                payload,
            )| {
                Ok::<CommandRecord, StorageError>(CommandRecord {
                    command_id,
                    created_at_ms,
                    updated_at_ms,
                    scope: scope_from_columns(
                        scope_kind.as_str(),
                        scope_workspace_id,
                        scope_repo_id,
                    )?,
                    target_task_id,
                    kind,
                    state: decode_command_state(&state)?,
                    idempotency_key,
                    created_by,
                    payload,
                })
            },
        )
        .transpose()?)
}

pub async fn get_command_last_update<'e, E>(
    executor: E,
    command_id: CommandId,
) -> Result<Option<CommandUpdateRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(
        CommandUpdateId,
        CommandId,
        i64,
        String,
        Option<String>,
        Option<i64>,
        Option<i64>,
        Option<Vec<u8>>,
    )> = sqlx::query_as(
        r#"
        SELECT
            id,
            command_id,
            created_at_ms,
            state,
            message,
            progress_current,
            progress_total,
            detail
        FROM command_updates
        WHERE command_id = ?1
        ORDER BY created_at_ms DESC, id DESC
        LIMIT 1
        "#,
    )
    .bind(command_id)
    .fetch_optional(executor)
    .await?;

    Ok(row
        .map(
            |(
                update_id,
                command_id,
                created_at_ms,
                state,
                message,
                progress_current,
                progress_total,
                detail,
            )| {
                Ok::<CommandUpdateRecord, StorageError>(CommandUpdateRecord {
                    update_id,
                    command_id,
                    created_at_ms,
                    state: decode_command_state(&state)?,
                    message,
                    progress_current,
                    progress_total,
                    detail,
                })
            },
        )
        .transpose()?)
}

pub async fn find_command_by_idempotency_key<'e, E>(
    executor: E,
    scope: CommandScope,
    kind: &str,
    idempotency_key: &str,
) -> Result<Option<CommandId>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let command_id: Option<CommandId> = match scope {
        CommandScope::None => {
            sqlx::query_scalar(
                r#"
                SELECT id
                FROM commands
                WHERE scope_kind = 'none'
                  AND scope_workspace_id IS NULL
                  AND scope_repo_id IS NULL
                  AND kind = ?1
                  AND idempotency_key = ?2
                LIMIT 1
                "#,
            )
            .bind(kind)
            .bind(idempotency_key)
            .fetch_optional(executor)
            .await?
        }
        CommandScope::Repo {
            workspace_id,
            repo_id,
        } => {
            sqlx::query_scalar(
                r#"
                SELECT id
                FROM commands
                WHERE scope_kind = 'repo'
                  AND scope_workspace_id = ?1
                  AND scope_repo_id = ?2
                  AND kind = ?3
                  AND idempotency_key = ?4
                LIMIT 1
                "#,
            )
            .bind(workspace_id)
            .bind(repo_id)
            .bind(kind)
            .bind(idempotency_key)
            .fetch_optional(executor)
            .await?
        }
    };

    Ok(command_id)
}

pub async fn insert_command_update<'e, E>(
    executor: E,
    update: &CommandUpdateRecord,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO command_updates (
            id,
            command_id,
            created_at_ms,
            state,
            message,
            progress_current,
            progress_total,
            detail
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#,
    )
    .bind(update.update_id)
    .bind(update.command_id)
    .bind(update.created_at_ms)
    .bind(update.state.as_str())
    .bind(update.message.as_deref())
    .bind(update.progress_current)
    .bind(update.progress_total)
    .bind(update.detail.as_deref())
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn update_command_state<'e, E>(
    executor: E,
    command_id: CommandId,
    state: CommandState,
    updated_at_ms: i64,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        UPDATE commands
        SET state = ?1,
            updated_at_ms = ?2
        WHERE id = ?3
        "#,
    )
    .bind(state.as_str())
    .bind(updated_at_ms)
    .bind(command_id)
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn count_inflight_commands<'e, E>(
    executor: E,
    scope: CommandScope,
) -> Result<i64, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    const INFLIGHT_STATES: &str = "('queued', 'accepted', 'running', 'blocked', 'resumable')";
    let count: i64 = match scope {
        CommandScope::None => {
            sqlx::query_scalar(&format!(
                r#"
                SELECT COUNT(*)
                FROM commands
                WHERE scope_kind = 'none'
                  AND scope_workspace_id IS NULL
                  AND scope_repo_id IS NULL
                  AND state IN {INFLIGHT_STATES}
                "#
            ))
            .fetch_one(executor)
            .await?
        }
        CommandScope::Repo {
            workspace_id,
            repo_id,
        } => {
            sqlx::query_scalar(&format!(
                r#"
                SELECT COUNT(*)
                FROM commands
                WHERE scope_kind = 'repo'
                  AND scope_workspace_id = ?1
                  AND scope_repo_id = ?2
                  AND state IN {INFLIGHT_STATES}
                "#
            ))
            .bind(workspace_id)
            .bind(repo_id)
            .fetch_one(executor)
            .await?
        }
    };

    Ok(count)
}

