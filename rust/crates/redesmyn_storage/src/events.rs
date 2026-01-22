use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_ids::{EventId, RepoId, WorkspaceId};
use sqlx::{Executor, Sqlite};

use crate::StorageError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventScope {
    None,
    Repo { workspace_id: WorkspaceId, repo_id: RepoId },
}

impl EventScope {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRow {
    pub rowid: i64,
    pub record: EventRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRecord {
    pub id: EventId,
    pub created_at_ms: i64,
    pub scope: EventScope,
    pub kind: String,
    pub payload: Vec<u8>,
}

impl EventRecord {
    #[must_use]
    pub fn new_now(id: EventId, kind: impl Into<String>, payload: Vec<u8>) -> Self {
        Self::new_now_in_scope(id, EventScope::None, kind, payload)
    }

    #[must_use]
    pub fn new_now_in_scope(
        id: EventId,
        scope: EventScope,
        kind: impl Into<String>,
        payload: Vec<u8>,
    ) -> Self {
        let created_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(i64::MAX);

        Self {
            id,
            created_at_ms,
            scope,
            kind: kind.into(),
            payload,
        }
    }
}

fn scope_from_columns(
    scope_kind: &str,
    scope_workspace_id: Option<WorkspaceId>,
    scope_repo_id: Option<RepoId>,
) -> Result<EventScope, StorageError> {
    match (scope_kind, scope_workspace_id, scope_repo_id) {
        ("none", None, None) => Ok(EventScope::None),
        ("repo", Some(workspace_id), Some(repo_id)) => Ok(EventScope::Repo {
            workspace_id,
            repo_id,
        }),
        ("none" | "repo", _, _) => Err(StorageError::InvalidData {
            message: format!(
                "invalid events scope columns: scope_kind={scope_kind} scope_workspace_id={scope_workspace_id:?} scope_repo_id={scope_repo_id:?}"
            ),
        }),
        _ => Err(StorageError::InvalidData {
            message: format!("unknown events scope_kind={scope_kind}"),
        }),
    }
}

pub async fn insert_event<'e, E>(executor: E, event: &EventRecord) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO events (
            id,
            created_at_ms,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            kind,
            payload
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#,
    )
    .bind(event.id)
    .bind(event.created_at_ms)
    .bind(event.scope.kind())
    .bind(event.scope.workspace_id())
    .bind(event.scope.repo_id())
    .bind(&event.kind)
    .bind(&event.payload)
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn get_event<'e, E>(executor: E, id: EventId) -> Result<Option<EventRecord>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(EventId, i64, String, Option<WorkspaceId>, Option<RepoId>, String, Vec<u8>)> =
        sqlx::query_as(
        r#"
        SELECT
            id,
            created_at_ms,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            kind,
            payload
        FROM events
        WHERE id = ?1
        "#,
        )
        .bind(id)
        .fetch_optional(executor)
        .await?;

    Ok(row
        .map(
            |(id, created_at_ms, scope_kind, scope_workspace_id, scope_repo_id, kind, payload)| {
                Ok::<EventRecord, StorageError>(EventRecord {
                    id,
                    created_at_ms,
                    scope: scope_from_columns(
                        scope_kind.as_str(),
                        scope_workspace_id,
                        scope_repo_id,
                    )?,
                    kind,
                    payload,
                })
            },
        )
        .transpose()?)
}

pub async fn get_event_rowid_and_scope<'e, E>(
    executor: E,
    id: EventId,
) -> Result<Option<(i64, EventScope)>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<(i64, String, Option<WorkspaceId>, Option<RepoId>)> = sqlx::query_as(
        r#"
        SELECT
            rowid,
            scope_kind,
            scope_workspace_id,
            scope_repo_id
        FROM events
        WHERE id = ?1
        "#,
    )
    .bind(id)
    .fetch_optional(executor)
    .await?;

    Ok(row
        .map(|(rowid, scope_kind, scope_workspace_id, scope_repo_id)| {
            Ok::<(i64, EventScope), StorageError>((
                rowid,
                scope_from_columns(scope_kind.as_str(), scope_workspace_id, scope_repo_id)?,
            ))
        })
        .transpose()?)
}

pub async fn list_event_rows_in_scope_after_rowid<'e, E>(
    executor: E,
    scope: EventScope,
    after_rowid: Option<i64>,
    limit: usize,
) -> Result<Vec<EventRow>, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let rows: Vec<(i64, EventId, i64, String, Option<WorkspaceId>, Option<RepoId>, String, Vec<u8>)> =
        match (scope, after_rowid) {
            (EventScope::None, Some(after_rowid)) => {
                sqlx::query_as(
                    r#"
                    SELECT
                        rowid,
                        id,
                        created_at_ms,
                        scope_kind,
                        scope_workspace_id,
                        scope_repo_id,
                        kind,
                        payload
                    FROM events
                    WHERE scope_kind = 'none'
                      AND scope_workspace_id IS NULL
                      AND scope_repo_id IS NULL
                      AND rowid > ?1
                    ORDER BY rowid
                    LIMIT ?2
                    "#,
                )
                .bind(after_rowid)
                .bind(limit as i64)
                .fetch_all(executor)
                .await?
            }
            (EventScope::Repo {
                workspace_id,
                repo_id,
            }, Some(after_rowid)) => {
                sqlx::query_as(
                    r#"
                    SELECT
                        rowid,
                        id,
                        created_at_ms,
                        scope_kind,
                        scope_workspace_id,
                        scope_repo_id,
                        kind,
                        payload
                    FROM events
                    WHERE scope_kind = 'repo'
                      AND scope_workspace_id = ?1
                      AND scope_repo_id = ?2
                      AND rowid > ?3
                    ORDER BY rowid
                    LIMIT ?4
                    "#,
                )
                .bind(workspace_id)
                .bind(repo_id)
                .bind(after_rowid)
                .bind(limit as i64)
                .fetch_all(executor)
                .await?
            }
            (EventScope::None, None) => {
                sqlx::query_as(
                    r#"
                    SELECT
                        rowid,
                        id,
                        created_at_ms,
                        scope_kind,
                        scope_workspace_id,
                        scope_repo_id,
                        kind,
                        payload
                    FROM events
                    WHERE scope_kind = 'none'
                      AND scope_workspace_id IS NULL
                      AND scope_repo_id IS NULL
                    ORDER BY rowid
                    LIMIT ?1
                    "#,
                )
                .bind(limit as i64)
                .fetch_all(executor)
                .await?
            }
            (EventScope::Repo {
                workspace_id,
                repo_id,
            }, None) => {
                sqlx::query_as(
                    r#"
                    SELECT
                        rowid,
                        id,
                        created_at_ms,
                        scope_kind,
                        scope_workspace_id,
                        scope_repo_id,
                        kind,
                        payload
                    FROM events
                    WHERE scope_kind = 'repo'
                      AND scope_workspace_id = ?1
                      AND scope_repo_id = ?2
                    ORDER BY rowid
                    LIMIT ?3
                    "#,
                )
                .bind(workspace_id)
                .bind(repo_id)
                .bind(limit as i64)
                .fetch_all(executor)
                .await?
            }
        };

    rows.into_iter()
        .map(
            |(
                rowid,
                id,
                created_at_ms,
                scope_kind,
                scope_workspace_id,
                scope_repo_id,
                kind,
                payload,
            )| {
                Ok(EventRow {
                    rowid,
                    record: EventRecord {
                        id,
                        created_at_ms,
                        scope: scope_from_columns(
                            scope_kind.as_str(),
                            scope_workspace_id,
                            scope_repo_id,
                        )?,
                        kind,
                        payload,
                    },
                })
            },
        )
        .collect()
}
