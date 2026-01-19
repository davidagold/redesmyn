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

    Ok(row.map(
        |(id, created_at_ms, scope_kind, scope_workspace_id, scope_repo_id, kind, payload)| {
            let scope = match (scope_kind.as_str(), scope_workspace_id, scope_repo_id) {
                ("none", None, None) => EventScope::None,
                ("repo", Some(workspace_id), Some(repo_id)) => EventScope::Repo {
                    workspace_id,
                    repo_id,
                },
                ("none" | "repo", _, _) => {
                    return Err(StorageError::InvalidData {
                        message: format!(
                            "invalid events scope columns: scope_kind={scope_kind} scope_workspace_id={scope_workspace_id:?} scope_repo_id={scope_repo_id:?}"
                        ),
                    });
                }
                _ => {
                    return Err(StorageError::InvalidData {
                        message: format!("unknown events scope_kind={scope_kind}"),
                    });
                }
            };

            Ok(EventRecord {
                id,
                created_at_ms,
                scope,
                kind,
                payload,
            })
        },
    )
    .transpose()?)
}
