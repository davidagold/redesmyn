use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_ids::EventId;
use sqlx::{Executor, Sqlite};

use crate::StorageError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRecord {
    pub id: EventId,
    pub created_at_ms: i64,
    pub kind: String,
    pub payload: Vec<u8>,
}

impl EventRecord {
    #[must_use]
    pub fn new_now(id: EventId, kind: impl Into<String>, payload: Vec<u8>) -> Self {
        let created_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(i64::MAX);

        Self {
            id,
            created_at_ms,
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
        INSERT INTO events (id, created_at_ms, kind, payload)
        VALUES (?1, ?2, ?3, ?4)
        "#,
    )
    .bind(event.id)
    .bind(event.created_at_ms)
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
    let row: Option<(EventId, i64, String, Vec<u8>)> = sqlx::query_as(
        r#"
        SELECT id, created_at_ms, kind, payload
        FROM events
        WHERE id = ?1
        "#,
    )
    .bind(id)
    .fetch_optional(executor)
    .await?;

    Ok(row.map(|(id, created_at_ms, kind, payload)| EventRecord {
        id,
        created_at_ms,
        kind,
        payload,
    }))
}
