use redesmyn_ids::{SessionEventId, SessionId};
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope};
use sqlx::SqlitePool;

pub(crate) async fn structured_turn_in_progress(
    pool: &SqlitePool,
    session_id: SessionId,
) -> Result<bool, ErrorEnvelope> {
    #[derive(sqlx::FromRow)]
    struct LastTurnEventRow {
        created_at_ms: i64,
        id: SessionEventId,
    }

    let last_started: Option<LastTurnEventRow> = sqlx::query_as(
        r#"
        SELECT created_at_ms, id
        FROM session_events
        WHERE session_id = ?1 AND kind = 'turn_started'
        ORDER BY created_at_ms DESC, id DESC
        LIMIT 1
        "#,
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await
    .map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::Internal, "Failed to query turn state.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    let Some(last_started) = last_started else {
        return Ok(false);
    };

    let last_completed: Option<LastTurnEventRow> = sqlx::query_as(
        r#"
        SELECT created_at_ms, id
        FROM session_events
        WHERE session_id = ?1 AND kind = 'turn_completed'
        ORDER BY created_at_ms DESC, id DESC
        LIMIT 1
        "#,
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await
    .map_err(|err| {
        ErrorEnvelope::new(ErrorCategory::Internal, "Failed to query turn state.")
            .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    let Some(last_completed) = last_completed else {
        return Ok(true);
    };

    Ok(last_started.created_at_ms > last_completed.created_at_ms
        || (last_started.created_at_ms == last_completed.created_at_ms
            && last_started.id > last_completed.id))
}

