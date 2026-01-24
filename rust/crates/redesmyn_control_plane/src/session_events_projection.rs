use sqlx::SqlitePool;

use redesmyn_protocol::session::SessionEventKind;
use redesmyn_protocol::{SessionEvent, Timestamp};

pub(crate) async fn apply_session_event_to_agent_session_row(
    pool: &SqlitePool,
    event: &SessionEvent,
) -> Result<(), redesmyn_storage::StorageError> {
    let now_ms = timestamp_ms(event.created_at);

    let (status, ended_at_ms, status_is_runtime) = match &event.kind {
        SessionEventKind::TurnStarted(_) => (Some("running"), None, true),
        SessionEventKind::TurnCompleted(completed) => {
            if completed.error.is_some() {
                (Some("error"), None, true)
            } else {
                (Some("blocked"), None, true)
            }
        }
        SessionEventKind::SessionEnded(_) => (Some("stopped"), Some(now_ms), false),
        _ => (None, None, false),
    };

    let external_session_ref_json = match &event.kind {
        SessionEventKind::TurnStarted(ev) => ev
            .external_session_ref
            .as_ref()
            .and_then(|r| serde_json::to_string(r).ok()),
        SessionEventKind::TurnCompleted(ev) => ev
            .external_session_ref
            .as_ref()
            .and_then(|r| serde_json::to_string(r).ok()),
        _ => None,
    };

    if status.is_none() && ended_at_ms.is_none() && external_session_ref_json.is_none() {
        return Ok(());
    }

    let mut query = sqlx::QueryBuilder::new("UPDATE agent_sessions SET updated_at_ms = ");
    query.push_bind(now_ms);

    if let Some(status) = status {
        query.push(", status = ");
        if status_is_runtime {
            query.push("CASE WHEN ended_at_ms IS NULL THEN ");
            query.push_bind(status);
            query.push(" ELSE status END");
        } else {
            query.push_bind(status);
        }
    }

    if let Some(ended_at_ms) = ended_at_ms {
        query.push(", ended_at_ms = COALESCE(ended_at_ms, ");
        query.push_bind(ended_at_ms);
        query.push(")");
    }

    if let Some(external) = external_session_ref_json {
        query.push(", external_session_ref = ");
        query.push_bind(external);
    }

    query.push(" WHERE session_id = ");
    query.push_bind(event.session_id);

    query.build().execute(pool).await?;
    Ok(())
}

fn timestamp_ms(value: Timestamp) -> i64 {
    let nanos = value.into_offset_date_time().unix_timestamp_nanos();
    let ms = nanos / 1_000_000;
    i64::try_from(ms).unwrap_or_else(|_| if ms.is_negative() { i64::MIN } else { i64::MAX })
}
