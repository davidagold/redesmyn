use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_ids::{EpicId, EventId, RequestId, TaskId};
use redesmyn_logging::tracing::warn;
use serde_json::Value;
use sqlx::SqlitePool;
use tokio::sync::{RwLock, broadcast};
use tokio::task::JoinHandle;

use redesmyn_storage::StorageError;
use redesmyn_storage::events::{EventRecord, EventScope};

use crate::event_log::{EventLog, EventLogSubscriptionItem};
use crate::session_events::TASK_SESSION_TURN_COMPLETED_EVENT;

const DIRECTOR_WAKE_DISPATCHED_EVENT: &str = "director.wake.dispatched";
const DIRECTOR_WAKE_ACKED_EVENT: &str = "director.wake.acked";
const DIRECTOR_WAKE_ACK_REJECTED_EVENT: &str = "director.wake.ack_rejected";
const DIRECTOR_WAKE_RESUME_REQUIRED_EVENT: &str = "director.wake.resume_required";

const REASON_TASK_TURN_COMPLETED: u64 = 1 << 0;
const REASON_COMMAND_OUTCOME: u64 = 1 << 1;
const REASON_CONDUCTOR_OVERRIDE: u64 = 1 << 2;
const REASON_NON_DIRECTOR_QUEUE_UPDATE: u64 = 1 << 3;
const REASON_BACKLOG: u64 = 1 << 4;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectorWakeConfig {
    pub backlog_page_size: usize,
    pub max_events_per_chunk: usize,
    pub max_chunk_bytes: usize,
    pub notifications_buffer: usize,
}

impl Default for DirectorWakeConfig {
    fn default() -> Self {
        Self {
            backlog_page_size: 256,
            max_events_per_chunk: 64,
            max_chunk_bytes: 48 * 1024,
            notifications_buffer: 128,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectorMode {
    Active,
    Paused,
    Error,
    ResumeRequired,
}

impl DirectorMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Error => "error",
            Self::ResumeRequired => "resume_required",
        }
    }

    fn from_db(value: &str) -> Result<Self, StorageError> {
        match value {
            "active" => Ok(Self::Active),
            "paused" => Ok(Self::Paused),
            "error" => Ok(Self::Error),
            "resume_required" => Ok(Self::ResumeRequired),
            _ => Err(StorageError::InvalidData {
                message: format!("unknown director_wake_state.director_mode={value}"),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectorWakeReason {
    TaskTurnCompleted,
    CommandOutcome,
    ConductorOverride,
    NonDirectorQueueUpdate,
    Backlog,
}

fn reason_mask_from_event(reason: DirectorWakeReason) -> u64 {
    match reason {
        DirectorWakeReason::TaskTurnCompleted => REASON_TASK_TURN_COMPLETED,
        DirectorWakeReason::CommandOutcome => REASON_COMMAND_OUTCOME,
        DirectorWakeReason::ConductorOverride => REASON_CONDUCTOR_OVERRIDE,
        DirectorWakeReason::NonDirectorQueueUpdate => REASON_NON_DIRECTOR_QUEUE_UPDATE,
        DirectorWakeReason::Backlog => REASON_BACKLOG,
    }
}

fn reasons_from_mask(mask: u64) -> Vec<DirectorWakeReason> {
    let mut out = Vec::new();
    if (mask & REASON_TASK_TURN_COMPLETED) != 0 {
        out.push(DirectorWakeReason::TaskTurnCompleted);
    }
    if (mask & REASON_COMMAND_OUTCOME) != 0 {
        out.push(DirectorWakeReason::CommandOutcome);
    }
    if (mask & REASON_CONDUCTOR_OVERRIDE) != 0 {
        out.push(DirectorWakeReason::ConductorOverride);
    }
    if (mask & REASON_NON_DIRECTOR_QUEUE_UPDATE) != 0 {
        out.push(DirectorWakeReason::NonDirectorQueueUpdate);
    }
    if (mask & REASON_BACKLOG) != 0 {
        out.push(DirectorWakeReason::Backlog);
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorWakeEvent {
    pub event_id: EventId,
    pub occurred_at_ms: i64,
    pub event_type: String,
    pub json_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorWakeChunk {
    pub index: u32,
    pub total: u32,
    pub events: Vec<DirectorWakeEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorWakeSummary {
    pub epic_id: EpicId,
    pub wake_id: String,
    pub cursor: Option<EventId>,
    pub high_water_event_id: EventId,
    pub queue_size: usize,
    pub reasons: Vec<DirectorWakeReason>,
    pub last_wake_reason: Vec<DirectorWakeReason>,
    pub last_wake_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorWakePayload {
    pub summary: DirectorWakeSummary,
    pub chunks: Vec<DirectorWakeChunk>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorWakeSnapshot {
    pub epic_id: EpicId,
    pub director_mode: DirectorMode,
    pub ack_cursor_event_id: Option<EventId>,
    pub in_flight_wake_id: Option<String>,
    pub in_flight_high_water_event_id: Option<EventId>,
    pub pending_high_water_event_id: Option<EventId>,
    pub pending_reasons: Vec<DirectorWakeReason>,
    pub last_wake_reason: Vec<DirectorWakeReason>,
    pub last_wake_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectorWakeNotification {
    WakeDispatched(DirectorWakePayload),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorWakeAckResult {
    pub advanced_to: EventId,
    pub replay_payload: Option<DirectorWakePayload>,
    pub next_payload: Option<DirectorWakePayload>,
}

#[derive(Debug, thiserror::Error)]
pub enum DirectorWakeError {
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("epic not found: {epic_id}")]
    EpicNotFound { epic_id: EpicId },
    #[error("wake is not in flight for epic {epic_id}")]
    WakeNotInFlight { epic_id: EpicId },
    #[error("wake id mismatch for epic {epic_id}: expected {expected}, got {actual}")]
    WakeIdMismatch {
        epic_id: EpicId,
        expected: String,
        actual: String,
    },
    #[error(
        "invalid ack for epic {epic_id}: wake_id={wake_id} ack={received_ack} expected_cursor={expected_cursor:?} high_water={high_water_event_id}"
    )]
    InvalidAck {
        epic_id: EpicId,
        wake_id: String,
        expected_cursor: Option<EventId>,
        high_water_event_id: EventId,
        received_ack: EventId,
    },
}

#[derive(Clone)]
pub struct DirectorWakeController {
    inner: Arc<DirectorWakeControllerInner>,
}

struct DirectorWakeControllerInner {
    pool: SqlitePool,
    event_log: EventLog,
    config: DirectorWakeConfig,
    notifications_tx: broadcast::Sender<DirectorWakeNotification>,
    watchers: RwLock<HashMap<EpicId, JoinHandle<()>>>,
}

#[derive(Debug, Clone)]
struct EpicScope {
    workspace_id: redesmyn_ids::WorkspaceId,
    repo_id: redesmyn_ids::RepoId,
}

#[derive(Debug, Clone)]
struct WakeStateRow {
    director_mode: DirectorMode,
    ack_cursor_event_id: Option<EventId>,
    in_flight_wake_id: Option<String>,
    in_flight_high_water_event_id: Option<EventId>,
    in_flight_reason_mask: u64,
    pending_high_water_event_id: Option<EventId>,
    pending_reason_mask: u64,
    last_wake_reason_mask: u64,
    last_wake_at_ms: Option<i64>,
}

impl DirectorWakeController {
    #[must_use]
    pub fn new(pool: SqlitePool, event_log: EventLog) -> Self {
        Self::new_with_config(pool, event_log, DirectorWakeConfig::default())
    }

    #[must_use]
    pub fn new_with_config(
        pool: SqlitePool,
        event_log: EventLog,
        config: DirectorWakeConfig,
    ) -> Self {
        let (notifications_tx, _rx) = broadcast::channel(config.notifications_buffer.max(1));
        Self {
            inner: Arc::new(DirectorWakeControllerInner {
                pool,
                event_log,
                config,
                notifications_tx,
                watchers: RwLock::new(HashMap::new()),
            }),
        }
    }

    pub fn subscribe_notifications(&self) -> broadcast::Receiver<DirectorWakeNotification> {
        self.inner.notifications_tx.subscribe()
    }

    pub async fn start_epic(&self, epic_id: EpicId) -> Result<(), DirectorWakeError> {
        self.ensure_state_row(epic_id).await?;
        let mut watchers = self.inner.watchers.write().await;
        if watchers.contains_key(&epic_id) {
            return Ok(());
        }

        let this = self.clone();
        let handle = tokio::spawn(async move {
            if let Err(err) = this.run_epic_event_loop(epic_id).await {
                warn!(epic_id = %epic_id, error = %err, "director wake event loop exited");
            }
        });
        watchers.insert(epic_id, handle);
        Ok(())
    }

    pub async fn stop_epic(&self, epic_id: EpicId) {
        let mut watchers = self.inner.watchers.write().await;
        if let Some(handle) = watchers.remove(&epic_id) {
            handle.abort();
        }
    }

    pub async fn set_mode(
        &self,
        epic_id: EpicId,
        mode: DirectorMode,
    ) -> Result<(), DirectorWakeError> {
        self.ensure_state_row(epic_id).await?;
        sqlx::query(
            r#"
            UPDATE director_wake_state
            SET updated_at_ms = ?2, director_mode = ?3
            WHERE epic_id = ?1
            "#,
        )
        .bind(epic_id)
        .bind(now_ms())
        .bind(mode.as_str())
        .execute(&self.inner.pool)
        .await?;

        if matches!(mode, DirectorMode::Active) {
            let _ = self.dispatch_pending_wake(epic_id).await?;
        }
        Ok(())
    }

    pub async fn mark_interrupted(&self, epic_id: EpicId) -> Result<(), DirectorWakeError> {
        self.ensure_state_row(epic_id).await?;
        let scope = self.epic_scope(epic_id).await?;
        let now = now_ms();

        let mut row = self.load_state_row(epic_id).await?;
        let in_flight_high_water = row.in_flight_high_water_event_id;
        if let Some(high_water) = in_flight_high_water {
            row.pending_high_water_event_id = self
                .newest_event_id(
                    scope.as_event_scope(),
                    row.pending_high_water_event_id,
                    Some(high_water),
                )
                .await?;
        }
        row.pending_reason_mask |= row.in_flight_reason_mask | REASON_BACKLOG;
        row.in_flight_wake_id = None;
        row.in_flight_high_water_event_id = None;
        row.in_flight_reason_mask = 0;
        row.director_mode = DirectorMode::ResumeRequired;

        sqlx::query(
            r#"
            UPDATE director_wake_state
            SET
                updated_at_ms = ?2,
                director_mode = ?3,
                in_flight_wake_id = NULL,
                in_flight_high_water_event_id = NULL,
                in_flight_reason_mask = 0,
                pending_high_water_event_id = ?4,
                pending_reason_mask = ?5
            WHERE epic_id = ?1
            "#,
        )
        .bind(epic_id)
        .bind(now)
        .bind(DirectorMode::ResumeRequired.as_str())
        .bind(row.pending_high_water_event_id)
        .bind(row.pending_reason_mask as i64)
        .execute(&self.inner.pool)
        .await?;

        let payload = serde_json::to_vec(&serde_json::json!({
            "epic_id": epic_id,
            "mode": "resume_required",
            "pending_high_water_event_id": row.pending_high_water_event_id,
        }))
        .unwrap_or_default();
        let _ = self
            .inner
            .event_log
            .append_event(
                scope.as_event_scope(),
                DIRECTOR_WAKE_RESUME_REQUIRED_EVENT,
                payload,
            )
            .await;
        Ok(())
    }

    pub async fn resume(
        &self,
        epic_id: EpicId,
    ) -> Result<Option<DirectorWakePayload>, DirectorWakeError> {
        self.set_mode(epic_id, DirectorMode::Active).await?;
        self.dispatch_pending_wake(epic_id).await
    }

    pub async fn dispatch_pending_wake(
        &self,
        epic_id: EpicId,
    ) -> Result<Option<DirectorWakePayload>, DirectorWakeError> {
        let (payload, is_new) = self.open_or_replay_wake(epic_id).await?;
        if let Some(ref wake_payload) = payload {
            if is_new {
                let _ = self
                    .inner
                    .notifications_tx
                    .send(DirectorWakeNotification::WakeDispatched(
                        wake_payload.clone(),
                    ));
            }
        }
        Ok(payload)
    }

    pub async fn ack_wake(
        &self,
        epic_id: EpicId,
        wake_id: &str,
        ack_event_id: EventId,
    ) -> Result<DirectorWakeAckResult, DirectorWakeError> {
        self.ensure_state_row(epic_id).await?;
        let scope = self.epic_scope(epic_id).await?;
        let mut row = self.load_state_row(epic_id).await?;

        let Some(in_flight_wake_id) = row.in_flight_wake_id.clone() else {
            return Err(DirectorWakeError::WakeNotInFlight { epic_id });
        };
        if in_flight_wake_id != wake_id {
            return Err(DirectorWakeError::WakeIdMismatch {
                epic_id,
                expected: in_flight_wake_id,
                actual: wake_id.to_owned(),
            });
        }

        let event_scope = scope.as_event_scope();
        let high_water = row
            .in_flight_high_water_event_id
            .ok_or(DirectorWakeError::WakeNotInFlight { epic_id })?;
        let high_water_rowid = self.event_rowid_in_scope(event_scope, high_water).await?;
        let Some((ack_rowid, ack_event_record)) = self
            .event_rowid_and_record_in_scope(event_scope, ack_event_id)
            .await?
        else {
            let err = DirectorWakeError::InvalidAck {
                epic_id,
                wake_id: wake_id.to_owned(),
                expected_cursor: row.ack_cursor_event_id,
                high_water_event_id: high_water,
                received_ack: ack_event_id,
            };
            self.append_invalid_ack_event(event_scope, &err).await;
            return Err(err);
        };
        let cursor_rowid = if let Some(cursor) = row.ack_cursor_event_id {
            Some(self.event_rowid_in_scope(event_scope, cursor).await?)
        } else {
            None
        };

        let valid_rowid_range =
            cursor_rowid.is_none_or(|cursor| ack_rowid > cursor) && ack_rowid <= high_water_rowid;
        let ack_payload = parse_json_payload(&ack_event_record.payload);
        let ack_is_relevant = self
            .event_matches_epic(epic_id, &ack_event_record, ack_payload.as_ref())
            .await?;
        if !valid_rowid_range || !ack_is_relevant {
            let err = DirectorWakeError::InvalidAck {
                epic_id,
                wake_id: wake_id.to_owned(),
                expected_cursor: row.ack_cursor_event_id,
                high_water_event_id: high_water,
                received_ack: ack_event_id,
            };
            self.append_invalid_ack_event(event_scope, &err).await;
            return Err(err);
        }

        row.ack_cursor_event_id = Some(ack_event_id);
        let replay_suffix = ack_event_id != high_water;
        if !replay_suffix {
            row.in_flight_wake_id = None;
            row.in_flight_high_water_event_id = None;
            row.in_flight_reason_mask = 0;
        }

        sqlx::query(
            r#"
            UPDATE director_wake_state
            SET
                updated_at_ms = ?2,
                ack_cursor_event_id = ?3,
                in_flight_wake_id = ?4,
                in_flight_high_water_event_id = ?5,
                in_flight_reason_mask = ?6
            WHERE epic_id = ?1
            "#,
        )
        .bind(epic_id)
        .bind(now_ms())
        .bind(row.ack_cursor_event_id)
        .bind(row.in_flight_wake_id.clone())
        .bind(row.in_flight_high_water_event_id)
        .bind(row.in_flight_reason_mask as i64)
        .execute(&self.inner.pool)
        .await?;

        let payload = serde_json::to_vec(&serde_json::json!({
            "epic_id": epic_id,
            "wake_id": wake_id,
            "ack_event_id": ack_event_id,
            "high_water_event_id": high_water,
            "replay_suffix": replay_suffix,
        }))
        .unwrap_or_default();
        let _ = self
            .inner
            .event_log
            .append_event(event_scope, DIRECTOR_WAKE_ACKED_EVENT, payload)
            .await;

        let replay_payload = if replay_suffix {
            let (payload, _) = self.open_or_replay_wake(epic_id).await?;
            payload
        } else {
            None
        };
        let next_payload = if !replay_suffix {
            self.dispatch_pending_wake(epic_id).await?
        } else {
            None
        };
        Ok(DirectorWakeAckResult {
            advanced_to: ack_event_id,
            replay_payload,
            next_payload,
        })
    }

    pub async fn snapshot(
        &self,
        epic_id: EpicId,
    ) -> Result<DirectorWakeSnapshot, DirectorWakeError> {
        self.ensure_state_row(epic_id).await?;
        let row = self.load_state_row(epic_id).await?;
        Ok(DirectorWakeSnapshot {
            epic_id,
            director_mode: row.director_mode,
            ack_cursor_event_id: row.ack_cursor_event_id,
            in_flight_wake_id: row.in_flight_wake_id,
            in_flight_high_water_event_id: row.in_flight_high_water_event_id,
            pending_high_water_event_id: row.pending_high_water_event_id,
            pending_reasons: reasons_from_mask(row.pending_reason_mask),
            last_wake_reason: reasons_from_mask(row.last_wake_reason_mask),
            last_wake_at_ms: row.last_wake_at_ms,
        })
    }

    async fn run_epic_event_loop(&self, epic_id: EpicId) -> Result<(), DirectorWakeError> {
        let scope = self.epic_scope(epic_id).await?;
        let mut after_event_id = self
            .resume_cursor_for_subscription(epic_id, scope.clone())
            .await?;

        loop {
            let mut sub = self
                .inner
                .event_log
                .subscribe(scope.as_event_scope(), after_event_id);

            loop {
                match sub.recv().await {
                    Some(EventLogSubscriptionItem::Event(event)) => {
                        if let Err(err) = self.handle_repo_event(epic_id, &scope, event).await {
                            warn!(epic_id = %epic_id, error = %err, "failed handling repo event");
                        }
                    }
                    Some(EventLogSubscriptionItem::ResyncRequired(resync)) => {
                        after_event_id = resync.resume_after_event_id;
                        break;
                    }
                    None => return Ok(()),
                }
            }
        }
    }

    async fn handle_repo_event(
        &self,
        epic_id: EpicId,
        _scope: &EpicScope,
        event: EventRecord,
    ) -> Result<(), DirectorWakeError> {
        let parsed = parse_json_payload(&event.payload);
        let relevant = self
            .event_matches_epic(epic_id, &event, parsed.as_ref())
            .await?;
        if !relevant {
            return Ok(());
        }

        let significant_reason = significant_reason_for_event(event.kind.as_str(), parsed.as_ref());
        let row = self.load_state_row(epic_id).await?;

        if row.in_flight_wake_id.is_some() || !matches!(row.director_mode, DirectorMode::Active) {
            let reason_mask = significant_reason
                .map(reason_mask_from_event)
                .unwrap_or(REASON_BACKLOG);
            sqlx::query(
                r#"
                UPDATE director_wake_state
                SET
                    updated_at_ms = ?2,
                    pending_high_water_event_id = ?3,
                    pending_reason_mask = pending_reason_mask | ?4
                WHERE epic_id = ?1
                "#,
            )
            .bind(epic_id)
            .bind(now_ms())
            .bind(event.id)
            .bind(reason_mask as i64)
            .execute(&self.inner.pool)
            .await?;
            return Ok(());
        }

        let Some(reason) = significant_reason else {
            return Ok(());
        };

        sqlx::query(
            r#"
            UPDATE director_wake_state
            SET
                updated_at_ms = ?2,
                pending_high_water_event_id = ?3,
                pending_reason_mask = pending_reason_mask | ?4
            WHERE epic_id = ?1
            "#,
        )
        .bind(epic_id)
        .bind(now_ms())
        .bind(event.id)
        .bind(reason_mask_from_event(reason) as i64)
        .execute(&self.inner.pool)
        .await?;

        let (payload, is_new) = self.open_or_replay_wake(epic_id).await?;
        if let Some(payload) = payload {
            if is_new {
                let _ = self
                    .inner
                    .notifications_tx
                    .send(DirectorWakeNotification::WakeDispatched(payload));
            }
        }

        Ok(())
    }

    async fn open_or_replay_wake(
        &self,
        epic_id: EpicId,
    ) -> Result<(Option<DirectorWakePayload>, bool), DirectorWakeError> {
        self.ensure_state_row(epic_id).await?;
        let scope = self.epic_scope(epic_id).await?;
        let mut row = self.load_state_row(epic_id).await?;

        if !matches!(row.director_mode, DirectorMode::Active) {
            return Ok((None, false));
        }

        let mut is_new = false;
        if row.in_flight_wake_id.is_none() {
            let Some(high_water) = row.pending_high_water_event_id else {
                return Ok((None, false));
            };
            let reason_mask = if row.pending_reason_mask == 0 {
                REASON_BACKLOG
            } else {
                row.pending_reason_mask
            };

            row.in_flight_wake_id = Some(RequestId::new().to_string());
            row.in_flight_high_water_event_id = Some(high_water);
            row.in_flight_reason_mask = reason_mask;
            row.last_wake_reason_mask = reason_mask;
            row.last_wake_at_ms = Some(now_ms());
            row.pending_high_water_event_id = None;
            row.pending_reason_mask = 0;
            is_new = true;

            sqlx::query(
                r#"
                UPDATE director_wake_state
                SET
                    updated_at_ms = ?2,
                    in_flight_wake_id = ?3,
                    in_flight_high_water_event_id = ?4,
                    in_flight_reason_mask = ?5,
                    pending_high_water_event_id = NULL,
                    pending_reason_mask = 0,
                    last_wake_reason_mask = ?5,
                    last_wake_at_ms = ?6
                WHERE epic_id = ?1
                "#,
            )
            .bind(epic_id)
            .bind(now_ms())
            .bind(row.in_flight_wake_id.clone())
            .bind(row.in_flight_high_water_event_id)
            .bind(row.in_flight_reason_mask as i64)
            .bind(row.last_wake_at_ms)
            .execute(&self.inner.pool)
            .await?;
        }

        let payload = self
            .build_payload(epic_id, scope.as_event_scope(), &row)
            .await?;
        if is_new {
            let payload_for_event = serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
                "wake_id": payload.summary.wake_id,
                "cursor": payload.summary.cursor,
                "high_water_event_id": payload.summary.high_water_event_id,
                "queue_size": payload.summary.queue_size,
                "chunk_count": payload.chunks.len(),
                "reasons": payload.summary.reasons,
                "last_wake_at_ms": payload.summary.last_wake_at_ms,
            }))
            .unwrap_or_default();
            let _ = self
                .inner
                .event_log
                .append_event(
                    scope.as_event_scope(),
                    DIRECTOR_WAKE_DISPATCHED_EVENT,
                    payload_for_event,
                )
                .await;
        }

        Ok((Some(payload), is_new))
    }

    async fn build_payload(
        &self,
        epic_id: EpicId,
        event_scope: EventScope,
        row: &WakeStateRow,
    ) -> Result<DirectorWakePayload, DirectorWakeError> {
        let wake_id = row
            .in_flight_wake_id
            .clone()
            .ok_or(DirectorWakeError::WakeNotInFlight { epic_id })?;
        let high_water = row
            .in_flight_high_water_event_id
            .ok_or(DirectorWakeError::WakeNotInFlight { epic_id })?;
        let high_water_rowid = self.event_rowid_in_scope(event_scope, high_water).await?;
        let ack_rowid = if let Some(cursor) = row.ack_cursor_event_id {
            Some(self.event_rowid_in_scope(event_scope, cursor).await?)
        } else {
            None
        };

        let events = self
            .load_events_window(epic_id, event_scope, ack_rowid, high_water_rowid)
            .await?;
        let chunks = chunk_events(
            events,
            self.inner.config.max_events_per_chunk.max(1),
            self.inner.config.max_chunk_bytes.max(1),
        );

        let summary = DirectorWakeSummary {
            epic_id,
            wake_id,
            cursor: row.ack_cursor_event_id,
            high_water_event_id: high_water,
            queue_size: chunks.iter().map(|chunk| chunk.events.len()).sum(),
            reasons: reasons_from_mask(row.in_flight_reason_mask),
            last_wake_reason: reasons_from_mask(row.last_wake_reason_mask),
            last_wake_at_ms: row.last_wake_at_ms,
        };

        Ok(DirectorWakePayload { summary, chunks })
    }

    async fn load_events_window(
        &self,
        epic_id: EpicId,
        event_scope: EventScope,
        ack_rowid: Option<i64>,
        high_water_rowid: i64,
    ) -> Result<Vec<DirectorWakeEvent>, DirectorWakeError> {
        let mut out = Vec::new();
        let mut after_rowid = ack_rowid;
        let mut task_cache: HashMap<TaskId, Option<EpicId>> = HashMap::new();
        let mut event_cache: HashMap<EventId, bool> = HashMap::new();

        loop {
            let page = redesmyn_storage::events::list_event_rows_in_scope_after_rowid(
                &self.inner.pool,
                event_scope,
                after_rowid,
                self.inner.config.backlog_page_size.max(1),
            )
            .await?;
            if page.is_empty() {
                break;
            }

            for row in &page {
                if row.rowid > high_water_rowid {
                    return Ok(out);
                }

                let is_relevant = if let Some(cached) = event_cache.get(&row.record.id) {
                    *cached
                } else {
                    let parsed = parse_json_payload(&row.record.payload);
                    let relevant = self
                        .event_matches_epic_with_task_cache(
                            epic_id,
                            &row.record,
                            parsed.as_ref(),
                            &mut task_cache,
                        )
                        .await?;
                    event_cache.insert(row.record.id, relevant);
                    relevant
                };

                if is_relevant {
                    out.push(DirectorWakeEvent {
                        event_id: row.record.id,
                        occurred_at_ms: row.record.created_at_ms,
                        event_type: row.record.kind.clone(),
                        json_payload: row.record.payload.clone(),
                    });
                }
            }

            after_rowid = page.last().map(|row| row.rowid);
            if after_rowid.is_some_and(|rowid| rowid >= high_water_rowid) {
                break;
            }
        }

        Ok(out)
    }

    async fn resume_cursor_for_subscription(
        &self,
        epic_id: EpicId,
        scope: EpicScope,
    ) -> Result<Option<EventId>, DirectorWakeError> {
        let row = self.load_state_row(epic_id).await?;
        let candidates = vec![
            row.ack_cursor_event_id,
            row.in_flight_high_water_event_id,
            row.pending_high_water_event_id,
        ];
        self.max_event_id_by_rowid(scope.as_event_scope(), candidates)
            .await
    }

    async fn load_state_row(&self, epic_id: EpicId) -> Result<WakeStateRow, DirectorWakeError> {
        let row: Option<(
            String,
            Option<EventId>,
            Option<String>,
            Option<EventId>,
            i64,
            Option<EventId>,
            i64,
            i64,
            Option<i64>,
        )> = sqlx::query_as(
            r#"
            SELECT
                director_mode,
                ack_cursor_event_id,
                in_flight_wake_id,
                in_flight_high_water_event_id,
                in_flight_reason_mask,
                pending_high_water_event_id,
                pending_reason_mask,
                last_wake_reason_mask,
                last_wake_at_ms
            FROM director_wake_state
            WHERE epic_id = ?1
            LIMIT 1
            "#,
        )
        .bind(epic_id)
        .fetch_optional(&self.inner.pool)
        .await?;

        let Some((
            director_mode,
            ack_cursor_event_id,
            in_flight_wake_id,
            in_flight_high_water_event_id,
            in_flight_reason_mask,
            pending_high_water_event_id,
            pending_reason_mask,
            last_wake_reason_mask,
            last_wake_at_ms,
        )) = row
        else {
            return Err(DirectorWakeError::EpicNotFound { epic_id });
        };

        Ok(WakeStateRow {
            director_mode: DirectorMode::from_db(&director_mode)?,
            ack_cursor_event_id,
            in_flight_wake_id,
            in_flight_high_water_event_id,
            in_flight_reason_mask: in_flight_reason_mask as u64,
            pending_high_water_event_id,
            pending_reason_mask: pending_reason_mask as u64,
            last_wake_reason_mask: last_wake_reason_mask as u64,
            last_wake_at_ms,
        })
    }

    async fn ensure_state_row(&self, epic_id: EpicId) -> Result<(), DirectorWakeError> {
        let exists = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT 1
            FROM epics
            WHERE id = ?1
            LIMIT 1
            "#,
        )
        .bind(epic_id)
        .fetch_optional(&self.inner.pool)
        .await?;
        if exists.is_none() {
            return Err(DirectorWakeError::EpicNotFound { epic_id });
        }

        sqlx::query(
            r#"
            INSERT OR IGNORE INTO director_wake_state (
                epic_id,
                created_at_ms,
                updated_at_ms,
                director_mode
            )
            VALUES (?1, ?2, ?2, 'paused')
            "#,
        )
        .bind(epic_id)
        .bind(now_ms())
        .execute(&self.inner.pool)
        .await?;

        Ok(())
    }

    async fn epic_scope(&self, epic_id: EpicId) -> Result<EpicScope, DirectorWakeError> {
        let row: Option<(redesmyn_ids::WorkspaceId, redesmyn_ids::RepoId)> = sqlx::query_as(
            r#"
            SELECT r.workspace_id, r.id
            FROM epics e
            JOIN repositories r ON r.id = e.repo_id
            WHERE e.id = ?1
            LIMIT 1
            "#,
        )
        .bind(epic_id)
        .fetch_optional(&self.inner.pool)
        .await?;
        let Some((workspace_id, repo_id)) = row else {
            return Err(DirectorWakeError::EpicNotFound { epic_id });
        };
        Ok(EpicScope {
            workspace_id,
            repo_id,
        })
    }

    async fn event_rowid_in_scope(
        &self,
        expected_scope: EventScope,
        event_id: EventId,
    ) -> Result<i64, DirectorWakeError> {
        let row =
            redesmyn_storage::events::get_event_rowid_and_scope(&self.inner.pool, event_id).await?;
        let Some((rowid, scope)) = row else {
            return Err(DirectorWakeError::Storage(StorageError::InvalidData {
                message: format!("missing event while resolving rowid: {event_id}"),
            }));
        };
        if scope != expected_scope {
            return Err(DirectorWakeError::Storage(StorageError::InvalidData {
                message: format!(
                    "event scope mismatch while resolving rowid: event_id={event_id} scope={scope:?} expected={expected_scope:?}"
                ),
            }));
        }
        Ok(rowid)
    }

    async fn event_rowid_and_record_in_scope(
        &self,
        expected_scope: EventScope,
        event_id: EventId,
    ) -> Result<Option<(i64, EventRecord)>, DirectorWakeError> {
        let row =
            redesmyn_storage::events::get_event_rowid_and_scope(&self.inner.pool, event_id).await?;
        let Some((rowid, scope)) = row else {
            return Ok(None);
        };
        if scope != expected_scope {
            return Ok(None);
        }

        let record = redesmyn_storage::events::get_event(&self.inner.pool, event_id).await?;
        Ok(record.map(|record| (rowid, record)))
    }

    async fn max_event_id_by_rowid(
        &self,
        scope: EventScope,
        candidates: Vec<Option<EventId>>,
    ) -> Result<Option<EventId>, DirectorWakeError> {
        let mut best: Option<(EventId, i64)> = None;
        for id in candidates.into_iter().flatten() {
            let rowid = self.event_rowid_in_scope(scope, id).await?;
            if best.is_none_or(|(_, best_rowid)| rowid > best_rowid) {
                best = Some((id, rowid));
            }
        }
        Ok(best.map(|(id, _)| id))
    }

    async fn newest_event_id(
        &self,
        scope: EventScope,
        a: Option<EventId>,
        b: Option<EventId>,
    ) -> Result<Option<EventId>, DirectorWakeError> {
        self.max_event_id_by_rowid(scope, vec![a, b]).await
    }

    async fn event_matches_epic(
        &self,
        epic_id: EpicId,
        event: &EventRecord,
        parsed: Option<&Value>,
    ) -> Result<bool, DirectorWakeError> {
        let mut cache = HashMap::new();
        self.event_matches_epic_with_task_cache(epic_id, event, parsed, &mut cache)
            .await
    }

    async fn event_matches_epic_with_task_cache(
        &self,
        epic_id: EpicId,
        event: &EventRecord,
        parsed: Option<&Value>,
        task_cache: &mut HashMap<TaskId, Option<EpicId>>,
    ) -> Result<bool, DirectorWakeError> {
        if event.kind.starts_with("director.wake.") {
            return Ok(false);
        }

        if let Some(payload_epic_id) = extract_epic_id(parsed) {
            return Ok(payload_epic_id == epic_id);
        }

        let task_id = extract_task_id(parsed);
        let Some(task_id) = task_id else {
            return Ok(false);
        };

        let cached = if let Some(value) = task_cache.get(&task_id) {
            *value
        } else {
            let row: Option<(EpicId,)> = sqlx::query_as(
                r#"
                SELECT epic_id
                FROM tasks
                WHERE id = ?1
                LIMIT 1
                "#,
            )
            .bind(task_id)
            .fetch_optional(&self.inner.pool)
            .await?;
            let value = row.map(|(value,)| value);
            task_cache.insert(task_id, value);
            value
        };
        Ok(cached == Some(epic_id))
    }

    async fn append_invalid_ack_event(&self, scope: EventScope, err: &DirectorWakeError) {
        let payload = serde_json::to_vec(&serde_json::json!({
            "error": err.to_string(),
        }))
        .unwrap_or_default();
        if let Err(append_err) = self
            .inner
            .event_log
            .append_event(scope, DIRECTOR_WAKE_ACK_REJECTED_EVENT, payload)
            .await
        {
            warn!(error = %append_err, "failed to append invalid ack observability event");
        }
    }
}

impl EpicScope {
    fn as_event_scope(&self) -> EventScope {
        EventScope::Repo {
            workspace_id: self.workspace_id,
            repo_id: self.repo_id,
        }
    }
}

fn parse_json_payload(payload: &[u8]) -> Option<Value> {
    serde_json::from_slice(payload).ok()
}

fn extract_epic_id(parsed: Option<&Value>) -> Option<EpicId> {
    parsed
        .and_then(|value| value.get("epic_id"))
        .and_then(Value::as_str)
        .and_then(|value| EpicId::from_str(value).ok())
}

fn extract_task_id(parsed: Option<&Value>) -> Option<TaskId> {
    let value = parsed?;
    let task_id = value
        .get("task_id")
        .or_else(|| value.get("target_task_id"))
        .and_then(Value::as_str)?;
    TaskId::from_str(task_id).ok()
}

fn significant_reason_for_event(kind: &str, parsed: Option<&Value>) -> Option<DirectorWakeReason> {
    if kind.starts_with("gate.") {
        return None;
    }

    if kind == TASK_SESSION_TURN_COMPLETED_EVENT {
        return Some(DirectorWakeReason::TaskTurnCompleted);
    }

    if matches!(
        kind,
        "command.succeeded" | "command.failed" | "command.canceled"
    ) {
        return Some(DirectorWakeReason::CommandOutcome);
    }

    if kind.starts_with("conductor.") {
        return Some(DirectorWakeReason::ConductorOverride);
    }

    if kind.starts_with("merge_queue.") || kind.starts_with("queue.") {
        let actor = parsed
            .and_then(|value| value.get("actor"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if actor != "director" {
            return Some(DirectorWakeReason::NonDirectorQueueUpdate);
        }
    }

    None
}

fn chunk_events(
    events: Vec<DirectorWakeEvent>,
    max_events_per_chunk: usize,
    max_chunk_bytes: usize,
) -> Vec<DirectorWakeChunk> {
    if events.is_empty() {
        return Vec::new();
    }

    let mut raw_chunks: Vec<Vec<DirectorWakeEvent>> = Vec::new();
    let mut current: Vec<DirectorWakeEvent> = Vec::new();
    let mut current_bytes: usize = 0;

    for event in events {
        let estimated_bytes = event.event_type.len() + event.json_payload.len() + 64;
        let would_overflow_count = current.len() >= max_events_per_chunk;
        let would_overflow_bytes =
            !current.is_empty() && current_bytes + estimated_bytes > max_chunk_bytes;

        if would_overflow_count || would_overflow_bytes {
            raw_chunks.push(current);
            current = Vec::new();
            current_bytes = 0;
        }

        current_bytes = current_bytes.saturating_add(estimated_bytes);
        current.push(event);
    }

    if !current.is_empty() {
        raw_chunks.push(current);
    }

    let total = raw_chunks.len() as u32;
    raw_chunks
        .into_iter()
        .enumerate()
        .map(|(idx, events)| DirectorWakeChunk {
            index: idx as u32,
            total,
            events,
        })
        .collect()
}
