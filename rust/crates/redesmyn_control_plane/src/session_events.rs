use tokio::sync::{broadcast, mpsc};

use prost::Message as _;
use redesmyn_ids::{EpicId, SessionEventId, SessionId, TaskId};
use redesmyn_logging::tracing::{Instrument as _, debug, warn};
use sqlx::SqlitePool;

use redesmyn_protocol::client::{SessionEventCursor, SessionEventKindFilter};
use redesmyn_protocol::pb::redesmyn::protocol::v1 as pbv1;
use redesmyn_protocol::session::{SessionEventKind, SessionScope, UnknownSessionEvent};
use redesmyn_protocol::{SessionEvent, SessionLiveEvent, Timestamp};
use redesmyn_storage::StorageError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionEventsResyncReason {
    /// The subscriber's receiver fell behind the hub buffer.
    Lagged,
    /// The cursor does not exist in the DB.
    CursorNotFound,
    /// The cursor exists, but is in a different session than the subscription.
    CursorSessionMismatch,
    /// The subscription failed to resume from the DB cursor due to a storage error.
    DbError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEventsResync {
    pub reason: SessionEventsResyncReason,
    pub resume_after: Option<SessionEventCursor>,
    pub dropped_events: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum SessionEventsSubscriptionItem {
    Event(SessionEvent),
    Live(SessionLiveEvent),
    ResyncRequired(SessionEventsResync),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionEventsConfig {
    /// Broadcast buffer size for the in-process subscription hub.
    ///
    /// Drop policy: when the hub buffer overflows, lagging subscribers observe a `Lagged` error and
    /// must resubscribe with a DB cursor to resync.
    pub hub_buffer: usize,
    /// Broadcast buffer size for the live-only session event hub.
    pub live_hub_buffer: usize,
    /// Per-subscriber channel size between the hub task and the consumer.
    pub subscription_buffer: usize,
    /// Page size used when replaying historical events from the DB on subscribe/resume.
    pub backlog_page_size: usize,
}

impl Default for SessionEventsConfig {
    fn default() -> Self {
        Self {
            hub_buffer: 1024,
            live_hub_buffer: 512,
            subscription_buffer: 128,
            backlog_page_size: 256,
        }
    }
}

#[derive(Clone)]
pub struct SessionEvents {
    pool: SqlitePool,
    hub: broadcast::Sender<SessionEvent>,
    live_hub: broadcast::Sender<SessionLiveEvent>,
    config: SessionEventsConfig,
}

impl SessionEvents {
    #[must_use]
    pub(crate) fn new_with_config(pool: SqlitePool, config: SessionEventsConfig) -> Self {
        let (hub, _rx) = broadcast::channel(config.hub_buffer.max(1));
        let (live_hub, _rx) = broadcast::channel(config.live_hub_buffer.max(1));
        Self {
            pool,
            hub,
            live_hub,
            config,
        }
    }

    #[must_use]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub async fn get_session_events(
        &self,
        session_id: SessionId,
        before: Option<SessionEventCursor>,
        limit: u32,
        kinds: &[SessionEventKindFilter],
    ) -> Result<(Vec<SessionEvent>, Option<SessionEventCursor>), StorageError> {
        let limit = limit.max(1);
        let page_limit = limit.saturating_add(1);

        let before_ms = before.map(|cursor| ms_from_timestamp(cursor.created_at));
        let before_id = before.map(|cursor| cursor.session_event_id);
        let kind_values = kinds_to_db_values(kinds);

        let mut query = sqlx::QueryBuilder::new(
            r#"
            SELECT
                id,
                session_id,
                created_at_ms,
                scope_kind,
                task_id,
                kind,
                turn_id,
                payload
            FROM session_events
            WHERE session_id =
            "#,
        );
        query.push_bind(session_id);

        if let (Some(before_ms), Some(before_id)) = (before_ms, before_id) {
            query.push(
                r#"
                AND (
                    created_at_ms < 
                "#,
            );
            query.push_bind(before_ms);
            query.push(
                r#"
                    OR (
                        created_at_ms = 
                "#,
            );
            query.push_bind(before_ms);
            query.push(
                r#"
                        AND id < 
                "#,
            );
            query.push_bind(before_id);
            query.push("))");
        }

        if !kind_values.is_empty() {
            query.push(" AND kind IN (");
            let mut separated = query.separated(", ");
            for value in kind_values {
                separated.push_bind(value);
            }
            query.push(")");
        }

        query.push(
            r#"
            ORDER BY created_at_ms DESC, id DESC
            LIMIT 
            "#,
        );
        query.push_bind(page_limit as i64);

        let mut rows = query
            .build_query_as::<SessionEventRow>()
            .fetch_all(&self.pool)
            .await?;

        let has_more = rows.len() > limit as usize;
        if has_more {
            rows.truncate(limit as usize);
        }

        let next_cursor = if has_more {
            rows.last().map(cursor_from_row)
        } else {
            None
        };

        rows.reverse();
        let events = rows.into_iter().map(SessionEventRow::into_event).collect();
        Ok((events, next_cursor))
    }

    pub async fn get_latest_task_session(
        &self,
        task_id: TaskId,
    ) -> Result<Option<SessionId>, StorageError> {
        let result = sqlx::query_scalar::<_, SessionId>(
            r#"
            SELECT session_id
            FROM session_events
            WHERE task_id = ?1
            ORDER BY created_at_ms DESC, id DESC
            LIMIT 1
            "#,
        )
        .bind(task_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(result)
    }

    pub async fn get_epic_pinned_chat_session(
        &self,
        epic_id: EpicId,
    ) -> Result<Option<SessionId>, StorageError> {
        let result = sqlx::query_scalar::<_, SessionId>(
            r#"
            SELECT session_id
            FROM session_events
            WHERE epic_id = ?1
              AND scope_kind = 'epic'
            ORDER BY created_at_ms DESC, id DESC
            LIMIT 1
            "#,
        )
        .bind(epic_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(result)
    }

    #[must_use]
    pub fn subscribe(
        &self,
        session_id: SessionId,
        after: Option<SessionEventCursor>,
    ) -> SessionEventsSubscription {
        let (tx, rx) = mpsc::channel(self.config.subscription_buffer.max(1));
        let pool = self.pool.clone();
        let mut hub_rx = self.hub.subscribe();
        let mut live_hub_rx = self.live_hub.subscribe();
        let config = self.config;

        tokio::spawn(
            async move {
                let mut last_cursor = after;

                enum ResumeStatus {
                    Continue,
                    Terminate,
                }

                if let Some(cursor) = after {
                    let span = redesmyn_logging::tracing::debug_span!(
                        "control_plane.session_events.subscribe.resume",
                        session_id = %session_id,
                        after_session_event_id = %cursor.session_event_id
                    );

                    let result: Result<ResumeStatus, StorageError> = async {
                        let Some(cursor_row) =
                            session_event_cursor_row(&pool, cursor.session_event_id).await?
                        else {
                            debug!("cursor not found; resync required");
                            let _ = tx
                                .send(SessionEventsSubscriptionItem::ResyncRequired(
                                    SessionEventsResync {
                                        reason: SessionEventsResyncReason::CursorNotFound,
                                        resume_after: None,
                                        dropped_events: None,
                                    },
                                ))
                                .await;
                            return Ok(ResumeStatus::Terminate);
                        };

                        if cursor_row.session_id != session_id {
                            debug!(
                                cursor_session_id = %cursor_row.session_id,
                                subscribe_session_id = %session_id,
                                "cursor session mismatch; resync required"
                            );
                            let _ = tx
                                .send(SessionEventsSubscriptionItem::ResyncRequired(
                                    SessionEventsResync {
                                        reason: SessionEventsResyncReason::CursorSessionMismatch,
                                        resume_after: None,
                                        dropped_events: None,
                                    },
                                ))
                                .await;
                            return Ok(ResumeStatus::Terminate);
                        }

                        loop {
                            let page = list_session_event_rows_after_cursor(
                                &pool,
                                session_id,
                                last_cursor,
                                config.backlog_page_size.max(1),
                            )
                            .await?;

                            if page.is_empty() {
                                break;
                            }

                            for row in page {
                                let cursor = cursor_from_row(&row);
                                last_cursor = Some(cursor);

                                if tx
                                    .send(SessionEventsSubscriptionItem::Event(row.into_event()))
                                    .await
                                    .is_err()
                                {
                                    return Ok(ResumeStatus::Terminate);
                                }
                            }
                        }

                        Ok(ResumeStatus::Continue)
                    }
                    .instrument(span)
                    .await;

                    match result {
                        Ok(ResumeStatus::Continue) => {}
                        Ok(ResumeStatus::Terminate) => return,
                        Err(err) => {
                            warn!(error = %err, "failed to resume subscription from cursor");
                            let _ = tx
                                .send(SessionEventsSubscriptionItem::ResyncRequired(
                                    SessionEventsResync {
                                        reason: SessionEventsResyncReason::DbError,
                                        resume_after: last_cursor,
                                        dropped_events: None,
                                    },
                                ))
                                .await;
                            return;
                        }
                    }
                }

                loop {
                    tokio::select! {
                        biased;
                        recv = hub_rx.recv() => match recv {
                            Ok(event) => {
                                if event.session_id != session_id {
                                    continue;
                                }

                                let cursor = SessionEventCursor {
                                    created_at: event.created_at,
                                    session_event_id: event.session_event_id,
                                };

                                if last_cursor.is_some_and(|last| cursor <= last) {
                                    continue;
                                }

                                last_cursor = Some(cursor);

                                if tx
                                    .send(SessionEventsSubscriptionItem::Event(event))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                                debug!(
                                    skipped,
                                    resume_after_session_event_id =
                                        last_cursor.map(|c| c.session_event_id.to_string()),
                                    "subscription fell behind; resync required"
                                );
                                let _ = tx
                                    .send(SessionEventsSubscriptionItem::ResyncRequired(
                                        SessionEventsResync {
                                            reason: SessionEventsResyncReason::Lagged,
                                            resume_after: last_cursor,
                                            dropped_events: Some(skipped),
                                        },
                                    ))
                                    .await;
                                return;
                            }
                            Err(broadcast::error::RecvError::Closed) => return,
                        },
                        recv = live_hub_rx.recv() => match recv {
                            Ok(event) => {
                                if event.session_id != session_id {
                                    continue;
                                }

                                match tx.try_send(SessionEventsSubscriptionItem::Live(event)) {
                                    Ok(()) => {}
                                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => return,
                                    Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                                        // Best-effort: drop live events under backpressure rather
                                        // than stalling durable session event delivery.
                                    }
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                                debug!(skipped, "live subscription fell behind; dropping deltas");
                            }
                            Err(broadcast::error::RecvError::Closed) => return,
                        },
                    }
                }
            }
            .instrument(redesmyn_logging::tracing::debug_span!(
                "control_plane.session_events.subscribe",
                session_id = %session_id
            )),
        );

        SessionEventsSubscription { rx }
    }

    pub fn publish_live_event(&self, event: SessionLiveEvent) {
        let _ = self.live_hub.send(event);
    }

    pub async fn append_session_event(&self, event: &SessionEvent) -> Result<(), StorageError> {
        let payload = event.to_protobuf().encode_to_vec();

        let (scope_kind, epic_id, task_id) = match event.scope {
            SessionScope::Task { task_id } => ("task", None, Some(task_id)),
            SessionScope::Chat => ("repo", None, None),
            SessionScope::Unknown => ("none", None, None),
            _ => ("none", None, None),
        };

        let (workspace_id, repo_id, resolved_epic_id) = match event.scope {
            SessionScope::Task { task_id } => resolve_task_scope(&self.pool, task_id).await?,
            SessionScope::Chat => {
                let (workspace_id, repo_id) =
                    resolve_chat_scope(&self.pool, event.session_id).await?;
                (Some(workspace_id), Some(repo_id), None)
            }
            _ => (None, None, None),
        };

        let epic_id = epic_id.or(resolved_epic_id);
        let message_preview = message_preview_from_kind(&event.kind);
        let artifact_id = artifact_id_from_kind(&event.kind);

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
        .bind(event.session_event_id)
        .bind(event.session_id)
        .bind(ms_from_timestamp(event.created_at))
        .bind(scope_kind)
        .bind(workspace_id)
        .bind(repo_id)
        .bind(epic_id)
        .bind(task_id)
        .bind(kind_db_value_from_kind(&event.kind))
        .bind(event.turn_id.clone())
        .bind(message_preview)
        .bind(artifact_id)
        .bind(payload)
        .execute(&self.pool)
        .await?;

        let _ = self.hub.send(event.clone());
        Ok(())
    }
}

pub struct SessionEventsSubscription {
    rx: mpsc::Receiver<SessionEventsSubscriptionItem>,
}

impl SessionEventsSubscription {
    pub async fn recv(&mut self) -> Option<SessionEventsSubscriptionItem> {
        self.rx.recv().await
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct SessionEventRow {
    id: SessionEventId,
    session_id: SessionId,
    created_at_ms: i64,
    scope_kind: String,
    task_id: Option<TaskId>,
    kind: String,
    turn_id: Option<String>,
    payload: Vec<u8>,
}

impl SessionEventRow {
    fn into_event(self) -> SessionEvent {
        let created_at = timestamp_from_ms(self.created_at_ms);

        if !self.payload.is_empty() {
            if let Ok(proto) = pbv1::SessionEvent::decode(&*self.payload) {
                if let Ok(event) = SessionEvent::try_from_protobuf(proto) {
                    if event.session_event_id == self.id && event.session_id == self.session_id {
                        return event;
                    }
                    debug!(
                        row_session_event_id = %self.id,
                        decoded_session_event_id = %event.session_event_id,
                        row_session_id = %self.session_id,
                        decoded_session_id = %event.session_id,
                        "decoded session event does not match row; using fallback"
                    );
                }
            }
        }

        let scope = match (self.scope_kind.as_str(), self.task_id) {
            ("task", Some(task_id)) => SessionScope::Task { task_id },
            ("none" | "repo" | "epic", _) => SessionScope::Chat,
            _ => SessionScope::Unknown,
        };

        SessionEvent {
            session_event_id: self.id,
            created_at,
            scope,
            session_id: self.session_id,
            turn_id: self.turn_id,
            kind: SessionEventKind::Unknown(UnknownSessionEvent {
                event_type: self.kind,
                json_payload: Vec::new(),
            }),
        }
    }
}

fn timestamp_from_ms(ms: i64) -> Timestamp {
    let nanos = i128::from(ms).saturating_mul(1_000_000);
    let datetime = time::OffsetDateTime::from_unix_timestamp_nanos(nanos)
        .unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
    Timestamp::from_offset_date_time(datetime)
}

fn ms_from_timestamp(value: Timestamp) -> i64 {
    let nanos = value.into_offset_date_time().unix_timestamp_nanos();
    let ms = nanos / 1_000_000;
    i64::try_from(ms).unwrap_or_else(|_| if ms.is_negative() { i64::MIN } else { i64::MAX })
}

fn cursor_from_row(row: &SessionEventRow) -> SessionEventCursor {
    SessionEventCursor {
        created_at: timestamp_from_ms(row.created_at_ms),
        session_event_id: row.id,
    }
}

fn kinds_to_db_values(kinds: &[SessionEventKindFilter]) -> Vec<&'static str> {
    kinds
        .iter()
        .filter_map(|kind| match kind {
            SessionEventKindFilter::SessionStarted => Some("session_started"),
            SessionEventKindFilter::SessionEnded => Some("session_ended"),
            SessionEventKindFilter::TurnStarted => Some("turn_started"),
            SessionEventKindFilter::TurnCompleted => Some("turn_completed"),
            SessionEventKindFilter::UserMessage => Some("user_message"),
            SessionEventKindFilter::AssistantMessage => Some("assistant_message"),
            SessionEventKindFilter::ToolInvocation => Some("tool_invocation"),
            SessionEventKindFilter::ToolResult => Some("tool_result"),
            SessionEventKindFilter::StatusUpdate => Some("status_update"),
            SessionEventKindFilter::ArtifactEmitted => Some("artifact_emitted"),
            SessionEventKindFilter::Unknown => None,
        })
        .collect()
}

fn kind_db_value_from_kind(kind: &SessionEventKind) -> &'static str {
    match kind {
        SessionEventKind::SessionStarted(_) => "session_started",
        SessionEventKind::SessionEnded(_) => "session_ended",
        SessionEventKind::TurnStarted(_) => "turn_started",
        SessionEventKind::TurnCompleted(_) => "turn_completed",
        SessionEventKind::UserMessage(_) => "user_message",
        SessionEventKind::AssistantMessage(_) => "assistant_message",
        SessionEventKind::ToolInvocation(_) => "tool_invocation",
        SessionEventKind::ToolResult(_) => "tool_result",
        SessionEventKind::StatusUpdate(_) => "status_update",
        SessionEventKind::ArtifactEmitted(_) => "artifact_emitted",
        SessionEventKind::Unknown(_) => "unknown",
    }
}

fn message_preview_from_kind(kind: &SessionEventKind) -> Option<String> {
    match kind {
        SessionEventKind::UserMessage(ev) => Some(ev.preview.clone()),
        SessionEventKind::AssistantMessage(ev) => Some(ev.preview.clone()),
        SessionEventKind::ToolInvocation(ev) => Some(ev.input_preview.clone()),
        SessionEventKind::ToolResult(ev) => Some(ev.output_preview.clone()),
        SessionEventKind::StatusUpdate(ev) => ev.message.clone(),
        _ => None,
    }
}

fn artifact_id_from_kind(kind: &SessionEventKind) -> Option<redesmyn_ids::ArtifactId> {
    match kind {
        SessionEventKind::UserMessage(ev) => ev.full_text_artifact.as_ref().map(|a| a.artifact_id),
        SessionEventKind::AssistantMessage(ev) => {
            ev.full_text_artifact.as_ref().map(|a| a.artifact_id)
        }
        SessionEventKind::ToolInvocation(ev) => ev.input_artifact.as_ref().map(|a| a.artifact_id),
        SessionEventKind::ToolResult(ev) => ev.output_artifact.as_ref().map(|a| a.artifact_id),
        SessionEventKind::ArtifactEmitted(ev) => Some(ev.artifact.artifact_id),
        _ => None,
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct CursorRow {
    session_id: SessionId,
}

async fn session_event_cursor_row(
    pool: &SqlitePool,
    session_event_id: SessionEventId,
) -> Result<Option<CursorRow>, StorageError> {
    let row = sqlx::query_as::<_, CursorRow>(
        r#"
        SELECT session_id
        FROM session_events
        WHERE id = ?1
        "#,
    )
    .bind(session_event_id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

async fn list_session_event_rows_after_cursor(
    pool: &SqlitePool,
    session_id: SessionId,
    after: Option<SessionEventCursor>,
    limit: usize,
) -> Result<Vec<SessionEventRow>, StorageError> {
    let after_ms = after.map(|cursor| ms_from_timestamp(cursor.created_at));
    let after_id = after.map(|cursor| cursor.session_event_id);

    let mut query = sqlx::QueryBuilder::new(
        r#"
        SELECT
            id,
            session_id,
            created_at_ms,
            scope_kind,
            task_id,
            kind,
            turn_id,
            payload
        FROM session_events
        WHERE session_id =
        "#,
    );
    query.push_bind(session_id);

    if let (Some(after_ms), Some(after_id)) = (after_ms, after_id) {
        query.push(
            r#"
            AND (
                created_at_ms >
            "#,
        );
        query.push_bind(after_ms);
        query.push(
            r#"
                OR (
                    created_at_ms =
            "#,
        );
        query.push_bind(after_ms);
        query.push(
            r#"
                    AND id >
            "#,
        );
        query.push_bind(after_id);
        query.push("))");
    }

    query.push(
        r#"
        ORDER BY created_at_ms ASC, id ASC
        LIMIT
        "#,
    );
    query.push_bind(i64::try_from(limit).unwrap_or(i64::MAX));

    Ok(query
        .build_query_as::<SessionEventRow>()
        .fetch_all(pool)
        .await?)
}

async fn resolve_task_scope(
    pool: &SqlitePool,
    task_id: TaskId,
) -> Result<
    (
        Option<redesmyn_ids::WorkspaceId>,
        Option<redesmyn_ids::RepoId>,
        Option<EpicId>,
    ),
    StorageError,
> {
    #[derive(Debug, Clone, sqlx::FromRow)]
    struct Row {
        workspace_id: redesmyn_ids::WorkspaceId,
        repo_id: redesmyn_ids::RepoId,
        epic_id: EpicId,
    }

    let row = sqlx::query_as::<_, Row>(
        r#"
        SELECT
            repositories.workspace_id as workspace_id,
            repositories.id as repo_id,
            tasks.epic_id as epic_id
        FROM tasks
        JOIN epics ON epics.id = tasks.epic_id
        JOIN repositories ON repositories.id = epics.repo_id
        WHERE tasks.id = ?1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await?;

    Ok(row
        .map(|row| (Some(row.workspace_id), Some(row.repo_id), Some(row.epic_id)))
        .unwrap_or((None, None, None)))
}

async fn resolve_chat_scope(
    pool: &SqlitePool,
    session_id: SessionId,
) -> Result<(redesmyn_ids::WorkspaceId, redesmyn_ids::RepoId), StorageError> {
    #[derive(Debug, Clone, sqlx::FromRow)]
    struct Row {
        workspace_id: redesmyn_ids::WorkspaceId,
        repo_id: redesmyn_ids::RepoId,
    }

    let row = sqlx::query_as::<_, Row>(
        r#"
        SELECT
            scope_workspace_id as workspace_id,
            scope_repo_id as repo_id
        FROM agent_sessions
        WHERE session_id = ?1
        "#,
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await?;

    let Some(row) = row else {
        return Err(StorageError::InvalidData {
            message: format!("session not found while resolving chat scope: {session_id}"),
        });
    };

    Ok((row.workspace_id, row.repo_id))
}
