use tokio::sync::{broadcast, mpsc};

use redesmyn_ids::EventId;
use redesmyn_logging::tracing::{Instrument as _, debug, warn};
use sqlx::SqlitePool;

use redesmyn_storage::events::{EventRecord, EventRow, EventScope};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventLogResyncReason {
    /// The subscriber's receiver fell behind the hub buffer.
    Lagged,
    /// The `after_event_id` cursor does not exist in the DB.
    CursorNotFound,
    /// The `after_event_id` exists, but is in a different scope than the subscription.
    CursorScopeMismatch,
    /// The subscription failed to resume from the DB cursor due to a storage error.
    DbError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventLogResync {
    pub reason: EventLogResyncReason,
    pub resume_after_event_id: Option<EventId>,
    pub dropped_events: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventLogSubscriptionItem {
    Event(EventRecord),
    ResyncRequired(EventLogResync),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventLogConfig {
    /// Broadcast buffer size for the in-process subscription hub.
    ///
    /// Drop policy: when the hub buffer overflows, lagging subscribers observe a `Lagged` error and
    /// must resubscribe with a DB cursor to resync.
    pub hub_buffer: usize,
    /// Per-subscriber channel size between the hub task and the consumer.
    pub subscription_buffer: usize,
    /// Page size used when replaying historical events from the DB on subscribe/resume.
    pub backlog_page_size: usize,
}

impl Default for EventLogConfig {
    fn default() -> Self {
        Self {
            hub_buffer: 1024,
            subscription_buffer: 128,
            backlog_page_size: 512,
        }
    }
}

#[derive(Clone)]
pub struct EventLog {
    pool: SqlitePool,
    hub: broadcast::Sender<EventRow>,
    config: EventLogConfig,
}

impl EventLog {
    #[must_use]
    pub(crate) fn new_with_config(pool: SqlitePool, config: EventLogConfig) -> Self {
        let (hub, _rx) = broadcast::channel(config.hub_buffer.max(1));
        Self { pool, hub, config }
    }

    pub async fn append_event(
        &self,
        scope: EventScope,
        kind: impl Into<String>,
        payload: Vec<u8>,
    ) -> Result<EventId, redesmyn_storage::StorageError> {
        let id = EventId::new();
        let record = EventRecord::new_now_in_scope(id, scope, kind, payload);

        async {
            redesmyn_storage::events::insert_event(&self.pool, &record).await?;
            let (rowid, _) = redesmyn_storage::events::get_event_rowid_and_scope(&self.pool, id)
                .await?
                .ok_or_else(|| redesmyn_storage::StorageError::InvalidData {
                    message: format!("inserted event is missing rowid: {id}"),
                })?;
            let _ = self.hub.send(EventRow { rowid, record });
            Ok(id)
        }
        .instrument(redesmyn_logging::tracing::debug_span!(
            "control_plane.event_log.append_event",
            event_id = %id
        ))
        .await
    }

    #[must_use]
    pub fn subscribe(
        &self,
        scope: EventScope,
        after_event_id: Option<EventId>,
    ) -> EventLogSubscription {
        let (tx, rx) = mpsc::channel(self.config.subscription_buffer.max(1));
        let pool = self.pool.clone();
        let mut hub_rx = self.hub.subscribe();
        let config = self.config;

        tokio::spawn(
            async move {
                let mut after_rowid: Option<i64> = None;
                let mut last_event_id: Option<EventId> = None;

                enum ResumeStatus {
                    Continue,
                    Terminate,
                }

                if let Some(after_event_id) = after_event_id {
                    last_event_id = Some(after_event_id);
                    let span = redesmyn_logging::tracing::debug_span!(
                        "control_plane.event_log.subscribe.resume",
                        after_event_id = %after_event_id
                    );

                    let result: Result<ResumeStatus, redesmyn_storage::StorageError> = async {
                        let Some((rowid, cursor_scope)) =
                            redesmyn_storage::events::get_event_rowid_and_scope(
                                &pool,
                                after_event_id,
                            )
                            .await?
                        else {
                            debug!("cursor not found; resync required");
                            let _ = tx
                                .send(EventLogSubscriptionItem::ResyncRequired(EventLogResync {
                                    reason: EventLogResyncReason::CursorNotFound,
                                    resume_after_event_id: None,
                                    dropped_events: None,
                                }))
                                .await;
                            return Ok(ResumeStatus::Terminate);
                        };

                        if cursor_scope != scope {
                            debug!(
                                cursor_scope = ?cursor_scope,
                                subscribe_scope = ?scope,
                                "cursor scope mismatch; resync required"
                            );
                            let _ = tx
                                .send(EventLogSubscriptionItem::ResyncRequired(EventLogResync {
                                    reason: EventLogResyncReason::CursorScopeMismatch,
                                    resume_after_event_id: None,
                                    dropped_events: None,
                                }))
                                .await;
                            return Ok(ResumeStatus::Terminate);
                        }

                        after_rowid = Some(rowid);

                        loop {
                            let page =
                                redesmyn_storage::events::list_event_rows_in_scope_after_rowid(
                                    &pool,
                                    scope,
                                    after_rowid,
                                    config.backlog_page_size.max(1),
                                )
                                .await?;
                            if page.is_empty() {
                                break;
                            }

                            for row in page {
                                let event_id = row.record.id;

                                if tx
                                    .send(EventLogSubscriptionItem::Event(row.record))
                                    .await
                                    .is_err()
                                {
                                    return Ok(ResumeStatus::Terminate);
                                }

                                after_rowid = Some(row.rowid);
                                last_event_id = Some(event_id);
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
                                .send(EventLogSubscriptionItem::ResyncRequired(EventLogResync {
                                    reason: EventLogResyncReason::DbError,
                                    resume_after_event_id: last_event_id,
                                    dropped_events: None,
                                }))
                                .await;
                            return;
                        }
                    }
                }

                loop {
                    match hub_rx.recv().await {
                        Ok(row) => {
                            if row.record.scope != scope {
                                continue;
                            }

                            if after_rowid.is_some_and(|after| row.rowid <= after) {
                                continue;
                            }

                            after_rowid = Some(row.rowid);
                            last_event_id = Some(row.record.id);

                            if tx
                                .send(EventLogSubscriptionItem::Event(row.record))
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            debug!(
                                skipped,
                                resume_after_event_id = last_event_id.map(|id| id.to_string()),
                                "subscription fell behind; resync required"
                            );
                            let _ = tx
                                .send(EventLogSubscriptionItem::ResyncRequired(EventLogResync {
                                    reason: EventLogResyncReason::Lagged,
                                    resume_after_event_id: last_event_id,
                                    dropped_events: Some(skipped),
                                }))
                                .await;
                            return;
                        }
                        Err(broadcast::error::RecvError::Closed) => return,
                    }
                }
            }
            .instrument(redesmyn_logging::tracing::debug_span!(
                "control_plane.event_log.subscribe",
                scope = ?scope
            )),
        );

        EventLogSubscription { rx }
    }
}

pub struct EventLogSubscription {
    rx: mpsc::Receiver<EventLogSubscriptionItem>,
}

impl EventLogSubscription {
    pub async fn recv(&mut self) -> Option<EventLogSubscriptionItem> {
        self.rx.recv().await
    }
}
