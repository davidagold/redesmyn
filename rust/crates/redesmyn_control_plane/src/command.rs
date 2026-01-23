use std::time::Duration;

use tokio::sync::broadcast;

use redesmyn_ids::{CommandId, CommandUpdateId, TaskId};
use redesmyn_logging::tracing::warn;
use sqlx::SqlitePool;

use redesmyn_protocol::client::{CommandState, CommandSummary, CommandUpdateSummary};
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope, Timestamp};
use redesmyn_storage::commands::{CommandRecord, CommandScope, CommandUpdateRecord};
use redesmyn_storage::schema::CommandState as StorageCommandState;

use crate::error::ControlPlaneError;
use crate::event_log::EventLog;

const COMMAND_UPDATES_HUB_BUFFER: usize = 256;

#[derive(Clone)]
pub struct Commands {
    pool: SqlitePool,
    event_log: EventLog,
    updates_tx: broadcast::Sender<CommandUpdateNotice>,
}

#[derive(Debug, Clone)]
pub struct CreateCommandResult {
    pub command: CommandSummary,
    pub created_new: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CommandUpdateNotice {
    command_id: CommandId,
    state: StorageCommandState,
}

#[derive(Debug, serde::Serialize)]
struct CommandUpdateEventPayload {
    command_id: CommandId,
    update_id: CommandUpdateId,
    state: String,
    message: Option<String>,
}

impl Commands {
    #[must_use]
    pub(crate) fn new(pool: SqlitePool, event_log: EventLog) -> Self {
        let (updates_tx, _rx) = broadcast::channel(COMMAND_UPDATES_HUB_BUFFER.max(1));
        Self {
            pool,
            event_log,
            updates_tx,
        }
    }

    pub async fn create_command(
        &self,
        scope: CommandScope,
        kind: String,
        target_task_id: Option<TaskId>,
        idempotency_key: Option<String>,
        created_by: Option<String>,
        payload: Vec<u8>,
    ) -> Result<CreateCommandResult, ControlPlaneError> {
        if kind.trim().is_empty() {
            return Err(ControlPlaneError::InvalidCommandKind { kind });
        }
        if idempotency_key.as_ref().is_some_and(|key| key.trim().is_empty()) {
            let idempotency_key = idempotency_key.clone().unwrap_or_default();
            return Err(ControlPlaneError::InvalidIdempotencyKey {
                idempotency_key,
            });
        }

        enum CreateResult {
            Existing {
                command: CommandRecord,
                last_update: Option<CommandUpdateRecord>,
            },
            IdempotencyConflict {
                existing_command_id: CommandId,
            },
            New {
                command: CommandRecord,
                update: CommandUpdateRecord,
            },
        }

        let kind_clone = kind.clone();
        let idempotency_key_clone = idempotency_key.clone();
        let scope_clone = scope;

        let result: CreateResult = redesmyn_storage::in_transaction(&self.pool, |conn| {
            Box::pin(async move {
                if let Some(key) = idempotency_key_clone.as_deref() {
                    if let Some(existing_id) = redesmyn_storage::commands::find_command_by_idempotency_key(
                        &mut *conn,
                        scope_clone,
                        &kind_clone,
                        key,
                    )
                    .await?
                    {
                        let Some(existing) =
                            redesmyn_storage::commands::get_command(&mut *conn, existing_id)
                                .await?
                        else {
                            return Err(redesmyn_storage::StorageError::InvalidData {
                                message: format!(
                                    "command exists for idempotency_key but row is missing: {existing_id}"
                                ),
                            });
                        };
                        let last_update =
                            redesmyn_storage::commands::get_command_last_update(
                                &mut *conn,
                                existing_id,
                            )
                            .await?;

                        if existing.payload != payload {
                            return Ok(CreateResult::IdempotencyConflict {
                                existing_command_id: existing_id,
                            });
                        }

                        return Ok(CreateResult::Existing {
                            command: existing,
                            last_update,
                        });
                    }
                }

                let command_id = CommandId::new();
                let mut command = CommandRecord::new_now(
                    command_id,
                    scope_clone,
                    kind_clone,
                    StorageCommandState::Queued,
                    target_task_id,
                    idempotency_key_clone,
                    created_by,
                    payload,
                );

                redesmyn_storage::commands::insert_command(&mut *conn, &command).await?;

                let update = CommandUpdateRecord::new_now(
                    CommandUpdateId::new(),
                    command_id,
                    StorageCommandState::Queued,
                    None,
                    None,
                    None,
                    None,
                );

                redesmyn_storage::commands::insert_command_update(&mut *conn, &update).await?;
                redesmyn_storage::commands::update_command_state(
                    &mut *conn,
                    command_id,
                    update.state,
                    update.created_at_ms,
                )
                .await?;

                command.updated_at_ms = update.created_at_ms;

                Ok(CreateResult::New { command, update })
            })
        })
        .await?;

        match result {
            CreateResult::Existing { command, last_update } => {
                let last_update = last_update
                    .map(command_update_summary_from_record)
                    .transpose()?;
                Ok(CreateCommandResult {
                    command: command_summary_from_records(command, last_update)?,
                    created_new: false,
                })
            }
            CreateResult::IdempotencyConflict {
                existing_command_id,
            } => Err(ControlPlaneError::CommandIdempotencyPayloadMismatch {
                idempotency_key: idempotency_key.unwrap_or_default(),
                existing_command_id,
            }),
            CreateResult::New { command, update } => {
                self.publish_update(&command, &update).await;
                let last_update = Some(command_update_summary_from_record(update)?);
                Ok(CreateCommandResult {
                    command: command_summary_from_records(command, last_update)?,
                    created_new: true,
                })
            }
        }
    }

    pub async fn get_command(
        &self,
        command_id: CommandId,
    ) -> Result<Option<CommandSummary>, ControlPlaneError> {
        let Some(command) = redesmyn_storage::commands::get_command(&self.pool, command_id).await?
        else {
            return Ok(None);
        };

        let last_update = redesmyn_storage::commands::get_command_last_update(&self.pool, command_id)
            .await?
            .map(command_update_summary_from_record)
            .transpose()?;

        Ok(Some(command_summary_from_records(command, last_update)?))
    }

    pub async fn append_update(
        &self,
        command_id: CommandId,
        next_state: StorageCommandState,
        message: Option<String>,
        progress_current: Option<i64>,
        progress_total: Option<i64>,
        detail: Option<Vec<u8>>,
    ) -> Result<CommandSummary, ControlPlaneError> {
        if redesmyn_storage::commands::get_command(&self.pool, command_id)
            .await?
            .is_none()
        {
            return Err(ControlPlaneError::CommandNotFound { command_id });
        }

        let (command, update) = redesmyn_storage::in_transaction(&self.pool, |conn| {
            Box::pin(async move {
                let Some(mut command) =
                    redesmyn_storage::commands::get_command(&mut *conn, command_id).await?
                else {
                    return Err(redesmyn_storage::StorageError::InvalidData {
                        message: format!("command disappeared during update: {command_id}"),
                    });
                };

                validate_transition(command.state, next_state).map_err(|_| {
                    redesmyn_storage::StorageError::InvalidData {
                        message: format!(
                            "invalid command state transition: {command_id} {} -> {}",
                            command.state.as_str(),
                            next_state.as_str()
                        ),
                    }
                })?;

                let update = CommandUpdateRecord::new_now(
                    CommandUpdateId::new(),
                    command_id,
                    next_state,
                    message,
                    progress_current,
                    progress_total,
                    detail,
                );

                redesmyn_storage::commands::insert_command_update(&mut *conn, &update).await?;
                redesmyn_storage::commands::update_command_state(
                    &mut *conn,
                    command_id,
                    next_state,
                    update.created_at_ms,
                )
                .await?;

                command.state = next_state;
                command.updated_at_ms = update.created_at_ms;

                Ok((command, update))
            })
        })
        .await?;

        self.publish_update(&command, &update).await;

        let last_update = Some(command_update_summary_from_record(update.clone())?);
        Ok(command_summary_from_records(command, last_update)?)
    }

    pub async fn wait_for_command(
        &self,
        command_id: CommandId,
        terminal_states: &[CommandState],
        timeout: Duration,
    ) -> Result<CommandSummary, ErrorEnvelope> {
        let mut rx = self.updates_tx.subscribe();

        let terminal_states: Vec<StorageCommandState> = if terminal_states.is_empty() {
            vec![
                StorageCommandState::Succeeded,
                StorageCommandState::Failed,
                StorageCommandState::Canceled,
            ]
        } else {
            terminal_states
                .iter()
                .filter_map(|state| storage_state_from_client_state(*state))
                .collect()
        };

        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let Some(current) = self.get_command(command_id).await.map_err(|err| {
                let envelope: ErrorEnvelope = err.into();
                envelope
            })? else {
                let detail =
                    ErrorDetail::from([("command_id".to_string(), command_id.to_string())]);
                return Err(ErrorEnvelope::new(ErrorCategory::NotFound, "Command not found.")
                    .with_detail(detail));
            };

            if storage_state_from_client_state(current.state)
                .is_some_and(|state| terminal_states.contains(&state))
            {
                return Ok(current);
            }

            tokio::select! {
                recv = rx.recv() => match recv {
                    Ok(notice) => {
                        if notice.command_id == command_id && terminal_states.contains(&notice.state) {
                            continue;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => {
                        let detail =
                            ErrorDetail::from([("command_id".to_string(), command_id.to_string())]);
                        return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, "Command wait channel closed.").with_detail(detail));
                    }
                },
                _ = tokio::time::sleep_until(deadline) => {
                    let detail =
                        ErrorDetail::from([("command_id".to_string(), command_id.to_string())]);
                    return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, "Timed out waiting for command.").with_detail(detail));
                }
            }
        }
    }

    pub async fn inflight_count(&self, scope: CommandScope) -> Result<i64, ControlPlaneError> {
        Ok(redesmyn_storage::commands::count_inflight_commands(&self.pool, scope).await?)
    }

    async fn publish_update(&self, command: &CommandRecord, update: &CommandUpdateRecord) {
        let _ = self.updates_tx.send(CommandUpdateNotice {
            command_id: command.command_id,
            state: update.state,
        });

        let event_scope = match command.scope {
            CommandScope::None => redesmyn_storage::events::EventScope::None,
            CommandScope::Repo {
                workspace_id,
                repo_id,
            } => redesmyn_storage::events::EventScope::Repo {
                workspace_id,
                repo_id,
            },
        };

        let event_kind = format!("command.{}", update.state.as_str());
        let payload = CommandUpdateEventPayload {
            command_id: command.command_id,
            update_id: update.update_id,
            state: update.state.as_str().to_string(),
            message: update.message.clone(),
        };

        let json_payload = match serde_json::to_vec(&payload) {
            Ok(json) => json,
            Err(err) => {
                warn!(error = %err, "failed to encode command update event payload");
                Vec::new()
            }
        };

        if let Err(err) = self
            .event_log
            .append_event(event_scope, event_kind, json_payload)
            .await
        {
            warn!(error = %err, "failed to append command update event");
        }
    }
}

fn validate_transition(
    from: StorageCommandState,
    to: StorageCommandState,
) -> Result<(), InvalidTransition> {
    if from == to {
        return Ok(());
    }

    if from.is_terminal() {
        return Err(InvalidTransition { from, to });
    }

    let ok = match from {
        StorageCommandState::Queued => true,
        StorageCommandState::Accepted => !matches!(to, StorageCommandState::Queued),
        StorageCommandState::Running => matches!(
            to,
            StorageCommandState::Blocked
                | StorageCommandState::Resumable
                | StorageCommandState::Succeeded
                | StorageCommandState::Failed
                | StorageCommandState::Canceled
        ),
        StorageCommandState::Blocked => matches!(
            to,
            StorageCommandState::Resumable
                | StorageCommandState::Running
                | StorageCommandState::Failed
                | StorageCommandState::Canceled
        ),
        StorageCommandState::Resumable => matches!(
            to,
            StorageCommandState::Running
                | StorageCommandState::Blocked
                | StorageCommandState::Succeeded
                | StorageCommandState::Failed
                | StorageCommandState::Canceled
        ),
        StorageCommandState::Succeeded
        | StorageCommandState::Failed
        | StorageCommandState::Canceled => false,
    };

    if ok { Ok(()) } else { Err(InvalidTransition { from, to }) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InvalidTransition {
    from: StorageCommandState,
    to: StorageCommandState,
}

fn command_summary_from_records(
    command: CommandRecord,
    last_update: Option<CommandUpdateSummary>,
) -> Result<CommandSummary, ControlPlaneError> {
    Ok(CommandSummary {
        command_id: command.command_id,
        created_at: timestamp_from_unix_ms(command.created_at_ms)?,
        updated_at: timestamp_from_unix_ms(command.updated_at_ms)?,
        kind: command.kind,
        state: map_command_state(command.state),
        target_task_id: command.target_task_id,
        last_update,
    })
}

fn command_update_summary_from_record(
    update: CommandUpdateRecord,
) -> Result<CommandUpdateSummary, ControlPlaneError> {
    Ok(CommandUpdateSummary {
        update_id: update.update_id,
        created_at: timestamp_from_unix_ms(update.created_at_ms)?,
        state: map_command_state(update.state),
        message: update.message,
        progress_current: update.progress_current.and_then(to_optional_u64),
        progress_total: update.progress_total.and_then(to_optional_u64),
    })
}

fn map_command_state(state: StorageCommandState) -> CommandState {
    match state {
        StorageCommandState::Queued => CommandState::Queued,
        StorageCommandState::Accepted => CommandState::Accepted,
        StorageCommandState::Running => CommandState::Running,
        StorageCommandState::Blocked => CommandState::Blocked,
        StorageCommandState::Resumable => CommandState::Resumable,
        StorageCommandState::Succeeded => CommandState::Succeeded,
        StorageCommandState::Failed => CommandState::Failed,
        StorageCommandState::Canceled => CommandState::Canceled,
    }
}

fn storage_state_from_client_state(state: CommandState) -> Option<StorageCommandState> {
    match state {
        CommandState::Unknown => None,
        CommandState::Queued => Some(StorageCommandState::Queued),
        CommandState::Accepted => Some(StorageCommandState::Accepted),
        CommandState::Running => Some(StorageCommandState::Running),
        CommandState::Blocked => Some(StorageCommandState::Blocked),
        CommandState::Resumable => Some(StorageCommandState::Resumable),
        CommandState::Succeeded => Some(StorageCommandState::Succeeded),
        CommandState::Failed => Some(StorageCommandState::Failed),
        CommandState::Canceled => Some(StorageCommandState::Canceled),
    }
}

fn timestamp_from_unix_ms(unix_ms: i64) -> Result<Timestamp, ControlPlaneError> {
    let nanos = i128::from(unix_ms)
        .checked_mul(1_000_000)
        .ok_or(ControlPlaneError::InvalidTimestamp { unix_ms })?;
    let dt = time::OffsetDateTime::from_unix_timestamp_nanos(nanos)
        .map_err(|_| ControlPlaneError::InvalidTimestamp { unix_ms })?;
    Ok(Timestamp::from_offset_date_time(dt))
}

fn to_optional_u64(value: i64) -> Option<u64> {
    u64::try_from(value).ok()
}

trait IsTerminal {
    fn is_terminal(self) -> bool;
}

impl IsTerminal for StorageCommandState {
    fn is_terminal(self) -> bool {
        matches!(
            self,
            StorageCommandState::Succeeded | StorageCommandState::Failed | StorageCommandState::Canceled
        )
    }
}
