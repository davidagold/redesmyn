use std::path::{Path, PathBuf};

use redesmyn_logging::tracing::{self, Instrument as _, info};
use sqlx::SqlitePool;
use tokio::runtime::Handle;

use crate::client_api::{ClientApiCodec, ClientApiServeError};
use crate::command::Commands;
use crate::daemon_router::DaemonRouter;
use crate::event_log::{EventLog, EventLogConfig};
use crate::session_events::{SessionEvents, SessionEventsConfig};
use crate::task_manager::TaskManager;

use redesmyn_ids::{CommandId, HostInstanceId, TaskId};
use redesmyn_protocol::daemon::{
    CommandUpdate as DaemonCommandUpdate, SessionEventBatch, SessionLiveEventBatch,
};
use redesmyn_protocol::sync_commands::{
    LOCAL_SYNC_EVENT_FAILED, LOCAL_SYNC_FROM_DOCS_KIND, LocalSyncFromDocsCommand,
};
use redesmyn_protocol::{ErrorEnvelope, RepoScope};
use redesmyn_storage::commands::CommandScope;
use redesmyn_storage::events::EventScope;
use redesmyn_storage::schema::CommandState as StorageCommandState;

#[derive(Debug, thiserror::Error)]
pub enum ControlPlaneInitError {
    #[error(transparent)]
    Storage(#[from] redesmyn_storage::StorageError),
}

#[derive(Debug, Clone)]
pub enum ControlPlaneDb {
    Path(PathBuf),
    InMemory,
    Pool(SqlitePool),
}

#[derive(Debug, Clone)]
pub struct ControlPlaneStartOptions {
    pub db: ControlPlaneDb,
    pub client_api_socket_path: Option<PathBuf>,
    pub client_api_codec: ClientApiCodec,
}

impl ControlPlaneStartOptions {
    #[must_use]
    pub fn from_config(config: &redesmyn_config::ControlPlaneConfig) -> Self {
        Self {
            db: ControlPlaneDb::Path(config.db.path.clone()),
            client_api_socket_path: Some(config.api.client_socket_path.clone()),
            client_api_codec: ClientApiCodec::Protobuf,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ControlPlaneStartError {
    #[error(transparent)]
    Storage(#[from] redesmyn_storage::StorageError),
    #[error(transparent)]
    ClientApiBind(#[from] ClientApiServeError),
    #[error("client API UDS is not supported on this platform")]
    ClientApiUnsupported,
}

#[derive(Clone)]
pub struct ControlPlane {
    pool: SqlitePool,
    event_log: EventLog,
    session_events: SessionEvents,
    commands: Commands,
    daemons: DaemonRouter,
}

impl ControlPlane {
    pub async fn open(db_path: impl AsRef<Path>) -> Result<Self, ControlPlaneInitError> {
        let db_path = db_path.as_ref().to_path_buf();
        let db_path_display = db_path.display().to_string();
        let span =
            redesmyn_logging::tracing::info_span!("control_plane.open", db_path = %db_path_display);

        async {
            info!(db_path = %db_path_display, "opening control plane");
            let pool = redesmyn_storage::open_sqlite_pool(&db_path).await?;
            Ok(Self::new(pool))
        }
        .instrument(span)
        .await
    }

    pub async fn open_test() -> Result<Self, ControlPlaneInitError> {
        async move {
            let pool = redesmyn_storage::open_test_sqlite_pool().await?;
            Ok(Self::new(pool))
        }
        .instrument(redesmyn_logging::tracing::info_span!(
            "control_plane.open_test"
        ))
        .await
    }

    #[must_use]
    pub fn new(pool: SqlitePool) -> Self {
        Self::new_with_event_log_config(pool, EventLogConfig::default())
    }

    #[must_use]
    pub fn new_with_event_log_config(pool: SqlitePool, config: EventLogConfig) -> Self {
        let event_log = EventLog::new_with_config(pool.clone(), config);
        let session_events =
            SessionEvents::new_with_config(pool.clone(), SessionEventsConfig::default());
        let commands = Commands::new(pool.clone(), event_log.clone());
        let daemons = DaemonRouter::new();
        Self {
            pool,
            event_log,
            session_events,
            commands,
            daemons,
        }
    }

    #[must_use]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    #[must_use]
    pub fn event_log(&self) -> &EventLog {
        &self.event_log
    }

    #[must_use]
    pub fn session_events(&self) -> &SessionEvents {
        &self.session_events
    }

    #[must_use]
    pub fn commands(&self) -> &Commands {
        &self.commands
    }

    #[must_use]
    pub fn daemons(&self) -> &DaemonRouter {
        &self.daemons
    }

    pub async fn issue_command(
        &self,
        scope: CommandScope,
        kind: String,
        target_task_id: Option<TaskId>,
        idempotency_key: Option<String>,
        created_by: Option<String>,
        payload: Vec<u8>,
    ) -> Result<redesmyn_protocol::client::CommandSummary, crate::error::ControlPlaneError> {
        let payload_for_create = payload.clone();
        let result = self
            .commands()
            .create_command(
                scope,
                kind.clone(),
                target_task_id,
                idempotency_key,
                created_by,
                payload_for_create,
            )
            .await?;

        if result.created_new {
            self.dispatch_repo_command_if_needed(scope, result.command.command_id, kind, payload)
                .await?;

            return self
                .commands()
                .get_command(result.command.command_id)
                .await?
                .ok_or(crate::error::ControlPlaneError::CommandNotFound {
                    command_id: result.command.command_id,
                });
        }

        Ok(result.command)
    }

    async fn dispatch_repo_command_if_needed(
        &self,
        scope: CommandScope,
        command_id: CommandId,
        kind: String,
        json_payload: Vec<u8>,
    ) -> Result<(), crate::error::ControlPlaneError> {
        let CommandScope::Repo {
            workspace_id,
            repo_id,
        } = scope
        else {
            return Ok(());
        };

        let repo_scope = RepoScope {
            workspace_id,
            repo_id,
        };

        if kind == LOCAL_SYNC_FROM_DOCS_KIND {
            self.commands()
                .append_update(
                    command_id,
                    StorageCommandState::Running,
                    Some("Syncing local docs…".to_string()),
                    None,
                    None,
                    None,
                )
                .await?;

            let sync_command =
                match serde_json::from_slice::<LocalSyncFromDocsCommand>(&json_payload) {
                    Ok(value) => value,
                    Err(err) => {
                        let message = format!("Sync failed: invalid payload ({err})");
                        self.commands()
                            .append_update(
                                command_id,
                                StorageCommandState::Failed,
                                Some(message),
                                None,
                                None,
                                None,
                            )
                            .await?;
                        return Ok(());
                    }
                };

            match crate::local_sync::run_local_sync_from_docs(
                &self.pool,
                self.event_log(),
                repo_scope,
                &sync_command,
            )
            .await
            {
                Ok(stats) => {
                    let summary = format!(
                        "Synced from local: epics +{}/~{}, tasks +{}/~{} (parent links ~{})",
                        stats.epics_created,
                        stats.epics_updated,
                        stats.tasks_created,
                        stats.tasks_updated,
                        stats.parent_links_updated
                    );
                    let detail = serde_json::to_vec(&stats).ok();
                    self.commands()
                        .append_update(
                            command_id,
                            StorageCommandState::Succeeded,
                            Some(summary),
                            None,
                            None,
                            detail,
                        )
                        .await?;
                }
                Err(err) => {
                    let message = format!("Sync failed: {err}");
                    let failed_payload = serde_json::json!({
                        "epic_slug": sync_command.epic_slug,
                        "message": message,
                    });
                    if let Err(event_err) = self
                        .event_log()
                        .append_event(
                            EventScope::Repo {
                                workspace_id,
                                repo_id,
                            },
                            LOCAL_SYNC_EVENT_FAILED,
                            serde_json::to_vec(&failed_payload).unwrap_or_default(),
                        )
                        .await
                    {
                        tracing::warn!(error = %event_err, "failed to append local sync failed event");
                    }

                    self.commands()
                        .append_update(
                            command_id,
                            StorageCommandState::Failed,
                            Some(message),
                            None,
                            None,
                            None,
                        )
                        .await?;
                }
            }

            return Ok(());
        }

        let dispatch_result = self
            .daemons()
            .dispatch_command(repo_scope, command_id, kind, json_payload)
            .await;

        if let Err(err) = dispatch_result {
            let detail = serde_json::to_vec(&err).unwrap_or_default();
            let _ = self
                .commands()
                .append_update(
                    command_id,
                    StorageCommandState::Failed,
                    Some(err.message),
                    None,
                    None,
                    Some(detail),
                )
                .await?;
        }

        Ok(())
    }

    pub async fn apply_daemon_command_update(
        &self,
        host_instance_id: HostInstanceId,
        update: DaemonCommandUpdate,
    ) -> Result<(), crate::error::ControlPlaneError> {
        let command_id = update.command_id;
        if !self
            .daemons()
            .authorize_command_update(host_instance_id, command_id)
            .await
        {
            tracing::warn!(
                host_instance_id = %host_instance_id,
                command_id = %command_id,
                "ignoring unauthorized daemon command update"
            );
            return Ok(());
        }

        let mapped = map_daemon_command_update(update);

        let _ = self
            .commands()
            .append_update(
                command_id,
                mapped.state,
                mapped.message,
                mapped.progress_current,
                mapped.progress_total,
                mapped.detail,
            )
            .await?;

        Ok(())
    }

    pub async fn apply_daemon_session_event_batch(
        &self,
        _host_instance_id: HostInstanceId,
        batch: SessionEventBatch,
    ) -> Result<(), crate::error::ControlPlaneError> {
        for event in batch.events {
            if let Err(err) = self.session_events.append_session_event(&event).await {
                tracing::warn!(error = %err, "failed to append session event");
                continue;
            }

            if let Err(err) =
                crate::session_events_projection::apply_session_event_to_agent_session_row(
                    &self.pool, &event,
                )
                .await
            {
                tracing::warn!(error = %err, "failed to apply session event to agent session row");
            }
        }

        Ok(())
    }

    pub async fn apply_daemon_session_live_event_batch(
        &self,
        _host_instance_id: HostInstanceId,
        batch: SessionLiveEventBatch,
    ) -> Result<(), crate::error::ControlPlaneError> {
        for event in batch.events {
            self.session_events.publish_live_event(event);
        }
        Ok(())
    }

    pub async fn start(
        options: ControlPlaneStartOptions,
    ) -> Result<ControlPlaneHandle, ControlPlaneStartError> {
        let span = tracing::info_span!("control_plane.start");
        let _enter = span.enter();

        let db_pool = open_db(options.db).await?;
        let control_plane = Self::new(db_pool);

        let state = std::sync::Arc::new(ControlPlaneState {
            control_plane: control_plane.clone(),
        });

        let mut tasks = TaskManager::new(Handle::current());

        #[cfg(unix)]
        if let Some(socket_path) = options.client_api_socket_path.clone() {
            let listener = crate::client_api::bind_client_socket(&socket_path)?;
            let mut server_shutdown = tasks.subscribe_shutdown();
            let codec = options.client_api_codec;
            let server_control_plane = control_plane.clone();

            tasks.spawn("client_api_uds", async move {
                if let Err(err) = crate::client_api::serve_client_api_listener(
                    listener,
                    server_control_plane,
                    socket_path,
                    codec,
                    &mut server_shutdown,
                )
                .await
                {
                    tracing::error!(error = %err, "client API server task exited");
                }
            });
        }

        #[cfg(not(unix))]
        if options.client_api_socket_path.is_some() {
            return Err(ControlPlaneStartError::ClientApiUnsupported);
        }

        tracing::info!("control plane started");

        Ok(ControlPlaneHandle { state, tasks })
    }
}

struct MappedDaemonCommandUpdate {
    state: StorageCommandState,
    message: Option<String>,
    progress_current: Option<i64>,
    progress_total: Option<i64>,
    detail: Option<Vec<u8>>,
}

fn map_daemon_command_update(update: DaemonCommandUpdate) -> MappedDaemonCommandUpdate {
    let state = match update.state {
        redesmyn_protocol::daemon::CommandState::Queued => StorageCommandState::Queued,
        redesmyn_protocol::daemon::CommandState::Accepted => StorageCommandState::Accepted,
        redesmyn_protocol::daemon::CommandState::Running => StorageCommandState::Running,
        redesmyn_protocol::daemon::CommandState::Blocked => StorageCommandState::Blocked,
        redesmyn_protocol::daemon::CommandState::Resumable => StorageCommandState::Resumable,
        redesmyn_protocol::daemon::CommandState::Succeeded => StorageCommandState::Succeeded,
        redesmyn_protocol::daemon::CommandState::Failed => StorageCommandState::Failed,
        redesmyn_protocol::daemon::CommandState::Canceled => StorageCommandState::Canceled,
        redesmyn_protocol::daemon::CommandState::Rejected => StorageCommandState::Failed,
    };

    let (progress_current, progress_total) = update
        .progress
        .map(|progress| {
            let percent: i64 = progress.percent.into();
            (Some(percent), Some(100))
        })
        .unwrap_or((None, None));

    #[derive(serde::Serialize)]
    struct UpdateDetail {
        detail: Option<redesmyn_protocol::ErrorDetail>,
        error: Option<ErrorEnvelope>,
    }

    let detail = serde_json::to_vec(&UpdateDetail {
        detail: update.detail,
        error: update.error,
    })
    .ok();

    MappedDaemonCommandUpdate {
        state,
        message: update.message,
        progress_current,
        progress_total,
        detail,
    }
}

struct ControlPlaneState {
    #[allow(dead_code)]
    control_plane: ControlPlane,
}

pub struct ControlPlaneHandle {
    #[allow(dead_code)]
    state: std::sync::Arc<ControlPlaneState>,
    tasks: TaskManager,
}

impl ControlPlaneHandle {
    #[must_use]
    pub fn control_plane(&self) -> ControlPlane {
        self.state.control_plane.clone()
    }

    /// Connect a client to the control plane using an in-proc transport.
    ///
    /// This returns the client side of an in-memory channel pair and spawns a
    /// server task that is shut down along with the control plane.
    pub fn connect_in_proc_client(
        &mut self,
        buffer: usize,
    ) -> redesmyn_transport::client::in_proc::InProcEndpoint {
        let (client, mut server) =
            redesmyn_transport::client::in_proc::InProcEndpoint::pair(buffer);

        let server_control_plane = self.state.control_plane.clone();
        let mut shutdown = self.tasks.subscribe_shutdown();

        self.tasks.spawn("client_api_in_proc", async move {
            let span = tracing::info_span!("client.api.connection", peer = "<in_proc>");
            let _enter = span.enter();

            if let Err(err) = crate::client_api::serve_connection(
                &mut server,
                server_control_plane,
                &mut shutdown,
            )
            .await
            {
                tracing::warn!(
                    error = %err,
                    "in-proc client API connection terminated with error"
                );
            }
        });

        client
    }

    /// Fixture-only helper for local desktop dev tooling.
    #[cfg(any(debug_assertions, test))]
    #[must_use]
    pub fn session_events(&self) -> crate::session_events::SessionEvents {
        self.state.control_plane.session_events().clone()
    }

    pub async fn shutdown(self) {
        let span = tracing::info_span!("control_plane.shutdown");
        let _enter = span.enter();

        self.tasks.trigger_shutdown();
        self.tasks.join_all().await;

        tracing::info!("control plane shut down");
    }
}

async fn open_db(db: ControlPlaneDb) -> Result<SqlitePool, ControlPlaneStartError> {
    match db {
        ControlPlaneDb::Path(path) => Ok(redesmyn_storage::open_sqlite_pool(path).await?),
        ControlPlaneDb::InMemory => Ok(redesmyn_storage::open_test_sqlite_pool().await?),
        ControlPlaneDb::Pool(pool) => Ok(pool),
    }
}
