use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_logging::tracing::{self, Instrument as _, info};
use sqlx::SqlitePool;
use tokio::runtime::Handle;

use crate::client_api::{ClientApiCodec, ClientApiServeError};
use crate::command::Commands;
use crate::daemon_router::DaemonRouter;
use crate::director_wake::DirectorWakeController;
use crate::event_log::{EventLog, EventLogConfig};
use crate::session_events::{SessionEvents, SessionEventsConfig};
use crate::task_manager::TaskManager;

use redesmyn_ids::{CommandId, HostId, HostInstanceId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::agent_commands::{
    AttachTaskAgentSessionCommand, ResumeByIdTaskAgentTurnCommand, SESSION_AGENT_ATTACH_SESSION,
    SESSION_AGENT_RESUME_BY_ID_TURN, SESSION_AGENT_START, StartAgentSessionCommand,
    StartTaskAgentSessionCommand, TASK_AGENT_START, TASK_AGENT_STOP,
};
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

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

// Guard legacy cleanup to old/stale rows so we avoid clobbering active sessions that
// predate runner ownership but are still generating recent updates.
const LEGACY_NULL_OWNER_RECONCILE_MIN_AGE_MS: i64 = 5 * 60 * 1000;

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
    director_wake: DirectorWakeController,
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
        let session_events = SessionEvents::new_with_config(
            pool.clone(),
            event_log.clone(),
            SessionEventsConfig::default(),
        );
        let director_wake = DirectorWakeController::new(pool.clone(), event_log.clone());
        let commands = Commands::new(pool.clone(), event_log.clone());
        let daemons = DaemonRouter::new();
        Self {
            pool,
            event_log,
            session_events,
            director_wake,
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
    pub fn director_wake(&self) -> &DirectorWakeController {
        &self.director_wake
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
            .dispatch_command(repo_scope, command_id, kind.clone(), json_payload.clone())
            .await;

        match dispatch_result {
            Ok(host_instance_id) => {
                if let Some(session_id) =
                    extract_session_id_for_owner_assignment(&kind, &json_payload)
                {
                    if let Err(err) = self
                        .assign_session_runner_instance(session_id, host_instance_id)
                        .await
                    {
                        tracing::warn!(
                            session_id = %session_id,
                            host_instance_id = %host_instance_id,
                            error = %err,
                            "failed to persist session runner ownership"
                        );
                    }
                }
            }
            Err(err) => {
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
        }

        Ok(())
    }

    pub async fn register_daemon_presence(
        &self,
        host_id: HostId,
        host_instance_id: HostInstanceId,
    ) -> Result<(), crate::error::ControlPlaneError> {
        let now = now_ms();
        let legacy_cutoff_ms = now.saturating_sub(LEGACY_NULL_OWNER_RECONCILE_MIN_AGE_MS);

        let affected_tasks: Vec<(WorkspaceId, redesmyn_ids::RepoId, TaskId)> =
            redesmyn_storage::in_transaction(&self.pool, |conn| {
                Box::pin(async move {
                    sqlx::query(
                        r#"
                        INSERT INTO hosts (id, created_at_ms, hostname)
                        VALUES (?1, ?2, ?3)
                        ON CONFLICT(id) DO NOTHING
                        "#,
                    )
                    .bind(host_id)
                    .bind(now)
                    .bind(std::env::var("HOSTNAME").ok())
                    .execute(&mut *conn)
                    .await?;

                    let stale_rows: Vec<(HostInstanceId,)> = sqlx::query_as(
                        r#"
                        SELECT host_instance_id
                        FROM daemon_presence
                        WHERE host_id = ?1
                          AND disconnected_at_ms IS NULL
                          AND host_instance_id != ?2
                        "#,
                    )
                    .bind(host_id)
                    .bind(host_instance_id)
                    .fetch_all(&mut *conn)
                    .await?;

                    sqlx::query(
                        r#"
                        UPDATE daemon_presence
                        SET
                            last_heartbeat_at_ms = CASE
                                WHEN last_heartbeat_at_ms < ?1 THEN ?1
                                ELSE last_heartbeat_at_ms
                            END,
                            disconnected_at_ms = COALESCE(disconnected_at_ms, ?1)
                        WHERE host_id = ?2
                          AND disconnected_at_ms IS NULL
                          AND host_instance_id != ?3
                        "#,
                    )
                    .bind(now)
                    .bind(host_id)
                    .bind(host_instance_id)
                    .execute(&mut *conn)
                    .await?;

                    sqlx::query(
                        r#"
                        INSERT INTO daemon_presence (
                            host_instance_id,
                            host_id,
                            connected_at_ms,
                            last_heartbeat_at_ms,
                            disconnected_at_ms
                        )
                        VALUES (?1, ?2, ?3, ?3, NULL)
                        ON CONFLICT(host_instance_id) DO UPDATE SET
                            last_heartbeat_at_ms = excluded.last_heartbeat_at_ms,
                            disconnected_at_ms = NULL
                        "#,
                    )
                    .bind(host_instance_id)
                    .bind(host_id)
                    .bind(now)
                    .execute(&mut *conn)
                    .await?;

                    let stale_instance_ids: Vec<HostInstanceId> =
                        stale_rows.into_iter().map(|(id,)| id).collect();
                    let mut affected_tasks = Vec::new();

                    if !stale_instance_ids.is_empty() {
                        let mut affected_builder = sqlx::QueryBuilder::new(
                            r#"
                            SELECT DISTINCT scope_workspace_id, scope_repo_id, task_id
                            FROM agent_sessions
                            WHERE scope_kind = 'task'
                              AND ended_at_ms IS NULL
                              AND task_id IS NOT NULL
                              AND runner_host_instance_id IN (
                            "#,
                        );
                        {
                            let mut ids = affected_builder.separated(", ");
                            for id in &stale_instance_ids {
                                ids.push_bind(*id);
                            }
                        }
                        affected_builder.push(")");
                        affected_tasks.extend(
                            affected_builder
                                .build_query_as::<(WorkspaceId, redesmyn_ids::RepoId, TaskId)>()
                                .fetch_all(&mut *conn)
                                .await?,
                        );

                        let mut reconcile_builder = sqlx::QueryBuilder::new(
                            r#"
                            UPDATE agent_sessions
                            SET
                                updated_at_ms = 
                            "#,
                        );
                        reconcile_builder.push_bind(now);
                        reconcile_builder.push(
                            r#",
                                status = 'error',
                                ended_at_ms = COALESCE(ended_at_ms, 
                            "#,
                        );
                        reconcile_builder.push_bind(now);
                        reconcile_builder.push(
                            r#")
                            WHERE ended_at_ms IS NULL
                              AND runner_host_instance_id IN (
                            "#,
                        );
                        {
                            let mut ids = reconcile_builder.separated(", ");
                            for id in &stale_instance_ids {
                                ids.push_bind(*id);
                            }
                        }
                        reconcile_builder.push(")");
                        reconcile_builder.build().execute(&mut *conn).await?;
                    }

                    let legacy_tasks: Vec<(WorkspaceId, redesmyn_ids::RepoId, TaskId)> =
                        sqlx::query_as(
                            r#"
                            SELECT DISTINCT scope_workspace_id, scope_repo_id, task_id
                            FROM agent_sessions
                            WHERE scope_kind = 'task'
                              AND ended_at_ms IS NULL
                              AND task_id IS NOT NULL
                              AND runner_host_instance_id IS NULL
                              AND status IN ('running', 'blocked')
                              AND updated_at_ms <= ?1
                            "#,
                        )
                        .bind(legacy_cutoff_ms)
                        .fetch_all(&mut *conn)
                        .await?;
                    if !legacy_tasks.is_empty() {
                        sqlx::query(
                            r#"
                            UPDATE agent_sessions
                            SET
                                updated_at_ms = ?1,
                                status = 'error',
                                ended_at_ms = COALESCE(ended_at_ms, ?2)
                            WHERE scope_kind = 'task'
                              AND ended_at_ms IS NULL
                              AND runner_host_instance_id IS NULL
                              AND status IN ('running', 'blocked')
                              AND updated_at_ms <= ?3
                            "#,
                        )
                        .bind(now)
                        .bind(now)
                        .bind(legacy_cutoff_ms)
                        .execute(&mut *conn)
                        .await?;
                        affected_tasks.extend(legacy_tasks);
                    }

                    Ok(affected_tasks)
                })
            })
            .await?;

        let mut seen = HashSet::new();
        for (workspace_id, repo_id, task_id) in affected_tasks {
            if !seen.insert((workspace_id, repo_id, task_id)) {
                continue;
            }
            self.append_reconcile_task_command(workspace_id, repo_id, task_id)
                .await?;
        }

        Ok(())
    }

    pub async fn unregister_daemon_presence(
        &self,
        host_instance_id: HostInstanceId,
    ) -> Result<(), crate::error::ControlPlaneError> {
        let now = now_ms();
        sqlx::query(
            r#"
            UPDATE daemon_presence
            SET
                last_heartbeat_at_ms = CASE
                    WHEN last_heartbeat_at_ms < ?1 THEN ?1
                    ELSE last_heartbeat_at_ms
                END,
                disconnected_at_ms = COALESCE(disconnected_at_ms, ?1)
            WHERE host_instance_id = ?2
            "#,
        )
        .bind(now)
        .bind(host_instance_id)
        .execute(&self.pool)
        .await
        .map_err(redesmyn_storage::StorageError::from)?;

        Ok(())
    }

    async fn append_reconcile_task_command(
        &self,
        workspace_id: WorkspaceId,
        repo_id: redesmyn_ids::RepoId,
        task_id: TaskId,
    ) -> Result<(), crate::error::ControlPlaneError> {
        let payload = serde_json::to_vec(&serde_json::json!({
            "task_id": task_id,
            "reason": "runner_disconnected",
        }))
        .unwrap_or_default();

        let created = self
            .commands()
            .create_command(
                CommandScope::Repo {
                    workspace_id,
                    repo_id,
                },
                TASK_AGENT_STOP.to_string(),
                Some(task_id),
                None,
                Some("control_plane.reconcile".to_string()),
                payload,
            )
            .await?;

        let _ = self
            .commands()
            .append_update(
                created.command.command_id,
                StorageCommandState::Failed,
                Some("Runner disconnected; previous task session was reconciled.".to_string()),
                None,
                None,
                None,
            )
            .await?;

        Ok(())
    }

    async fn assign_session_runner_instance(
        &self,
        session_id: SessionId,
        host_instance_id: HostInstanceId,
    ) -> Result<(), crate::error::ControlPlaneError> {
        sqlx::query(
            r#"
            UPDATE agent_sessions
            SET
                runner_host_instance_id = ?1,
                runner_host_id = (
                    SELECT host_id
                    FROM daemon_presence
                    WHERE host_instance_id = ?1
                    LIMIT 1
                )
            WHERE session_id = ?2
            "#,
        )
        .bind(host_instance_id)
        .bind(session_id)
        .execute(&self.pool)
        .await
        .map_err(redesmyn_storage::StorageError::from)?;
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

fn extract_session_id_for_owner_assignment(kind: &str, payload: &[u8]) -> Option<SessionId> {
    match kind {
        TASK_AGENT_START => serde_json::from_slice::<StartTaskAgentSessionCommand>(payload)
            .ok()
            .map(|command| command.session_id),
        SESSION_AGENT_START => serde_json::from_slice::<StartAgentSessionCommand>(payload)
            .ok()
            .map(|command| command.session_id),
        SESSION_AGENT_RESUME_BY_ID_TURN => {
            serde_json::from_slice::<ResumeByIdTaskAgentTurnCommand>(payload)
                .ok()
                .map(|command| command.session_id)
        }
        SESSION_AGENT_ATTACH_SESSION => {
            serde_json::from_slice::<AttachTaskAgentSessionCommand>(payload)
                .ok()
                .map(|command| command.session_id)
        }
        _ => None,
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
