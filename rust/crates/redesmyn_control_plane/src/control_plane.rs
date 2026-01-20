use std::path::PathBuf;
use std::time::Duration;

use sqlx::SqlitePool;

use redesmyn_logging::tracing;

use crate::client_api::{ClientApiCodec, ClientApiContext, ClientApiServeError};
use crate::command::{CommandRegistry, CommandState};
use crate::event_log::EventLog;
use crate::task_manager::TaskManager;

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

#[derive(Debug)]
struct ControlPlaneState {
    #[allow(dead_code)]
    db_pool: SqlitePool,
    event_log: EventLog,
    commands: CommandRegistry,
}

pub struct ControlPlane;

impl ControlPlane {
    pub async fn start(options: ControlPlaneStartOptions) -> Result<ControlPlaneHandle, ControlPlaneStartError> {
        let span = tracing::info_span!("control_plane.start");
        let _enter = span.enter();

        let db_pool = open_db(options.db).await?;
        let event_log = EventLog::new(1024);
        let commands = CommandRegistry::new(event_log.clone());
        let state = std::sync::Arc::new(ControlPlaneState {
            db_pool,
            event_log: event_log.clone(),
            commands,
        });

        let mut tasks = TaskManager::new();

        let mut tick_shutdown = tasks.subscribe_shutdown();
        let tick_event_log = event_log.clone();
        tasks.spawn("event_log_tick", async move {
            let mut tick = tokio::time::interval(Duration::from_secs(2));
            loop {
                tokio::select! {
                    _ = tick_shutdown.recv() => break,
                    _ = tick.tick() => tick_event_log.publish("event_log.tick", Vec::new()),
                }
            }
        });

        state
            .commands
            .set_state(
                state.commands.create(Some("control_plane.startup".to_string())).await,
                CommandState::Succeeded,
                None,
            )
            .await;

        #[cfg(unix)]
        if let Some(socket_path) = options.client_api_socket_path.clone() {
            let listener = crate::client_api::bind_client_socket(&socket_path)?;
            let mut server_shutdown = tasks.subscribe_shutdown();
            let codec = options.client_api_codec;
            let ctx = ClientApiContext {
                event_log: event_log.clone(),
            };
            tasks.spawn("client_api_uds", async move {
                if let Err(err) = crate::client_api::serve_client_api_listener(
                    listener,
                    socket_path,
                    codec,
                    ctx,
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

pub struct ControlPlaneHandle {
    #[allow(dead_code)]
    state: std::sync::Arc<ControlPlaneState>,
    tasks: TaskManager,
}

impl ControlPlaneHandle {
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

