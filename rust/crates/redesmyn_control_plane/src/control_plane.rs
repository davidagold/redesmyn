use std::path::{Path, PathBuf};

use redesmyn_logging::tracing::{self, Instrument as _, info};
use sqlx::SqlitePool;
use tokio::runtime::Handle;

use crate::client_api::{ClientApiCodec, ClientApiServeError};
use crate::command::{CommandRegistry, CommandState};
use crate::event_log::{EventLog, EventLogConfig};
use crate::session_events::{SessionEvents, SessionEventsConfig};
use crate::task_manager::TaskManager;

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
        Self {
            pool,
            event_log,
            session_events,
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

    pub async fn start(
        options: ControlPlaneStartOptions,
    ) -> Result<ControlPlaneHandle, ControlPlaneStartError> {
        let span = tracing::info_span!("control_plane.start");
        let _enter = span.enter();

        let db_pool = open_db(options.db).await?;
        let control_plane = Self::new(db_pool);

        let commands = CommandRegistry::new(control_plane.event_log().clone());
        commands
            .set_state(
                commands
                    .create(Some("control_plane.startup".to_string()))
                    .await,
                CommandState::Succeeded,
                None,
            )
            .await;

        let state = std::sync::Arc::new(ControlPlaneState {
            control_plane: control_plane.clone(),
            commands,
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

struct ControlPlaneState {
    #[allow(dead_code)]
    control_plane: ControlPlane,
    #[allow(dead_code)]
    commands: CommandRegistry,
}

pub struct ControlPlaneHandle {
    #[allow(dead_code)]
    state: std::sync::Arc<ControlPlaneState>,
    tasks: TaskManager,
}

impl ControlPlaneHandle {
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
