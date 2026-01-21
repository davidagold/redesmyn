use std::path::Path;

use redesmyn_logging::tracing::{Instrument as _, info};
use sqlx::SqlitePool;

use crate::event_log::{EventLog, EventLogConfig};

#[derive(Debug, thiserror::Error)]
pub enum ControlPlaneInitError {
    #[error(transparent)]
    Storage(#[from] redesmyn_storage::StorageError),
}

#[derive(Clone)]
pub struct ControlPlane {
    pool: SqlitePool,
    event_log: EventLog,
}

impl ControlPlane {
    pub async fn open(db_path: impl AsRef<Path>) -> Result<Self, ControlPlaneInitError> {
        let db_path = db_path.as_ref().to_path_buf();
        let db_path_display = db_path.display().to_string();
        let span = redesmyn_logging::tracing::info_span!("control_plane.open", db_path = %db_path_display);

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
        .instrument(redesmyn_logging::tracing::info_span!("control_plane.open_test"))
        .await
    }

    #[must_use]
    pub fn new(pool: SqlitePool) -> Self {
        Self::new_with_event_log_config(pool, EventLogConfig::default())
    }

    #[must_use]
    pub fn new_with_event_log_config(pool: SqlitePool, config: EventLogConfig) -> Self {
        let event_log = EventLog::new_with_config(pool.clone(), config);
        Self { pool, event_log }
    }

    #[must_use]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    #[must_use]
    pub fn event_log(&self) -> &EventLog {
        &self.event_log
    }
}
