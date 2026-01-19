use std::{
    future::Future,
    path::Path,
    pin::Pin,
    str::FromStr,
    time::{Duration, Instant},
};

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!();

use redesmyn_logging::tracing::{Instrument, info, warn};
use sqlx::{
    SqliteConnection, SqlitePool,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
};

use crate::StorageError;

pub type TxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(30);

pub async fn open_sqlite_pool(db_path: impl AsRef<Path>) -> Result<SqlitePool, StorageError> {
    let db_path = db_path.as_ref().to_path_buf();
    let db_path_display = db_path.display().to_string();

    async {
        if let Some(parent) = db_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            // Startup-only. OK to block briefly while ensuring the DB directory exists.
            std::fs::create_dir_all(parent).map_err(|err| StorageError::CreateDbDir {
                path: parent.to_path_buf(),
                source: err,
            })?;
        }

        let connect_options = SqliteConnectOptions::new()
            .filename(&db_path)
            .create_if_missing(true)
            .busy_timeout(SQLITE_BUSY_TIMEOUT)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(connect_options)
            .await?;

        apply_migrations(&pool).await?;

        Ok(pool)
    }
    .instrument(redesmyn_logging::tracing::info_span!(
        "storage.open_sqlite_pool",
        db_path = %db_path_display
    ))
    .await
}

pub async fn open_test_sqlite_pool() -> Result<SqlitePool, StorageError> {
    async {
        let connect_options = SqliteConnectOptions::from_str("sqlite::memory:")?
            .busy_timeout(SQLITE_BUSY_TIMEOUT)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(connect_options)
            .await?;

        apply_migrations(&pool).await?;

        Ok(pool)
    }
    .instrument(redesmyn_logging::tracing::info_span!(
        "storage.open_test_sqlite_pool"
    ))
    .await
}

pub async fn apply_migrations(pool: &SqlitePool) -> Result<(), StorageError> {
    async {
        let start = Instant::now();
        MIGRATOR.run(pool).await?;
        info!(
            elapsed_ms = start.elapsed().as_millis(),
            "sqlite migrations applied"
        );
        Ok(())
    }
    .instrument(redesmyn_logging::tracing::info_span!(
        "storage.apply_migrations"
    ))
    .await
}

pub async fn in_transaction<T>(
    pool: &SqlitePool,
    f: impl for<'c> FnOnce(&'c mut SqliteConnection) -> TxFuture<'c, Result<T, StorageError>>,
) -> Result<T, StorageError> {
    let mut conn = pool.acquire().await?;

    sqlx::query("BEGIN").execute(&mut *conn).await?;

    let result = f(&mut *conn).await;
    match result {
        Ok(value) => {
            sqlx::query("COMMIT").execute(&mut *conn).await?;
            Ok(value)
        }
        Err(err) => {
            if let Err(rollback_err) = sqlx::query("ROLLBACK").execute(&mut *conn).await {
                warn!(?rollback_err, "sqlite ROLLBACK failed");
            }
            Err(err)
        }
    }
}
