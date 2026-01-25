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
    migrate::MigrateError,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
};

use crate::StorageError;

pub type TxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(30);
const LEGACY_ID_MAP_MIGRATION_VERSION: i64 = 20260120001000;

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
        match MIGRATOR.run(pool).await {
            Ok(()) => {}
            Err(err) => {
                if try_repair_migrate_error(pool, &err).await? {
                    MIGRATOR.run(pool).await?;
                } else {
                    return Err(err.into());
                }
            }
        }
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

async fn try_repair_migrate_error(
    pool: &SqlitePool,
    err: &MigrateError,
) -> Result<bool, StorageError> {
    let MigrateError::ExecuteMigration(source, version) = err else {
        return Ok(false);
    };
    if *version != LEGACY_ID_MAP_MIGRATION_VERSION {
        return Ok(false);
    }
    if !sqlite_error_is_table_already_exists(source, "legacy_id_map") {
        return Ok(false);
    }
    if !sqlite_table_exists(pool, "legacy_id_map").await? {
        return Ok(false);
    }
    if sqlite_migration_is_applied(pool, LEGACY_ID_MAP_MIGRATION_VERSION).await? {
        return Ok(false);
    }
    if !legacy_id_map_schema_matches(pool).await? {
        warn!(
            version = LEGACY_ID_MAP_MIGRATION_VERSION,
            "refusing to repair migration; legacy_id_map schema does not match expected"
        );
        return Ok(false);
    }

    let Some(migration) = MIGRATOR
        .iter()
        .find(|candidate| candidate.version == LEGACY_ID_MAP_MIGRATION_VERSION)
    else {
        warn!(
            version = LEGACY_ID_MAP_MIGRATION_VERSION,
            "refusing to repair migration; embedded migration metadata missing"
        );
        return Ok(false);
    };

    warn!(
        version = LEGACY_ID_MAP_MIGRATION_VERSION,
        "repairing missing sqlx migration record for legacy_id_map"
    );

    sqlx::query(
        r#"
        INSERT INTO _sqlx_migrations (version, description, success, checksum, execution_time)
        VALUES (?1, ?2, 1, ?3, 0)
        ON CONFLICT(version) DO UPDATE SET
            success = excluded.success,
            checksum = excluded.checksum
        "#,
    )
    .bind(LEGACY_ID_MAP_MIGRATION_VERSION)
    .bind(migration.description.as_ref())
    .bind(migration.checksum.as_ref())
    .execute(pool)
    .await?;

    Ok(true)
}

fn sqlite_error_is_table_already_exists(err: &sqlx::Error, table_name: &str) -> bool {
    let Some(db_err) = err.as_database_error() else {
        return false;
    };
    let message = db_err.message();
    message.contains("already exists") && message.contains(table_name)
}

async fn sqlite_table_exists(pool: &SqlitePool, table_name: &str) -> Result<bool, StorageError> {
    let found: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?1
        LIMIT 1
        "#,
    )
    .bind(table_name)
    .fetch_optional(pool)
    .await?;
    Ok(found.is_some())
}

async fn sqlite_migration_is_applied(
    pool: &SqlitePool,
    version: i64,
) -> Result<bool, StorageError> {
    let found: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM _sqlx_migrations
        WHERE version = ?1 AND success = 1
        LIMIT 1
        "#,
    )
    .bind(version)
    .fetch_optional(pool)
    .await?;
    Ok(found.is_some())
}

#[derive(Debug, sqlx::FromRow)]
struct SqliteTableInfoRow {
    name: String,
    #[sqlx(rename = "type")]
    type_name: String,
    notnull: i64,
    pk: i64,
}

#[derive(Debug, sqlx::FromRow)]
struct SqliteIndexListRow {
    name: String,
    #[sqlx(rename = "unique")]
    is_unique: i64,
}

#[derive(Debug, sqlx::FromRow)]
struct SqliteIndexInfoRow {
    name: String,
}

async fn legacy_id_map_schema_matches(pool: &SqlitePool) -> Result<bool, StorageError> {
    let columns: Vec<SqliteTableInfoRow> =
        sqlx::query_as("PRAGMA table_info(legacy_id_map);")
            .fetch_all(pool)
            .await?;
    if columns.is_empty() {
        return Ok(false);
    }

    let mut column_by_name = std::collections::HashMap::new();
    for column in columns {
        column_by_name.insert(column.name.clone(), column);
    }

    let Some(table_name) = column_by_name.get("table_name") else {
        return Ok(false);
    };
    if !sqlite_type_matches(&table_name.type_name, "TEXT")
        || table_name.notnull != 1
        || table_name.pk != 1
    {
        return Ok(false);
    }

    let Some(legacy_id) = column_by_name.get("legacy_id") else {
        return Ok(false);
    };
    if !sqlite_type_matches(&legacy_id.type_name, "INTEGER")
        || legacy_id.notnull != 1
        || legacy_id.pk != 2
    {
        return Ok(false);
    }

    let Some(new_id) = column_by_name.get("new_id") else {
        return Ok(false);
    };
    if !sqlite_type_matches(&new_id.type_name, "BLOB") || new_id.notnull != 1 || new_id.pk != 0 {
        return Ok(false);
    }

    let Some(created_at_ms) = column_by_name.get("created_at_ms") else {
        return Ok(false);
    };
    if !sqlite_type_matches(&created_at_ms.type_name, "INTEGER")
        || created_at_ms.notnull != 1
        || created_at_ms.pk != 0
    {
        return Ok(false);
    }

    let indexes: Vec<SqliteIndexListRow> = sqlx::query_as("PRAGMA index_list(legacy_id_map);")
        .fetch_all(pool)
        .await?;

    if !indexes
        .iter()
        .any(|idx| idx.name == "idx_legacy_id_map_table_name" && idx.is_unique == 0)
    {
        return Ok(false);
    }

    for index in indexes.iter().filter(|idx| idx.is_unique == 1) {
        let cols: Vec<SqliteIndexInfoRow> =
            sqlx::query_as(&format!("PRAGMA index_info({});", index.name))
                .fetch_all(pool)
                .await?;
        if cols.len() == 1 && cols[0].name == "new_id" {
            return Ok(true);
        }
    }

    Ok(false)
}

fn sqlite_type_matches(actual: &str, expected_prefix: &str) -> bool {
    actual
        .trim()
        .to_ascii_uppercase()
        .starts_with(&expected_prefix.to_ascii_uppercase())
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
