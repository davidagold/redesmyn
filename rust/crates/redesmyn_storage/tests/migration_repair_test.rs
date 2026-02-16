use redesmyn_ids::{EpicId, RepoId, WorkspaceId};
use redesmyn_storage::{StorageError, apply_migrations, open_test_sqlite_pool};

const DIRECTOR_MODE_ERROR_FIELDS_MIGRATION_VERSION: i64 = 20260210003000;
const DIRECTOR_MODE_ERROR_CONSTRAINTS_MIGRATION_VERSION: i64 = 20260210004000;

async fn insert_scope(pool: &sqlx::SqlitePool) -> (WorkspaceId, RepoId, EpicId) {
    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();

    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, 1, 1, 'w')
        "#,
    )
    .bind(workspace_id)
    .execute(pool)
    .await
    .expect("insert workspace");
    sqlx::query(
        r#"
        INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, 1, 1, 'r', 'r')
        "#,
    )
    .bind(repo_id)
    .bind(workspace_id)
    .execute(pool)
    .await
    .expect("insert repo");
    sqlx::query(
        r#"
        INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, 1, 1, 'e', 'e')
        "#,
    )
    .bind(epic_id)
    .bind(repo_id)
    .execute(pool)
    .await
    .expect("insert epic");

    (workspace_id, repo_id, epic_id)
}

async fn delete_director_mode_migration_rows(pool: &sqlx::SqlitePool) {
    sqlx::query(
        r#"
        DELETE FROM _sqlx_migrations
        WHERE version IN (?1, ?2)
        "#,
    )
    .bind(DIRECTOR_MODE_ERROR_FIELDS_MIGRATION_VERSION)
    .bind(DIRECTOR_MODE_ERROR_CONSTRAINTS_MIGRATION_VERSION)
    .execute(pool)
    .await
    .expect("delete director mode migration rows");
}

async fn migration_applied(pool: &sqlx::SqlitePool, version: i64) -> Option<i64> {
    sqlx::query_scalar(
        r#"
        SELECT 1
        FROM _sqlx_migrations
        WHERE version = ?1 AND success = 1
        LIMIT 1
        "#,
    )
    .bind(version)
    .fetch_optional(pool)
    .await
    .expect("query applied migration")
}

async fn director_mode_columns(pool: &sqlx::SqlitePool) -> Vec<String> {
    sqlx::query_scalar(
        r#"
        SELECT name
        FROM pragma_table_info('director_mode_state')
        ORDER BY cid
        "#,
    )
    .fetch_all(pool)
    .await
    .expect("query director_mode_state columns")
}

async fn recreate_director_mode_state_without_error_columns(pool: &sqlx::SqlitePool) {
    sqlx::query("DROP TABLE director_mode_state")
        .execute(pool)
        .await
        .expect("drop director_mode_state");

    sqlx::query(
        r#"
        CREATE TABLE director_mode_state (
            epic_id BLOB(16) PRIMARY KEY NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            lifecycle TEXT NOT NULL DEFAULT 'inactive',
            director_session_id BLOB(16),
            activation_intent TEXT,
            resume_required_reason TEXT,
            resume_required_at_ms INTEGER,
            FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
            FOREIGN KEY (director_session_id) REFERENCES agent_sessions (session_id) ON DELETE SET NULL,
            CHECK (
                lifecycle IN ('inactive', 'active', 'paused', 'resume_required', 'error')
            ),
            CHECK (
                activation_intent IS NULL
                OR activation_intent IN ('run_in_current_session', 'run_in_new_session')
            ),
            CHECK (
                (
                    lifecycle = 'resume_required'
                    AND resume_required_at_ms IS NOT NULL
                )
                OR (
                    lifecycle <> 'resume_required'
                    AND resume_required_reason IS NULL
                    AND resume_required_at_ms IS NULL
                )
            )
        )
        "#,
    )
    .execute(pool)
    .await
    .expect("create pre-003000 director_mode_state");
}

async fn recreate_director_mode_state_with_error_reason_only(pool: &sqlx::SqlitePool) {
    sqlx::query("DROP TABLE director_mode_state")
        .execute(pool)
        .await
        .expect("drop director_mode_state");

    sqlx::query(
        r#"
        CREATE TABLE director_mode_state (
            epic_id BLOB(16) PRIMARY KEY NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            lifecycle TEXT NOT NULL DEFAULT 'inactive',
            director_session_id BLOB(16),
            activation_intent TEXT,
            resume_required_reason TEXT,
            resume_required_at_ms INTEGER,
            error_reason TEXT,
            FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
            FOREIGN KEY (director_session_id) REFERENCES agent_sessions (session_id) ON DELETE SET NULL,
            CHECK (
                lifecycle IN ('inactive', 'active', 'paused', 'resume_required', 'error')
            ),
            CHECK (
                activation_intent IS NULL
                OR activation_intent IN ('run_in_current_session', 'run_in_new_session')
            ),
            CHECK (
                (
                    lifecycle = 'resume_required'
                    AND resume_required_at_ms IS NOT NULL
                )
                OR (
                    lifecycle <> 'resume_required'
                    AND resume_required_reason IS NULL
                    AND resume_required_at_ms IS NULL
                )
            )
        )
        "#,
    )
    .execute(pool)
    .await
    .expect("create director_mode_state with error_reason only");
}

async fn recreate_director_mode_state_with_error_at_only(pool: &sqlx::SqlitePool) {
    sqlx::query("DROP TABLE director_mode_state")
        .execute(pool)
        .await
        .expect("drop director_mode_state");

    sqlx::query(
        r#"
        CREATE TABLE director_mode_state (
            epic_id BLOB(16) PRIMARY KEY NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            lifecycle TEXT NOT NULL DEFAULT 'inactive',
            director_session_id BLOB(16),
            activation_intent TEXT,
            resume_required_reason TEXT,
            resume_required_at_ms INTEGER,
            error_at_ms INTEGER,
            FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
            FOREIGN KEY (director_session_id) REFERENCES agent_sessions (session_id) ON DELETE SET NULL,
            CHECK (
                lifecycle IN ('inactive', 'active', 'paused', 'resume_required', 'error')
            ),
            CHECK (
                activation_intent IS NULL
                OR activation_intent IN ('run_in_current_session', 'run_in_new_session')
            ),
            CHECK (
                (
                    lifecycle = 'resume_required'
                    AND resume_required_at_ms IS NOT NULL
                )
                OR (
                    lifecycle <> 'resume_required'
                    AND resume_required_reason IS NULL
                    AND resume_required_at_ms IS NULL
                )
            )
        )
        "#,
    )
    .execute(pool)
    .await
    .expect("create director_mode_state with error_at_ms only");
}

async fn recreate_director_mode_state_without_error_checks(pool: &sqlx::SqlitePool) {
    sqlx::query("DROP TABLE director_mode_state")
        .execute(pool)
        .await
        .expect("drop director_mode_state");

    sqlx::query(
        r#"
        CREATE TABLE director_mode_state (
            epic_id BLOB(16) PRIMARY KEY NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            lifecycle TEXT NOT NULL DEFAULT 'inactive',
            director_session_id BLOB(16),
            activation_intent TEXT,
            resume_required_reason TEXT,
            resume_required_at_ms INTEGER,
            error_reason TEXT,
            error_at_ms INTEGER,
            FOREIGN KEY (epic_id) REFERENCES epics (id) ON DELETE CASCADE,
            FOREIGN KEY (director_session_id) REFERENCES agent_sessions (session_id) ON DELETE SET NULL,
            CHECK (
                lifecycle IN ('inactive', 'active', 'paused', 'resume_required', 'error')
            ),
            CHECK (
                activation_intent IS NULL
                OR activation_intent IN ('run_in_current_session', 'run_in_new_session')
            ),
            CHECK (
                (
                    lifecycle = 'resume_required'
                    AND resume_required_at_ms IS NOT NULL
                )
                OR (
                    lifecycle <> 'resume_required'
                    AND resume_required_reason IS NULL
                    AND resume_required_at_ms IS NULL
                )
            )
        )
        "#,
    )
    .execute(pool)
    .await
    .expect("create pre-004000 director_mode_state");
}

#[tokio::test]
async fn repairs_missing_legacy_id_map_migration_row() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");

    sqlx::query("DELETE FROM _sqlx_migrations WHERE version = 20260120001000")
        .execute(&pool)
        .await
        .expect("delete migration row");

    apply_migrations(&pool)
        .await
        .expect("expected migration repair to succeed");

    let found: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM _sqlx_migrations
        WHERE version = 20260120001000 AND success = 1
        LIMIT 1
        "#,
    )
    .fetch_optional(&pool)
    .await
    .expect("query applied migration");

    assert_eq!(found, Some(1));
}

#[tokio::test]
async fn refuses_to_repair_legacy_id_map_when_schema_mismatch() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");

    sqlx::query("DELETE FROM _sqlx_migrations WHERE version = 20260120001000")
        .execute(&pool)
        .await
        .expect("delete migration row");

    sqlx::query("DROP TABLE legacy_id_map")
        .execute(&pool)
        .await
        .expect("drop legacy_id_map");
    sqlx::query("CREATE TABLE legacy_id_map (table_name TEXT NOT NULL)")
        .execute(&pool)
        .await
        .expect("create incompatible legacy_id_map");

    let err = apply_migrations(&pool).await.expect_err("expected error");
    match err {
        StorageError::Migrate(sqlx::migrate::MigrateError::ExecuteMigration(_, 20260120001000)) => {
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn repairs_archived_at_rename_migration_when_schema_already_updated() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");

    sqlx::query("DELETE FROM _sqlx_migrations WHERE version = 20260210000000")
        .execute(&pool)
        .await
        .expect("delete migration row");

    apply_migrations(&pool)
        .await
        .expect("expected archived_at migration repair to succeed");

    let found: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM _sqlx_migrations
        WHERE version = 20260210000000 AND success = 1
        LIMIT 1
        "#,
    )
    .fetch_optional(&pool)
    .await
    .expect("query repaired migration row");

    assert_eq!(found, Some(1));
}

#[tokio::test]
async fn repairs_epic_id_migration_when_column_already_exists() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");

    sqlx::query("DELETE FROM _sqlx_migrations WHERE version = 20260210001000")
        .execute(&pool)
        .await
        .expect("delete migration row");

    apply_migrations(&pool)
        .await
        .expect("expected epic_id migration repair to succeed");

    let found: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM _sqlx_migrations
        WHERE version = 20260210001000 AND success = 1
        LIMIT 1
        "#,
    )
    .fetch_optional(&pool)
    .await
    .expect("query repaired migration row");
    assert_eq!(found, Some(1));

    let index_exists: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM sqlite_master
        WHERE type = 'index'
          AND name = 'idx_agent_sessions_epic_created_at'
        LIMIT 1
        "#,
    )
    .fetch_optional(&pool)
    .await
    .expect("query epic index");
    assert_eq!(index_exists, Some(1));
}

#[tokio::test]
async fn upgrades_director_mode_state_from_pre_error_metadata_schema() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");

    delete_director_mode_migration_rows(&pool).await;

    recreate_director_mode_state_without_error_columns(&pool).await;

    let (_, _, epic_id) = insert_scope(&pool).await;

    sqlx::query(
        r#"
        INSERT INTO director_mode_state (
            epic_id,
            updated_at_ms,
            lifecycle,
            director_session_id,
            activation_intent,
            resume_required_reason,
            resume_required_at_ms
        )
        VALUES (?1, 12345, 'error', NULL, NULL, NULL, NULL)
        "#,
    )
    .bind(epic_id)
    .execute(&pool)
    .await
    .expect("insert pre-upgrade director_mode_state row");

    apply_migrations(&pool)
        .await
        .expect("expected director error metadata migration to succeed");

    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_FIELDS_MIGRATION_VERSION).await,
        Some(1)
    );
    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_CONSTRAINTS_MIGRATION_VERSION).await,
        Some(1)
    );

    let cols = director_mode_columns(&pool).await;
    assert!(
        cols.iter().any(|name| name == "error_reason"),
        "expected error_reason column after upgrade"
    );
    assert!(
        cols.iter().any(|name| name == "error_at_ms"),
        "expected error_at_ms column after upgrade"
    );

    let upgraded_row: (Option<String>, Option<i64>) = sqlx::query_as(
        r#"
        SELECT error_reason, error_at_ms
        FROM director_mode_state
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .fetch_one(&pool)
    .await
    .expect("query upgraded row");
    assert_eq!(upgraded_row.0, None);
    assert_eq!(upgraded_row.1, Some(12345));
}

#[tokio::test]
async fn preserves_existing_director_mode_error_metadata_when_repairing_003000() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");
    let (_, _, epic_id) = insert_scope(&pool).await;

    sqlx::query(
        r#"
        INSERT INTO director_mode_state (
            epic_id,
            updated_at_ms,
            lifecycle,
            director_session_id,
            activation_intent,
            resume_required_reason,
            resume_required_at_ms,
            error_reason,
            error_at_ms
        )
        VALUES (?1, 99999, 'error', NULL, NULL, NULL, NULL, 'preserve me', 77777)
        "#,
    )
    .bind(epic_id)
    .execute(&pool)
    .await
    .expect("insert director_mode_state row with error metadata");

    delete_director_mode_migration_rows(&pool).await;

    apply_migrations(&pool)
        .await
        .expect("expected 003000 repair to succeed");

    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_FIELDS_MIGRATION_VERSION).await,
        Some(1)
    );
    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_CONSTRAINTS_MIGRATION_VERSION).await,
        Some(1)
    );

    let preserved: (Option<String>, Option<i64>) = sqlx::query_as(
        r#"
        SELECT error_reason, error_at_ms
        FROM director_mode_state
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .fetch_one(&pool)
    .await
    .expect("query preserved row");

    assert_eq!(preserved.0.as_deref(), Some("preserve me"));
    assert_eq!(preserved.1, Some(77777));
}

#[tokio::test]
async fn repairs_003000_when_only_error_reason_column_exists() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");
    delete_director_mode_migration_rows(&pool).await;
    recreate_director_mode_state_with_error_reason_only(&pool).await;

    let (_, _, epic_id) = insert_scope(&pool).await;
    sqlx::query(
        r#"
        INSERT INTO director_mode_state (
            epic_id,
            updated_at_ms,
            lifecycle,
            director_session_id,
            activation_intent,
            resume_required_reason,
            resume_required_at_ms,
            error_reason
        )
        VALUES (?1, 12345, 'error', NULL, NULL, NULL, NULL, 'reason-only')
        "#,
    )
    .bind(epic_id)
    .execute(&pool)
    .await
    .expect("insert row in reason-only schema");

    apply_migrations(&pool)
        .await
        .expect("expected migration repair for reason-only schema");

    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_FIELDS_MIGRATION_VERSION).await,
        Some(1)
    );
    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_CONSTRAINTS_MIGRATION_VERSION).await,
        Some(1)
    );

    let cols = director_mode_columns(&pool).await;
    assert!(cols.iter().any(|name| name == "error_reason"));
    assert!(cols.iter().any(|name| name == "error_at_ms"));

    let repaired: (Option<String>, Option<i64>) = sqlx::query_as(
        r#"
        SELECT error_reason, error_at_ms
        FROM director_mode_state
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .fetch_one(&pool)
    .await
    .expect("query repaired row");
    assert_eq!(repaired.0.as_deref(), Some("reason-only"));
    assert_eq!(repaired.1, Some(12345));
}

#[tokio::test]
async fn repairs_003000_when_only_error_at_ms_column_exists() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");
    delete_director_mode_migration_rows(&pool).await;
    recreate_director_mode_state_with_error_at_only(&pool).await;

    let (_, _, epic_id) = insert_scope(&pool).await;
    sqlx::query(
        r#"
        INSERT INTO director_mode_state (
            epic_id,
            updated_at_ms,
            lifecycle,
            director_session_id,
            activation_intent,
            resume_required_reason,
            resume_required_at_ms,
            error_at_ms
        )
        VALUES (?1, 22222, 'error', NULL, NULL, NULL, NULL, 55555)
        "#,
    )
    .bind(epic_id)
    .execute(&pool)
    .await
    .expect("insert row in error_at-only schema");

    apply_migrations(&pool)
        .await
        .expect("expected migration repair for error_at-only schema");

    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_FIELDS_MIGRATION_VERSION).await,
        Some(1)
    );
    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_CONSTRAINTS_MIGRATION_VERSION).await,
        Some(1)
    );

    let cols = director_mode_columns(&pool).await;
    assert!(cols.iter().any(|name| name == "error_reason"));
    assert!(cols.iter().any(|name| name == "error_at_ms"));

    let repaired: (Option<String>, Option<i64>) = sqlx::query_as(
        r#"
        SELECT error_reason, error_at_ms
        FROM director_mode_state
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .fetch_one(&pool)
    .await
    .expect("query repaired row");
    assert_eq!(repaired.0, None);
    assert_eq!(repaired.1, Some(55555));
}

#[tokio::test]
async fn upgrades_004000_preserve_error_metadata_and_enforce_constraints() {
    let pool = open_test_sqlite_pool().await.expect("open sqlite pool");

    sqlx::query(
        r#"
        DELETE FROM _sqlx_migrations
        WHERE version = ?1
        "#,
    )
    .bind(DIRECTOR_MODE_ERROR_CONSTRAINTS_MIGRATION_VERSION)
    .execute(&pool)
    .await
    .expect("delete 004000 migration row");

    recreate_director_mode_state_without_error_checks(&pool).await;

    let (_, _, epic_id) = insert_scope(&pool).await;
    sqlx::query(
        r#"
        INSERT INTO director_mode_state (
            epic_id,
            updated_at_ms,
            lifecycle,
            director_session_id,
            activation_intent,
            resume_required_reason,
            resume_required_at_ms,
            error_reason,
            error_at_ms
        )
        VALUES (?1, 30303, 'error', NULL, NULL, NULL, NULL, 'preserve-004000', 20202)
        "#,
    )
    .bind(epic_id)
    .execute(&pool)
    .await
    .expect("insert pre-004000 row");

    apply_migrations(&pool)
        .await
        .expect("expected 004000 migration to succeed");
    assert_eq!(
        migration_applied(&pool, DIRECTOR_MODE_ERROR_CONSTRAINTS_MIGRATION_VERSION).await,
        Some(1)
    );

    let preserved: (Option<String>, Option<i64>) = sqlx::query_as(
        r#"
        SELECT error_reason, error_at_ms
        FROM director_mode_state
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .fetch_one(&pool)
    .await
    .expect("query post-004000 row");
    assert_eq!(preserved.0.as_deref(), Some("preserve-004000"));
    assert_eq!(preserved.1, Some(20202));

    let non_error_metadata_err = sqlx::query(
        r#"
        UPDATE director_mode_state
        SET lifecycle = 'inactive', error_reason = 'must-fail', error_at_ms = 20202
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .execute(&pool)
    .await;
    assert!(
        non_error_metadata_err.is_err(),
        "expected CHECK failure for non-error lifecycle metadata"
    );

    let missing_error_timestamp_err = sqlx::query(
        r#"
        UPDATE director_mode_state
        SET lifecycle = 'error', error_reason = 'still-error', error_at_ms = NULL
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .execute(&pool)
    .await;
    assert!(
        missing_error_timestamp_err.is_err(),
        "expected CHECK failure for error lifecycle without error_at_ms"
    );
}
