use redesmyn_ids::{EpicId, RepoId, WorkspaceId};
use redesmyn_storage::{StorageError, apply_migrations, open_test_sqlite_pool};

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

    sqlx::query("DELETE FROM _sqlx_migrations WHERE version = 20260210003000")
        .execute(&pool)
        .await
        .expect("delete migration row");

    sqlx::query("DROP TABLE director_mode_state")
        .execute(&pool)
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
    .execute(&pool)
    .await
    .expect("create old director_mode_state");

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
    .execute(&pool)
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
    .execute(&pool)
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
    .execute(&pool)
    .await
    .expect("insert epic");

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

    let found: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM _sqlx_migrations
        WHERE version = 20260210003000 AND success = 1
        LIMIT 1
        "#,
    )
    .fetch_optional(&pool)
    .await
    .expect("query applied migration");
    assert_eq!(found, Some(1));

    let cols: Vec<String> = sqlx::query_scalar(
        r#"
        SELECT name
        FROM pragma_table_info('director_mode_state')
        ORDER BY cid
        "#,
    )
    .fetch_all(&pool)
    .await
    .expect("query table info");
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
