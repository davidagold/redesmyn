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
