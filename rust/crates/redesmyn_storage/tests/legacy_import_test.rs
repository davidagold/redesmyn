use std::path::Path;

use redesmyn_ids::TaskId;
use redesmyn_storage::legacy_import::{ImportLegacyOptions, import_legacy};
use sqlx::{Connection as _, SqliteConnection, sqlite::SqliteConnectOptions};

async fn create_legacy_db(legacy_db_path: &Path, repo_root: &Path) {
    let connect_options = SqliteConnectOptions::new()
        .filename(legacy_db_path)
        .create_if_missing(true);

    let mut conn = SqliteConnection::connect_with(&connect_options)
        .await
        .unwrap();

    // Minimal subset of the legacy schema needed by the importer.
    sqlx::query(r#"CREATE TABLE alembic_version (version_num TEXT NOT NULL)"#)
        .execute(&mut conn)
        .await
        .unwrap();
    sqlx::query(r#"INSERT INTO alembic_version (version_num) VALUES ('test_head')"#)
        .execute(&mut conn)
        .await
        .unwrap();

    sqlx::query(
        r#"
        CREATE TABLE repositories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT NOT NULL,
            repo_id TEXT NOT NULL,
            repo_root TEXT NOT NULL
        )
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();

    sqlx::query(
        r#"
        CREATE TABLE epics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repository_id INTEGER NOT NULL,
            slug TEXT NOT NULL,
            name TEXT NOT NULL
        )
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();

    sqlx::query(
        r#"
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epic_id INTEGER NOT NULL,
            parent_task_id INTEGER,
            branch_name TEXT,
            title TEXT NOT NULL,
            local_path TEXT,
            merge_ready_at TEXT,
            state TEXT NOT NULL
        )
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();

    sqlx::query(
        r#"
        CREATE TABLE agent_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            agent_preview TEXT NOT NULL
        )
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();

    let repo_root_str = repo_root.display().to_string();

    sqlx::query(
        r#"
        INSERT INTO repositories (id, workspace_id, repo_id, repo_root)
        VALUES (1, 'default', 'deadbeefcafef00d', ?1)
        "#,
    )
    .bind(&repo_root_str)
    .execute(&mut conn)
    .await
    .unwrap();

    sqlx::query(
        r#"
        INSERT INTO epics (id, repository_id, slug, name)
        VALUES (10, 1, 'gpui', 'GPUI + Rust Port')
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();

    // Parent task (T-1)
    sqlx::query(
        r#"
        INSERT INTO tasks (
            id,
            epic_id,
            parent_task_id,
            branch_name,
            title,
            local_path,
            merge_ready_at,
            state
        )
        VALUES (
            100,
            10,
            NULL,
            'rn/gpui/T-1-rust-workspace-bootstrap',
            'T-1 Rust workspace bootstrap + crate boundaries',
            'epics/gpui/tasks/T-1/README.md',
            NULL,
            'todo'
        )
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();

    // Child task (T-2)
    sqlx::query(
        r#"
        INSERT INTO tasks (
            id,
            epic_id,
            parent_task_id,
            branch_name,
            title,
            local_path,
            merge_ready_at,
            state
        )
        VALUES (
            101,
            10,
            100,
            'rn/gpui/T-2-ulid',
            'T-2 ULID newtypes everywhere',
            'epics/gpui/tasks/T-2/README.md',
            '2026-01-20T00:00:00Z',
            'todo'
        )
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();

    // Two sessions for the same task; importer should pick the latest (highest id).
    sqlx::query(
        r#"
        INSERT INTO agent_sessions (id, task_id, agent_preview)
        VALUES (
            200,
            101,
            '{"last_assistant_message_preview":"old preview","last_message_turn_id":"turn-old","last_assistant_message_at":"2026-01-20T00:00:00Z"}'
        )
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();

    sqlx::query(
        r#"
        INSERT INTO agent_sessions (id, task_id, agent_preview)
        VALUES (
            201,
            101,
            '{"last_assistant_message_preview":"new preview","last_message_turn_id":"turn-new","last_assistant_message_at":"2026-01-20T00:00:01Z"}'
        )
        "#,
    )
    .execute(&mut conn)
    .await
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn imports_legacy_db_idempotently_and_does_not_mutate_legacy_db_file() {
    redesmyn_logging::init();

    let dir = tempfile::tempdir().unwrap();
    let repo_root = dir.path().join("repo");
    std::fs::create_dir_all(repo_root.join(".redesmyn")).unwrap();

    // The importer matches legacy `repositories.repo_root` to `repo_root.display()`.
    std::fs::create_dir_all(repo_root.join(".git")).unwrap();

    let legacy_db_path = repo_root.join(".redesmyn").join("redesmyn.sqlite3");
    let rust_db_path = repo_root.join(".redesmyn").join("redesmyn_rust.sqlite3");

    create_legacy_db(&legacy_db_path, &repo_root).await;

    let legacy_bytes_before = std::fs::read(&legacy_db_path).unwrap();

    let options = ImportLegacyOptions {
        repo_root: repo_root.clone(),
        legacy_db_path: legacy_db_path.clone(),
        rust_db_path: rust_db_path.clone(),
        dry_run: false,
    };

    let first = import_legacy(options.clone()).await.unwrap();
    assert!(first.applied.is_some());
    assert_eq!(first.planned.epics, 1);
    assert_eq!(first.planned.tasks, 2);
    assert_eq!(first.planned.session_previews, 1);

    let legacy_bytes_after = std::fs::read(&legacy_db_path).unwrap();
    assert_eq!(legacy_bytes_after, legacy_bytes_before);

    // Verify imported counts and key invariants.
    {
        let pool = redesmyn_storage::open_sqlite_pool(&rust_db_path)
            .await
            .unwrap();

        let workspaces: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM workspaces")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(workspaces, 1);

        let repositories: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM repositories")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(repositories, 1);

        let epics: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM epics")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(epics, 1);

        let tasks: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tasks")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(tasks, 2);

        let session_events: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM session_events")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(session_events, 1);

        let legacy_id_map: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM legacy_id_map")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(legacy_id_map, 6);

        let parent_id: TaskId = sqlx::query_scalar("SELECT id FROM tasks WHERE local_ref = 'T-1'")
            .fetch_one(&pool)
            .await
            .unwrap();
        let child_id: TaskId = sqlx::query_scalar("SELECT id FROM tasks WHERE local_ref = 'T-2'")
            .fetch_one(&pool)
            .await
            .unwrap();

        let child_parent: Option<TaskId> =
            sqlx::query_scalar("SELECT parent_task_id FROM tasks WHERE id = ?1")
                .bind(child_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(child_parent, Some(parent_id));

        let merge_readiness: String =
            sqlx::query_scalar("SELECT merge_readiness FROM tasks WHERE id = ?1")
                .bind(child_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(merge_readiness, "ready");

        let has_state: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM pragma_table_info('tasks') WHERE name = 'state' LIMIT 1",
        )
        .fetch_optional(&pool)
        .await
        .unwrap();
        if has_state.is_some() {
            let state: String = sqlx::query_scalar("SELECT state FROM tasks WHERE id = ?1")
                .bind(child_id)
                .fetch_one(&pool)
                .await
                .unwrap();
            assert_eq!(state, "todo");
        }

        let preview: String =
            sqlx::query_scalar("SELECT message_preview FROM session_events LIMIT 1")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(preview, "new preview");

        let has_agent_sessions: Option<i64> = sqlx::query_scalar(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'agent_sessions' LIMIT 1",
        )
        .fetch_optional(&pool)
        .await
        .unwrap();
        if has_agent_sessions.is_some() {
            let sessions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM agent_sessions")
                .fetch_one(&pool)
                .await
                .unwrap();
            assert_eq!(sessions, 1);
        }
    }

    // Re-running is idempotent (no duplicates).
    let _second = import_legacy(options).await.unwrap();

    let pool = redesmyn_storage::open_sqlite_pool(&rust_db_path)
        .await
        .unwrap();
    let workspaces: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM workspaces")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(workspaces, 1);

    let repositories: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM repositories")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(repositories, 1);

    let epics: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM epics")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(epics, 1);

    let tasks: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tasks")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(tasks, 2);

    let session_events: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM session_events")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(session_events, 1);

    let legacy_id_map: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM legacy_id_map")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(legacy_id_map, 6);
}
