use redesmyn_control_plane::ControlPlane;
use redesmyn_ids::{EpicId, RepoId, TaskId, WorkspaceId};
use redesmyn_protocol::client::{AgentKind, AgentMessageConflictAction, StartAgentRequest};
use redesmyn_protocol::ErrorCategory;

async fn seed_repo_and_task(control_plane: &ControlPlane) -> (WorkspaceId, RepoId, TaskId) {
    let pool = control_plane.pool();

    let now_ms = 1_i64;
    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();
    let task_id = TaskId::new();

    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, ?4)
        "#,
    )
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("test")
    .execute(pool)
    .await
    .expect("insert workspace");

    sqlx::query(
        r#"
        INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
    )
    .bind(repo_id)
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("repo")
    .bind("Repo")
    .execute(pool)
    .await
    .expect("insert repo");

    sqlx::query(
        r#"
        INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
    )
    .bind(epic_id)
    .bind(repo_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("epic")
    .bind("Epic")
    .execute(pool)
    .await
    .expect("insert epic");

    sqlx::query(
        r#"
        INSERT INTO tasks (id, epic_id, created_at_ms, updated_at_ms, title, merge_readiness)
        VALUES (?1, ?2, ?3, ?4, ?5, 'unknown')
        "#,
    )
    .bind(task_id)
    .bind(epic_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("Task")
    .execute(pool)
    .await
    .expect("insert task");

    (workspace_id, repo_id, task_id)
}

#[tokio::test]
async fn start_agent_returns_unavailable_and_rolls_back_session_when_no_daemon_is_connected() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, task_id) = seed_repo_and_task(&control_plane).await;

    let result = control_plane
        .start_agent(
            workspace_id,
            repo_id,
            StartAgentRequest {
                task_id,
                agent_kind: AgentKind::Codex,
                initial_prompt: None,
                on_conflict: AgentMessageConflictAction::Fail,
            },
        )
        .await;

    let err = result.expect_err("expected start-agent failure without daemon");
    assert_eq!(err.category, ErrorCategory::Unavailable);
    assert!(
        err.message.contains("No daemon connection is available"),
        "unexpected start-agent error message: {}",
        err.message
    );

    let active_sessions: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM agent_sessions
        WHERE task_id = ?1 AND ended_at_ms IS NULL
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("count active sessions");
    assert_eq!(active_sessions, 0, "expected no active task sessions");

    let stopped_sessions: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM agent_sessions
        WHERE task_id = ?1 AND status = 'stopped' AND ended_at_ms IS NOT NULL
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("count stopped sessions");
    assert_eq!(
        stopped_sessions, 1,
        "expected a rolled-back stopped session"
    );
}
