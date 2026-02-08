use tokio::sync::mpsc;

use redesmyn_control_plane::ControlPlane;
use redesmyn_ids::{HostId, HostInstanceId, RepoId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::ProtocolVersion;
use redesmyn_protocol::agent_commands::{StartTaskAgentSessionCommand, TASK_AGENT_START};
use redesmyn_storage::commands::CommandScope;

fn now_ms() -> i64 {
    1_700_000_000_000
}

async fn seed_repo_task(
    pool: &sqlx::SqlitePool,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
) {
    let epic_id = redesmyn_ids::EpicId::new();
    let now = now_ms();

    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, 'dev')
        "#,
    )
    .bind(workspace_id)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await
    .expect("insert workspace");

    sqlx::query(
        r#"
        INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, ?3, ?4, 'demo', 'Demo Repo')
        "#,
    )
    .bind(repo_id)
    .bind(workspace_id)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await
    .expect("insert repo");

    sqlx::query(
        r#"
        INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, ?3, ?4, 'gpui', 'GPUI')
        "#,
    )
    .bind(epic_id)
    .bind(repo_id)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await
    .expect("insert epic");

    sqlx::query(
        r#"
        INSERT INTO tasks (
            id, epic_id, parent_task_id, created_at_ms, updated_at_ms, local_ref, title, branch_name, merge_readiness, state
        )
        VALUES (?1, ?2, NULL, ?3, ?4, 'T-1', 'Task one', 'rn/gpui/T-1', 'unknown', 'in_progress')
        "#,
    )
    .bind(task_id)
    .bind(epic_id)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await
    .expect("insert task");
}

#[tokio::test]
async fn register_daemon_presence_reconciles_sessions_from_stale_instance() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let pool = control_plane.pool().clone();

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let task_id = TaskId::new();
    seed_repo_task(&pool, workspace_id, repo_id, task_id).await;

    let host_id = HostId::new();
    let stale_instance_id = HostInstanceId::new();
    let now = now_ms();

    sqlx::query("INSERT INTO hosts (id, created_at_ms, hostname) VALUES (?1, ?2, 'devbox')")
        .bind(host_id)
        .bind(now)
        .execute(&pool)
        .await
        .expect("insert host");

    sqlx::query(
        r#"
        INSERT INTO daemon_presence (
            host_instance_id, host_id, connected_at_ms, last_heartbeat_at_ms, disconnected_at_ms
        )
        VALUES (?1, ?2, ?3, ?3, NULL)
        "#,
    )
    .bind(stale_instance_id)
    .bind(host_id)
    .bind(now)
    .execute(&pool)
    .await
    .expect("insert stale daemon presence");

    let session_id = SessionId::new();
    sqlx::query(
        r#"
        INSERT INTO agent_sessions (
            session_id,
            created_at_ms,
            updated_at_ms,
            scope_workspace_id,
            scope_repo_id,
            scope_kind,
            task_id,
            agent_kind,
            status,
            external_session_ref,
            title,
            started_at_ms,
            ended_at_ms,
            closed_at_ms,
            runner_host_id,
            runner_host_instance_id
        )
        VALUES (
            ?1, ?2, ?2, ?3, ?4, 'task', ?5, 'codex', 'running', '{"type":"none"}',
            NULL, ?2, NULL, NULL, ?6, ?7
        )
        "#,
    )
    .bind(session_id)
    .bind(now)
    .bind(workspace_id)
    .bind(repo_id)
    .bind(task_id)
    .bind(host_id)
    .bind(stale_instance_id)
    .execute(&pool)
    .await
    .expect("insert running task session");

    let new_instance_id = HostInstanceId::new();
    control_plane
        .register_daemon_presence(host_id, new_instance_id)
        .await
        .expect("register daemon presence");

    let (status, ended_at_ms): (String, Option<i64>) = sqlx::query_as(
        "SELECT status, ended_at_ms FROM agent_sessions WHERE session_id = ?1",
    )
    .bind(session_id)
    .fetch_one(&pool)
    .await
    .expect("fetch reconciled session");
    assert_eq!(status, "error");
    assert!(ended_at_ms.is_some());

    let disconnected_at_ms: Option<i64> = sqlx::query_scalar(
        "SELECT disconnected_at_ms FROM daemon_presence WHERE host_instance_id = ?1",
    )
    .bind(stale_instance_id)
    .fetch_one(&pool)
    .await
    .expect("fetch stale presence");
    assert!(disconnected_at_ms.is_some());

    let (kind, state): (String, String) = sqlx::query_as(
        r#"
        SELECT kind, state
        FROM commands
        WHERE target_task_id = ?1
        ORDER BY created_at_ms DESC, id DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_one(&pool)
    .await
    .expect("fetch reconcile command");
    assert_eq!(kind, "task.agent.stop");
    assert_eq!(state, "failed");
}

#[tokio::test]
async fn task_agent_start_dispatch_assigns_runner_ownership() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let pool = control_plane.pool().clone();

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let task_id = TaskId::new();
    seed_repo_task(&pool, workspace_id, repo_id, task_id).await;

    let host_id = HostId::new();
    let host_instance_id = HostInstanceId::new();
    control_plane
        .register_daemon_presence(host_id, host_instance_id)
        .await
        .expect("register daemon");

    let (outbound_tx, _outbound_rx) = mpsc::channel(8);
    control_plane
        .daemons()
        .register_connection(host_id, host_instance_id, ProtocolVersion::CURRENT, outbound_tx)
        .await;

    let session_id = SessionId::new();
    sqlx::query(
        r#"
        INSERT INTO agent_sessions (
            session_id,
            created_at_ms,
            updated_at_ms,
            scope_workspace_id,
            scope_repo_id,
            scope_kind,
            task_id,
            agent_kind,
            status,
            external_session_ref,
            title,
            started_at_ms,
            ended_at_ms,
            closed_at_ms
        )
        VALUES (
            ?1, ?2, ?2, ?3, ?4, 'task', ?5, 'codex', 'running', '{"type":"none"}',
            NULL, ?2, NULL, NULL
        )
        "#,
    )
    .bind(session_id)
    .bind(now_ms())
    .bind(workspace_id)
    .bind(repo_id)
    .bind(task_id)
    .execute(&pool)
    .await
    .expect("insert task session");

    let payload = serde_json::to_vec(&StartTaskAgentSessionCommand {
        session_id,
        task_id,
        task_branch_name: Some("rn/gpui/T-1".to_string()),
        agent_kind: redesmyn_protocol::client::AgentKind::Codex,
        initial_prompt: None,
        policy_snapshot: None,
        stop_session_ids: Vec::new(),
    })
    .expect("encode payload");

    let _command = control_plane
        .issue_command(
            CommandScope::Repo {
                workspace_id,
                repo_id,
            },
            TASK_AGENT_START.to_string(),
            Some(task_id),
            None,
            None,
            payload,
        )
        .await
        .expect("issue task.agent.start");

    let (runner_host_id, runner_host_instance_id): (Option<HostId>, Option<HostInstanceId>) =
        sqlx::query_as(
            r#"
            SELECT runner_host_id, runner_host_instance_id
            FROM agent_sessions
            WHERE session_id = ?1
            "#,
        )
        .bind(session_id)
        .fetch_one(&pool)
        .await
        .expect("fetch assigned runner ownership");
    assert_eq!(runner_host_id, Some(host_id));
    assert_eq!(runner_host_instance_id, Some(host_instance_id));
}

#[tokio::test]
async fn register_daemon_presence_reconciles_legacy_unowned_running_sessions() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let pool = control_plane.pool().clone();

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let task_id = TaskId::new();
    seed_repo_task(&pool, workspace_id, repo_id, task_id).await;

    let session_id = SessionId::new();
    sqlx::query(
        r#"
        INSERT INTO agent_sessions (
            session_id,
            created_at_ms,
            updated_at_ms,
            scope_workspace_id,
            scope_repo_id,
            scope_kind,
            task_id,
            agent_kind,
            status,
            external_session_ref,
            title,
            started_at_ms,
            ended_at_ms,
            closed_at_ms,
            runner_host_id,
            runner_host_instance_id
        )
        VALUES (
            ?1, ?2, ?2, ?3, ?4, 'task', ?5, 'codex', 'running', '{"type":"none"}',
            NULL, ?2, NULL, NULL, NULL, NULL
        )
        "#,
    )
    .bind(session_id)
    .bind(now_ms())
    .bind(workspace_id)
    .bind(repo_id)
    .bind(task_id)
    .execute(&pool)
    .await
    .expect("insert legacy running task session");

    control_plane
        .register_daemon_presence(HostId::new(), HostInstanceId::new())
        .await
        .expect("register daemon presence");

    let (status, ended_at_ms): (String, Option<i64>) = sqlx::query_as(
        "SELECT status, ended_at_ms FROM agent_sessions WHERE session_id = ?1",
    )
    .bind(session_id)
    .fetch_one(&pool)
    .await
    .expect("fetch reconciled legacy session");
    assert_eq!(status, "error");
    assert!(ended_at_ms.is_some());

    let (kind, state): (String, String) = sqlx::query_as(
        r#"
        SELECT kind, state
        FROM commands
        WHERE target_task_id = ?1
        ORDER BY created_at_ms DESC, id DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_one(&pool)
    .await
    .expect("fetch reconcile command");
    assert_eq!(kind, "task.agent.stop");
    assert_eq!(state, "failed");
}
