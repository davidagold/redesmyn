use redesmyn_ids::{
    CommandId, CommandUpdateId, EpicId, EventId, HostId, HostInstanceId, RepoId, SessionEventId,
    SessionId, TaskId, WorkspaceId,
};
use redesmyn_storage::director_mode::{
    DirectorModeLifecycle, MergeAuthorityPolicySource, set_epic_merge_authority_override,
};
use redesmyn_storage::epic_graph::load_epic_graph;
use redesmyn_storage::open_test_sqlite_pool;

#[tokio::test]
async fn epic_graph_query_returns_compact_projection() {
    let pool = open_test_sqlite_pool().await.expect("db");
    let now_ms: i64 = 1_700_000_000_000;

    let workspace_id = WorkspaceId::new();
    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, ?4)
        "#,
    )
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("dev")
    .execute(&pool)
    .await
    .expect("insert workspace");

    let repo_id = RepoId::new();
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
    .bind("demo")
    .bind("Demo Repo")
    .execute(&pool)
    .await
    .expect("insert repo");

    let epic_id = EpicId::new();
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
    .bind("gpui")
    .bind("GPUI + Rust Port")
    .execute(&pool)
    .await
    .expect("insert epic");

    let parent_task_id = TaskId::new();
    let child_task_id = TaskId::new();

    sqlx::query(
        r#"
        INSERT INTO tasks (
            id,
            epic_id,
            parent_task_id,
            created_at_ms,
            updated_at_ms,
            local_ref,
            title,
            branch_name,
            merge_readiness,
            state
        )
        VALUES (?1, ?2, NULL, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#,
    )
    .bind(parent_task_id)
    .bind(epic_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("T-1")
    .bind("Root task")
    .bind(Option::<String>::None)
    .bind("unknown")
    .bind("todo")
    .execute(&pool)
    .await
    .expect("insert parent task");

    sqlx::query(
        r#"
        INSERT INTO tasks (
            id,
            epic_id,
            parent_task_id,
            created_at_ms,
            updated_at_ms,
            local_ref,
            title,
            branch_name,
            merge_readiness,
            state
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        "#,
    )
    .bind(child_task_id)
    .bind(epic_id)
    .bind(parent_task_id)
    .bind(now_ms + 1)
    .bind(now_ms + 1)
    .bind("T-2")
    .bind("Child task")
    .bind(Some("feature/t-2"))
    .bind("ready")
    .bind("in_progress")
    .execute(&pool)
    .await
    .expect("insert child task");

    let other_epic_id = EpicId::new();
    sqlx::query(
        r#"
        INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
    )
    .bind(other_epic_id)
    .bind(repo_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("other")
    .bind("Other Epic")
    .execute(&pool)
    .await
    .expect("insert other epic");

    let other_task_id = TaskId::new();
    sqlx::query(
        r#"
        INSERT INTO tasks (
            id,
            epic_id,
            parent_task_id,
            created_at_ms,
            updated_at_ms,
            local_ref,
            title,
            branch_name,
            merge_readiness,
            state
        )
        VALUES (?1, ?2, NULL, ?3, ?4, ?5, ?6, NULL, 'unknown', 'todo')
        "#,
    )
    .bind(other_task_id)
    .bind(other_epic_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("X-1")
    .bind("Other task")
    .execute(&pool)
    .await
    .expect("insert other task");

    let command_id = CommandId::new();
    sqlx::query(
        r#"
        INSERT INTO commands (
            id,
            created_at_ms,
            updated_at_ms,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            target_task_id,
            kind,
            state,
            payload
        )
        VALUES (?1, ?2, ?3, 'repo', ?4, ?5, ?6, ?7, ?8, ?9)
        "#,
    )
    .bind(command_id)
    .bind(now_ms)
    .bind(now_ms + 10)
    .bind(workspace_id)
    .bind(repo_id)
    .bind(child_task_id)
    .bind("task.agent.start")
    .bind("running")
    .bind(Vec::<u8>::new())
    .execute(&pool)
    .await
    .expect("insert command");

    let other_command_id = CommandId::new();
    sqlx::query(
        r#"
        INSERT INTO commands (
            id,
            created_at_ms,
            updated_at_ms,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            target_task_id,
            kind,
            state,
            payload
        )
        VALUES (?1, ?2, ?3, 'repo', ?4, ?5, ?6, ?7, ?8, ?9)
        "#,
    )
    .bind(other_command_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind(workspace_id)
    .bind(repo_id)
    .bind(other_task_id)
    .bind("task.agent.start")
    .bind("running")
    .bind(Vec::<u8>::new())
    .execute(&pool)
    .await
    .expect("insert other command");

    let update_id_old = CommandUpdateId::new();
    let update_id_new = CommandUpdateId::new();
    sqlx::query(
        r#"
        INSERT INTO command_updates (
            id,
            command_id,
            created_at_ms,
            state,
            message,
            progress_current,
            progress_total,
            detail
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL)
        "#,
    )
    .bind(update_id_old)
    .bind(command_id)
    .bind(now_ms + 2)
    .bind("running")
    .bind(Some("working..."))
    .bind(Some(1_i64))
    .bind(Some(10_i64))
    .execute(&pool)
    .await
    .expect("insert old update");

    sqlx::query(
        r#"
        INSERT INTO command_updates (
            id,
            command_id,
            created_at_ms,
            state,
            message,
            progress_current,
            progress_total,
            detail
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL)
        "#,
    )
    .bind(update_id_new)
    .bind(command_id)
    .bind(now_ms + 3)
    .bind("running")
    .bind(Some("still working..."))
    .bind(Some(2_i64))
    .bind(Some(10_i64))
    .execute(&pool)
    .await
    .expect("insert new update");

    let host_id = HostId::new();
    let host_instance_id = HostInstanceId::new();
    sqlx::query(
        r#"
        INSERT INTO hosts (id, created_at_ms, hostname)
        VALUES (?1, ?2, ?3)
        "#,
    )
    .bind(host_id)
    .bind(now_ms)
    .bind(Some("devbox"))
    .execute(&pool)
    .await
    .expect("insert host");

    sqlx::query(
        r#"
        INSERT INTO daemon_presence (
            host_instance_id,
            host_id,
            connected_at_ms,
            last_heartbeat_at_ms,
            disconnected_at_ms
        )
        VALUES (?1, ?2, ?3, ?4, NULL)
        "#,
    )
    .bind(host_instance_id)
    .bind(host_id)
    .bind(now_ms)
    .bind(now_ms + 5)
    .execute(&pool)
    .await
    .expect("insert daemon presence");

    let session_id = SessionId::new();
    let session_event_id_old = SessionEventId::new();
    let session_event_id_new = SessionEventId::new();
    insert_session_event(
        &pool,
        session_event_id_old,
        session_id,
        now_ms + 1,
        workspace_id,
        repo_id,
        epic_id,
        child_task_id,
        "turn.started",
        Some("starting"),
    )
    .await
    .expect("insert old session event");

    insert_session_event(
        &pool,
        session_event_id_new,
        session_id,
        now_ms + 4,
        workspace_id,
        repo_id,
        epic_id,
        child_task_id,
        "turn.completed",
        Some("done"),
    )
    .await
    .expect("insert new session event");

    let event_id = EventId::new();
    sqlx::query(
        r#"
        INSERT INTO events (
            id,
            created_at_ms,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            kind,
            payload
        )
        VALUES (?1, ?2, 'repo', ?3, ?4, ?5, X'')
        "#,
    )
    .bind(event_id)
    .bind(now_ms + 6)
    .bind(workspace_id)
    .bind(repo_id)
    .bind("event_log.appended")
    .execute(&pool)
    .await
    .expect("insert event");

    let graph = load_epic_graph(&pool, "gpui", None)
        .await
        .expect("load graph")
        .expect("graph present");

    assert_eq!(graph.epic.epic_id, epic_id);
    assert_eq!(graph.epic.slug, "gpui");
    assert_eq!(graph.tasks.len(), 2);
    assert!(
        graph
            .tasks
            .iter()
            .any(|task| task.task_id == parent_task_id)
    );
    assert!(graph.tasks.iter().any(|task| task.task_id == child_task_id));

    assert_eq!(
        graph.commands.len(),
        1,
        "should filter commands by epic tasks"
    );
    assert_eq!(graph.commands[0].command_id, command_id);

    let last_update = graph
        .command_last_updates
        .get(&command_id)
        .expect("last update present");
    assert_eq!(last_update.update_id, update_id_new);

    assert_eq!(graph.daemon_presences.len(), 1);
    assert_eq!(graph.daemon_presences[0].host_instance_id, host_instance_id);

    assert_eq!(graph.session_summaries.len(), 1);
    assert_eq!(
        graph.session_summaries[0].session_event_id,
        session_event_id_new
    );
    assert_eq!(
        graph.director_mode.lifecycle,
        DirectorModeLifecycle::Inactive
    );
    assert_eq!(graph.director_mode.director_session_id, None);
    assert_eq!(graph.merge_authority_policy.yolo_merge, false);
    assert_eq!(
        graph.merge_authority_policy.source,
        MergeAuthorityPolicySource::GlobalDefault
    );

    set_epic_merge_authority_override(&pool, epic_id, true)
        .await
        .expect("set epic policy override");
    let graph_with_override = load_epic_graph(&pool, "gpui", None)
        .await
        .expect("reload graph")
        .expect("graph with override present");
    assert_eq!(graph_with_override.merge_authority_policy.yolo_merge, true);
    assert_eq!(
        graph_with_override.merge_authority_policy.source,
        MergeAuthorityPolicySource::EpicOverride
    );

    assert_eq!(graph.as_of_event_id, Some(event_id));
}

async fn insert_session_event(
    pool: &sqlx::SqlitePool,
    session_event_id: SessionEventId,
    session_id: SessionId,
    created_at_ms: i64,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    epic_id: EpicId,
    task_id: TaskId,
    kind: &str,
    message_preview: Option<&str>,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
        INSERT OR IGNORE INTO agent_sessions (
            session_id,
            created_at_ms,
            updated_at_ms,
            scope_workspace_id,
            scope_repo_id,
            scope_kind,
            task_id,
            agent_kind,
            status
        )
        VALUES (?1, ?2, ?3, ?4, ?5, 'task', ?6, 'shell', 'stopped')
        "#,
    )
    .bind(session_id)
    .bind(created_at_ms)
    .bind(created_at_ms)
    .bind(workspace_id)
    .bind(repo_id)
    .bind(task_id)
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        INSERT INTO session_events (
            id,
            session_id,
            created_at_ms,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            epic_id,
            task_id,
            kind,
            turn_id,
            message_preview,
            artifact_id,
            payload
        )
        VALUES (?1, ?2, ?3, 'task', ?4, ?5, ?6, ?7, ?8, NULL, ?9, NULL, X'')
        "#,
    )
    .bind(session_event_id)
    .bind(session_id)
    .bind(created_at_ms)
    .bind(workspace_id)
    .bind(repo_id)
    .bind(epic_id)
    .bind(task_id)
    .bind(kind)
    .bind(message_preview)
    .execute(pool)
    .await?;

    Ok(())
}
