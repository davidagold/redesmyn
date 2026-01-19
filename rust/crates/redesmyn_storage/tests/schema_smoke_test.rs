use redesmyn_ids::{
    ArtifactId, CommandId, CommandUpdateId, EpicId, EventId, HostId, HostInstanceId, RepoId,
    SessionEventId, SessionId, TaskId, TaskRelationId, WorkspaceId,
};
use redesmyn_storage::{
    events::{EventRecord, EventScope, get_event, insert_event},
    in_transaction, open_test_sqlite_pool,
};

#[tokio::test(flavor = "current_thread")]
async fn can_insert_and_query_core_schema() {
    redesmyn_logging::init();

    let pool = open_test_sqlite_pool().await.unwrap();

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();
    let parent_task_id = TaskId::new();
    let child_task_id = TaskId::new();
    let relation_id = TaskRelationId::new();

    let command_id = CommandId::new();
    let command_update_id = CommandUpdateId::new();

    let artifact_id = ArtifactId::new();
    let session_id = SessionId::new();
    let session_event_id = SessionEventId::new();

    let host_id = HostId::new();
    let host_instance_id = HostInstanceId::new();

    let t0_ms: i64 = 1_700_000_000_000;

    in_transaction(&pool, |conn| {
        Box::pin(async move {
            sqlx::query(
                r#"
                INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
                VALUES (?1, ?2, ?3, ?4)
                "#,
            )
            .bind(workspace_id)
            .bind(t0_ms)
            .bind(t0_ms)
            .bind("local")
            .execute(&mut *conn)
            .await?;

            sqlx::query(
                r#"
                INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                "#,
            )
            .bind(repo_id)
            .bind(workspace_id)
            .bind(t0_ms)
            .bind(t0_ms)
            .bind("repo")
            .bind("Repo")
            .execute(&mut *conn)
            .await?;

            sqlx::query(
                r#"
                INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                "#,
            )
            .bind(epic_id)
            .bind(repo_id)
            .bind(t0_ms)
            .bind(t0_ms)
            .bind("gpui")
            .bind("GPUI")
            .execute(&mut *conn)
            .await?;

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
                    merge_readiness
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                "#,
            )
            .bind(parent_task_id)
            .bind(epic_id)
            .bind(None::<TaskId>)
            .bind(t0_ms)
            .bind(t0_ms)
            .bind("T-17")
            .bind("Parent task")
            .bind("rn/gpui/T-17-parent")
            .bind("unknown")
            .execute(&mut *conn)
            .await?;

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
                    merge_readiness
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                "#,
            )
            .bind(child_task_id)
            .bind(epic_id)
            .bind(parent_task_id)
            .bind(t0_ms + 1)
            .bind(t0_ms + 1)
            .bind(None::<String>)
            .bind("Child task")
            .bind("rn/gpui/T-17-child")
            .bind("unknown")
            .execute(&mut *conn)
            .await?;

            sqlx::query(
                r#"
                INSERT INTO task_relations (id, created_at_ms, from_task_id, to_task_id, kind)
                VALUES (?1, ?2, ?3, ?4, ?5)
                "#,
            )
            .bind(relation_id)
            .bind(t0_ms)
            .bind(parent_task_id)
            .bind(child_task_id)
            .bind("after")
            .execute(&mut *conn)
            .await?;

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
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                "#,
            )
            .bind(command_id)
            .bind(t0_ms)
            .bind(t0_ms)
            .bind("repo")
            .bind(workspace_id)
            .bind(repo_id)
            .bind(child_task_id)
            .bind("task.merge")
            .bind("accepted")
            .bind(Vec::<u8>::new())
            .execute(&mut *conn)
            .await?;

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
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                "#,
            )
            .bind(command_update_id)
            .bind(command_id)
            .bind(t0_ms + 2)
            .bind("running")
            .bind("starting")
            .bind(0_i64)
            .bind(1_i64)
            .bind(None::<Vec<u8>>)
            .execute(&mut *conn)
            .await?;

            sqlx::query(
                r#"
                INSERT INTO artifacts (
                    id,
                    created_at_ms,
                    scope_kind,
                    scope_workspace_id,
                    scope_repo_id,
                    kind,
                    content_hash,
                    byte_len,
                    mime,
                    storage_hint
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                "#,
            )
            .bind(artifact_id)
            .bind(t0_ms)
            .bind("repo")
            .bind(workspace_id)
            .bind(repo_id)
            .bind("log")
            .bind(None::<String>)
            .bind(Some(3_i64))
            .bind(Some("text/plain"))
            .bind(Some("local://artifact/1"))
            .execute(&mut *conn)
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
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                "#,
            )
            .bind(session_event_id)
            .bind(session_id)
            .bind(t0_ms)
            .bind("task")
            .bind(workspace_id)
            .bind(repo_id)
            .bind(None::<EpicId>)
            .bind(child_task_id)
            .bind("AssistantMessage")
            .bind(None::<String>)
            .bind(Some("hi"))
            .bind(Some(artifact_id))
            .bind(Vec::<u8>::new())
            .execute(&mut *conn)
            .await?;

            sqlx::query(
                r#"
                INSERT INTO hosts (id, created_at_ms, hostname)
                VALUES (?1, ?2, ?3)
                "#,
            )
            .bind(host_id)
            .bind(t0_ms)
            .bind("localhost")
            .execute(&mut *conn)
            .await?;

            sqlx::query(
                r#"
                INSERT INTO daemon_presence (
                    host_instance_id,
                    host_id,
                    connected_at_ms,
                    last_heartbeat_at_ms,
                    disconnected_at_ms
                )
                VALUES (?1, ?2, ?3, ?4, ?5)
                "#,
            )
            .bind(host_instance_id)
            .bind(host_id)
            .bind(t0_ms)
            .bind(t0_ms + 5)
            .bind(None::<i64>)
            .execute(&mut *conn)
            .await?;

            let event = EventRecord {
                id: EventId::new(),
                created_at_ms: t0_ms,
                scope: EventScope::Repo {
                    workspace_id,
                    repo_id,
                },
                kind: "test.event".to_owned(),
                payload: vec![1, 2, 3],
            };
            insert_event(&mut *conn, &event).await?;

            Ok(event)
        })
    })
    .await
    .unwrap();

    let (task_count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM tasks WHERE epic_id = ?1")
        .bind(epic_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(task_count, 2);

    let (child_count,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM tasks WHERE parent_task_id = ?1")
            .bind(parent_task_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(child_count, 1);

    let (loaded_task_id,): (TaskId,) =
        sqlx::query_as("SELECT id FROM tasks WHERE epic_id = ?1 AND local_ref = ?2")
            .bind(epic_id)
            .bind("T-17")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(loaded_task_id, parent_task_id);

    let (loaded_command_id,): (CommandId,) = sqlx::query_as(
        r#"
        SELECT id
        FROM commands
        WHERE
            scope_kind = 'repo'
            AND scope_workspace_id = ?1
            AND scope_repo_id = ?2
            AND state = 'accepted'
        LIMIT 1
        "#,
    )
    .bind(workspace_id)
    .bind(repo_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(loaded_command_id, command_id);

    let (update_count,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM command_updates WHERE command_id = ?1")
            .bind(command_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(update_count, 1);

    let (session_event_count,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM session_events WHERE session_id = ?1")
            .bind(session_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(session_event_count, 1);

    let (presence_count,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM daemon_presence WHERE host_id = ?1")
            .bind(host_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(presence_count, 1);

    let (event_id,): (EventId,) = sqlx::query_as(
        r#"
        SELECT id
        FROM events
        WHERE
            scope_kind = 'repo'
            AND scope_workspace_id = ?1
            AND scope_repo_id = ?2
        LIMIT 1
        "#,
    )
    .bind(workspace_id)
    .bind(repo_id)
    .fetch_one(&pool)
    .await
    .unwrap();

    let loaded_event = get_event(&pool, event_id).await.unwrap().unwrap();
    assert_eq!(loaded_event.scope, EventScope::Repo { workspace_id, repo_id });
    assert_eq!(loaded_event.created_at_ms, t0_ms);
    assert_eq!(loaded_event.kind, "test.event");
    assert_eq!(loaded_event.payload, vec![1, 2, 3]);
}
