use redesmyn_ids::{
    ArtifactId, CommandId, CommandUpdateId, EpicId, EventId, HostId, HostInstanceId, RepoId,
    SessionEventId, SessionId, TaskId, TaskRelationId, WorkspaceId,
};
use redesmyn_storage::{
    StorageError,
    events::{EventRecord, EventScope, get_event, insert_event},
    in_transaction, open_test_sqlite_pool,
    schema::{CommandState, MergeReadiness, RepoScopeKind, SessionScopeKind, TaskRelationKind},
};
use sqlx::SqliteConnection;

async fn insert_task_agent_session(
    conn: &mut SqliteConnection,
    session_id: SessionId,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
    now_ms: i64,
) -> Result<(), StorageError> {
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
            interface_mode,
            status
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        "#,
    )
    .bind(session_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind(workspace_id)
    .bind(repo_id)
    .bind("task")
    .bind(task_id)
    .bind("shell")
    .bind("shell_tmux")
    .bind("stopped")
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_workspace(
    conn: &mut SqliteConnection,
    workspace_id: WorkspaceId,
    now_ms: i64,
    name: &str,
) -> Result<(), StorageError> {
    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, ?4)
        "#,
    )
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind(name)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_repository(
    conn: &mut SqliteConnection,
    repo_id: RepoId,
    workspace_id: WorkspaceId,
    now_ms: i64,
    slug: &str,
    title: &str,
) -> Result<(), StorageError> {
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
    .bind(slug)
    .bind(title)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_epic(
    conn: &mut SqliteConnection,
    epic_id: EpicId,
    repo_id: RepoId,
    now_ms: i64,
    slug: &str,
    title: &str,
) -> Result<(), StorageError> {
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
    .bind(slug)
    .bind(title)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_task(
    conn: &mut SqliteConnection,
    task_id: TaskId,
    epic_id: EpicId,
    parent_task_id: Option<TaskId>,
    now_ms: i64,
    local_ref: Option<&str>,
    title: &str,
    branch_name: Option<&str>,
    merge_readiness: MergeReadiness,
) -> Result<(), StorageError> {
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
    .bind(task_id)
    .bind(epic_id)
    .bind(parent_task_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind(local_ref)
    .bind(title)
    .bind(branch_name)
    .bind(merge_readiness.as_str())
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_task_relation(
    conn: &mut SqliteConnection,
    relation_id: TaskRelationId,
    now_ms: i64,
    from_task_id: TaskId,
    to_task_id: TaskId,
    kind: TaskRelationKind,
) -> Result<(), StorageError> {
    sqlx::query(
        r#"
        INSERT INTO task_relations (id, created_at_ms, from_task_id, to_task_id, kind)
        VALUES (?1, ?2, ?3, ?4, ?5)
        "#,
    )
    .bind(relation_id)
    .bind(now_ms)
    .bind(from_task_id)
    .bind(to_task_id)
    .bind(kind.as_str())
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_command(
    conn: &mut SqliteConnection,
    command_id: CommandId,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    target_task_id: Option<TaskId>,
    now_ms: i64,
    kind: &str,
    state: CommandState,
    payload: Vec<u8>,
) -> Result<(), StorageError> {
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
    .bind(now_ms)
    .bind(now_ms)
    .bind(RepoScopeKind::Repo.as_str())
    .bind(workspace_id)
    .bind(repo_id)
    .bind(target_task_id)
    .bind(kind)
    .bind(state.as_str())
    .bind(payload)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_command_update(
    conn: &mut SqliteConnection,
    update_id: CommandUpdateId,
    command_id: CommandId,
    now_ms: i64,
    state: CommandState,
    message: Option<&str>,
    progress_current: Option<i64>,
    progress_total: Option<i64>,
) -> Result<(), StorageError> {
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
    .bind(update_id)
    .bind(command_id)
    .bind(now_ms)
    .bind(state.as_str())
    .bind(message)
    .bind(progress_current)
    .bind(progress_total)
    .bind(None::<Vec<u8>>)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_artifact(
    conn: &mut SqliteConnection,
    artifact_id: ArtifactId,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    now_ms: i64,
    kind: &str,
) -> Result<(), StorageError> {
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
    .bind(now_ms)
    .bind(RepoScopeKind::Repo.as_str())
    .bind(workspace_id)
    .bind(repo_id)
    .bind(kind)
    .bind(None::<String>)
    .bind(Some(3_i64))
    .bind(Some("text/plain"))
    .bind(Some("local://artifact/1"))
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_task_scoped_session_event(
    conn: &mut SqliteConnection,
    session_event_id: SessionEventId,
    session_id: SessionId,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    epic_id: EpicId,
    task_id: TaskId,
    artifact_id: Option<ArtifactId>,
    now_ms: i64,
) -> Result<(), StorageError> {
    insert_task_agent_session(
        &mut *conn,
        session_id,
        workspace_id,
        repo_id,
        task_id,
        now_ms,
    )
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
    .bind(now_ms)
    .bind(SessionScopeKind::Task.as_str())
    .bind(workspace_id)
    .bind(repo_id)
    .bind(epic_id)
    .bind(task_id)
    .bind("AssistantMessage")
    .bind(None::<String>)
    .bind(Some("hi"))
    .bind(artifact_id)
    .bind(Vec::<u8>::new())
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_host(
    conn: &mut SqliteConnection,
    host_id: HostId,
    now_ms: i64,
    hostname: Option<&str>,
) -> Result<(), StorageError> {
    sqlx::query(
        r#"
        INSERT INTO hosts (id, created_at_ms, hostname)
        VALUES (?1, ?2, ?3)
        "#,
    )
    .bind(host_id)
    .bind(now_ms)
    .bind(hostname)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn insert_daemon_presence(
    conn: &mut SqliteConnection,
    host_instance_id: HostInstanceId,
    host_id: HostId,
    connected_at_ms: i64,
    last_heartbeat_at_ms: i64,
) -> Result<(), StorageError> {
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
    .bind(connected_at_ms)
    .bind(last_heartbeat_at_ms)
    .bind(None::<i64>)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

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

    let other_workspace_id = WorkspaceId::new();
    let other_repo_id = RepoId::new();
    let other_epic_id = EpicId::new();
    let other_task_id = TaskId::new();

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
            insert_workspace(&mut *conn, workspace_id, t0_ms, "local").await?;
            insert_repository(&mut *conn, repo_id, workspace_id, t0_ms, "repo", "Repo").await?;
            insert_epic(&mut *conn, epic_id, repo_id, t0_ms, "gpui", "GPUI").await?;

            insert_workspace(&mut *conn, other_workspace_id, t0_ms, "other").await?;
            insert_repository(
                &mut *conn,
                other_repo_id,
                other_workspace_id,
                t0_ms,
                "other-repo",
                "Other Repo",
            )
            .await?;
            insert_epic(
                &mut *conn,
                other_epic_id,
                other_repo_id,
                t0_ms,
                "other",
                "Other Epic",
            )
            .await?;

            insert_task(
                &mut *conn,
                parent_task_id,
                epic_id,
                None,
                t0_ms,
                Some("T-17"),
                "Parent task",
                Some("rn/gpui/T-17-parent"),
                MergeReadiness::Unknown,
            )
            .await?;
            insert_task(
                &mut *conn,
                child_task_id,
                epic_id,
                Some(parent_task_id),
                t0_ms + 1,
                None,
                "Child task",
                Some("rn/gpui/T-17-child"),
                MergeReadiness::Unknown,
            )
            .await?;

            insert_task(
                &mut *conn,
                other_task_id,
                other_epic_id,
                None,
                t0_ms,
                None,
                "Other task",
                None,
                MergeReadiness::Unknown,
            )
            .await?;

            insert_task_relation(
                &mut *conn,
                relation_id,
                t0_ms,
                parent_task_id,
                child_task_id,
                TaskRelationKind::After,
            )
            .await?;

            insert_command(
                &mut *conn,
                command_id,
                workspace_id,
                repo_id,
                Some(child_task_id),
                t0_ms,
                "task.merge",
                CommandState::Accepted,
                Vec::<u8>::new(),
            )
            .await?;
            insert_command_update(
                &mut *conn,
                command_update_id,
                command_id,
                t0_ms + 2,
                CommandState::Running,
                Some("starting"),
                Some(0),
                Some(1),
            )
            .await?;

            insert_artifact(&mut *conn, artifact_id, workspace_id, repo_id, t0_ms, "log").await?;
            insert_task_scoped_session_event(
                &mut *conn,
                session_event_id,
                session_id,
                workspace_id,
                repo_id,
                epic_id,
                child_task_id,
                Some(artifact_id),
                t0_ms,
            )
            .await?;

            insert_host(&mut *conn, host_id, t0_ms, Some("localhost")).await?;
            insert_daemon_presence(&mut *conn, host_instance_id, host_id, t0_ms, t0_ms + 5).await?;

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
    assert_eq!(
        loaded_event.scope,
        EventScope::Repo {
            workspace_id,
            repo_id
        }
    );
    assert_eq!(loaded_event.created_at_ms, t0_ms);
    assert_eq!(loaded_event.kind, "test.event");
    assert_eq!(loaded_event.payload, vec![1, 2, 3]);

    // Composite FK invariants: workspace/repo pairings must exist (prevents workspace A + repo B).
    let result = sqlx::query(
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
    .bind(CommandId::new())
    .bind(t0_ms)
    .bind(t0_ms)
    .bind(RepoScopeKind::Repo.as_str())
    .bind(workspace_id)
    .bind(other_repo_id)
    .bind(child_task_id)
    .bind("test.invalid_scope_pair")
    .bind(CommandState::Accepted.as_str())
    .bind(Vec::<u8>::new())
    .execute(&pool)
    .await;
    assert!(
        result.is_err(),
        "expected commands composite scope FK violation for workspace/repo pairing"
    );

    // Parent pointers must stay within the epic (no cross-epic parents).
    let result = sqlx::query(
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
    .bind(TaskId::new())
    .bind(epic_id)
    .bind(other_task_id)
    .bind(t0_ms)
    .bind(t0_ms)
    .bind(None::<String>)
    .bind("Invalid parent")
    .bind(None::<String>)
    .bind(MergeReadiness::Unknown.as_str())
    .execute(&pool)
    .await;
    assert!(
        result.is_err(),
        "expected tasks composite parent FK violation for cross-epic parent"
    );

    // Deleting a parent detaches children (no implicit subtree deletes).
    sqlx::query("DELETE FROM tasks WHERE id = ?1")
        .bind(parent_task_id)
        .execute(&pool)
        .await
        .unwrap();

    let (remaining_child_parent_id,): (Option<TaskId>,) =
        sqlx::query_as("SELECT parent_task_id FROM tasks WHERE id = ?1")
            .bind(child_task_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(remaining_child_parent_id, None);

    let (child_still_exists,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM tasks WHERE id = ?1")
        .bind(child_task_id)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(child_still_exists, 1);

    // Session scope chains must be consistent: repo -> epic, epic -> task.
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
            interface_mode,
            status
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        "#,
    )
    .bind(session_id)
    .bind(t0_ms)
    .bind(t0_ms)
    .bind(workspace_id)
    .bind(repo_id)
    .bind("chat")
    .bind(None::<TaskId>)
    .bind("shell")
    .bind("shell_tmux")
    .bind("stopped")
    .execute(&pool)
    .await
    .unwrap();

    let result = sqlx::query(
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
    .bind(SessionEventId::new())
    .bind(session_id)
    .bind(t0_ms)
    .bind(SessionScopeKind::Epic.as_str())
    .bind(workspace_id)
    .bind(repo_id)
    .bind(other_epic_id)
    .bind(None::<TaskId>)
    .bind("SessionStarted")
    .bind(None::<String>)
    .bind(None::<String>)
    .bind(None::<ArtifactId>)
    .bind(Vec::<u8>::new())
    .execute(&pool)
    .await;
    assert!(
        result.is_err(),
        "expected session_events repo->epic composite FK violation"
    );

    let result = sqlx::query(
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
    .bind(SessionEventId::new())
    .bind(session_id)
    .bind(t0_ms)
    .bind(SessionScopeKind::Task.as_str())
    .bind(workspace_id)
    .bind(repo_id)
    .bind(epic_id)
    .bind(other_task_id)
    .bind("UserMessage")
    .bind(None::<String>)
    .bind(None::<String>)
    .bind(None::<ArtifactId>)
    .bind(Vec::<u8>::new())
    .execute(&pool)
    .await;
    assert!(
        result.is_err(),
        "expected session_events epic->task composite FK violation"
    );
}
