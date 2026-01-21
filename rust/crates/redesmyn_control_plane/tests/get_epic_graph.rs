#![cfg(unix)]

use std::time::Duration;

use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_control_plane::ControlPlane;
use redesmyn_ids::{
    CommandId, CommandUpdateId, EpicId, EventId, HostId, HostInstanceId, RepoId, RequestId,
    SessionEventId, SessionId, TaskId, WorkspaceId,
};
use redesmyn_protocol::ProtocolEnvelope;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, CommandState, GetEpicGraphRequest, Request, RequestPayload,
    ResponseResult, TaskState,
};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::codec::ProtobufCodec;
use redesmyn_transport::client::framed::FramedEndpoint;

#[tokio::test]
async fn get_epic_graph_returns_typed_projection() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let db = redesmyn_storage::open_test_sqlite_pool()
        .await
        .expect("open test DB");

    let now_ms: i64 = 1_700_000_000_000;
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
    .bind("dev")
    .execute(&db)
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
    .bind("demo")
    .bind("Demo Repo")
    .execute(&db)
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
    .bind("gpui")
    .bind("GPUI + Rust Port")
    .execute(&db)
    .await
    .expect("insert epic");

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
    .bind(task_id)
    .bind(epic_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("T-1")
    .bind("Task one")
    .execute(&db)
    .await
    .expect("insert task");

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
        VALUES (?1, ?2, ?3, 'repo', ?4, ?5, ?6, ?7, 'running', X'')
        "#,
    )
    .bind(command_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind(workspace_id)
    .bind(repo_id)
    .bind(task_id)
    .bind("task.agent.start")
    .execute(&db)
    .await
    .expect("insert command");

    let command_update_id = CommandUpdateId::new();
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
        VALUES (?1, ?2, ?3, 'running', ?4, 0, 10, NULL)
        "#,
    )
    .bind(command_update_id)
    .bind(command_id)
    .bind(now_ms + 1)
    .bind(Some("starting"))
    .execute(&db)
    .await
    .expect("insert command update");

    let host_id = HostId::new();
    let host_instance_id = HostInstanceId::new();
    sqlx::query("INSERT INTO hosts (id, created_at_ms, hostname) VALUES (?1, ?2, 'devbox')")
        .bind(host_id)
        .bind(now_ms)
        .execute(&db)
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
    .bind(now_ms + 2)
    .execute(&db)
    .await
    .expect("insert daemon presence");

    let (_session_id, session_event_id) =
        seed_session_event(&db, now_ms + 3, workspace_id, repo_id, epic_id, task_id).await;

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
        VALUES (?1, ?2, 'repo', ?3, ?4, 'event_log.appended', X'')
        "#,
    )
    .bind(event_id)
    .bind(now_ms + 4)
    .bind(workspace_id)
    .bind(repo_id)
    .execute(&db)
    .await
    .expect("insert event");

    let server_socket_path = socket_path.clone();
    let server_control_plane = ControlPlane::new(db.clone());
    let server = tokio::spawn(async move {
        redesmyn_control_plane::client_api::serve_client_api_uds(
            server_control_plane,
            server_socket_path,
            ClientApiCodec::Protobuf,
        )
        .await
    });

    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(socket_path.exists(), "socket file was not created");

    let stream = tokio::net::UnixStream::connect(&socket_path)
        .await
        .expect("connect");
    let mut conn = FramedEndpoint::new(stream, ProtobufCodec::new());

    let request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request {
            request_id,
            payload: RequestPayload::GetEpicGraph(GetEpicGraphRequest {
                epic_slug: "gpui".to_string(),
            }),
        }),
    ))
    .await
    .expect("send request");

    let frame = conn.recv().await.expect("recv");
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    assert_eq!(resp.request_id, request_id);

    let ResponseResult::GetEpicGraph(payload) = resp.result else {
        panic!("expected GetEpicGraph response, got {:?}", resp.result);
    };

    assert_eq!(payload.graph.epic_slug, "gpui");
    assert_eq!(payload.graph.epic_id, Some(epic_id));
    assert_eq!(
        payload.graph.epic_title.as_deref(),
        Some("GPUI + Rust Port")
    );
    assert_eq!(payload.graph.workspace_id, Some(workspace_id));
    assert_eq!(payload.graph.repo_id, Some(repo_id));
    assert_eq!(payload.graph.as_of_event_id, Some(event_id));

    assert_eq!(payload.graph.nodes.len(), 1);
    assert_eq!(payload.graph.nodes[0].task_id, Some(task_id));
    assert_eq!(payload.graph.nodes[0].state, TaskState::Todo);

    assert_eq!(payload.graph.command_summaries.len(), 1);
    assert_eq!(payload.graph.command_summaries[0].command_id, command_id);
    assert_eq!(payload.graph.command_summaries[0].state, CommandState::Running);
    assert!(
        payload.graph.command_summaries[0].last_update.is_some(),
        "expected last_update"
    );

    assert_eq!(payload.graph.daemon_presences.len(), 1);
    assert_eq!(
        payload.graph.daemon_presences[0].host_instance_id,
        host_instance_id
    );

    assert_eq!(payload.graph.session_summaries.len(), 1);
    assert_eq!(
        payload.graph.session_summaries[0].session_event_id,
        session_event_id
    );

    server.abort();
    let _ = server.await;
}

async fn seed_session_event(
    db: &sqlx::SqlitePool,
    created_at_ms: i64,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    epic_id: EpicId,
    task_id: TaskId,
) -> (SessionId, SessionEventId) {
    let session_id = SessionId::new();
    let session_event_id = SessionEventId::new();

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
        VALUES (?1, ?2, ?3, 'task', ?4, ?5, ?6, ?7, 'turn.started', 't1', 'preview', NULL, X'')
        "#,
    )
    .bind(session_event_id)
    .bind(session_id)
    .bind(created_at_ms)
    .bind(workspace_id)
    .bind(repo_id)
    .bind(epic_id)
    .bind(task_id)
    .execute(db)
    .await
    .expect("insert session event");

    (session_id, session_event_id)
}
