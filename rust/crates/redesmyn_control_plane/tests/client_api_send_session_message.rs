use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_ids::{RepoId, RequestId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::ProtocolEnvelope;
use redesmyn_protocol::client::{
    AgentMessageConflictAction, ClientFrame, ClientMessage, Request, RequestPayload,
    ResponseResult, SendSessionMessageRequest,
};
use redesmyn_protocol::session::SessionEventKind;
use redesmyn_storage::schema::{
    AgentInterfaceMode as StorageAgentInterfaceMode, AgentKind as StorageAgentKind,
    AgentSessionScopeKind as StorageAgentSessionScopeKind,
    AgentSessionStatus as StorageAgentSessionStatus,
};
use redesmyn_storage::sessions::{AgentSessionRecord, get_agent_session, insert_agent_session};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::in_proc::InProcEndpoint;

async fn request(conn: &mut InProcEndpoint, payload: RequestPayload) -> ResponseResult {
    let request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request {
            request_id,
            payload,
        }),
    ))
    .await
    .expect("send request");

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let frame = conn.recv().await.expect("recv frame");
            match frame.message {
                ClientMessage::Response(resp) if resp.request_id == request_id => {
                    return resp.result;
                }
                ClientMessage::Response(_) => continue,
                ClientMessage::Event(_)
                | ClientMessage::Subscribe(_)
                | ClientMessage::Unsubscribe(_)
                | ClientMessage::Request(_) => continue,
            }
        }
    })
    .await
    .expect("response timeout")
}

async fn insert_workspace(pool: &redesmyn_storage::SqlitePool, workspace_id: WorkspaceId) {
    let now_ms = 1_i64;
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
    .unwrap();
}

async fn insert_repo(
    pool: &redesmyn_storage::SqlitePool,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
) {
    let now_ms = 2_i64;
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
    .unwrap();
}

async fn insert_epic(pool: &redesmyn_storage::SqlitePool, repo_id: RepoId) -> redesmyn_ids::EpicId {
    let epic_id = redesmyn_ids::EpicId::new();
    let now_ms = 3_i64;
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
    .unwrap();
    epic_id
}

async fn insert_task(pool: &redesmyn_storage::SqlitePool, epic_id: redesmyn_ids::EpicId) -> TaskId {
    let task_id = TaskId::new();
    let now_ms = 4_i64;
    sqlx::query(
        r#"
        INSERT INTO tasks (id, epic_id, created_at_ms, updated_at_ms, title, merge_readiness)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
    )
    .bind(task_id)
    .bind(epic_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("Task")
    .bind("unknown")
    .execute(pool)
    .await
    .unwrap();
    task_id
}

#[tokio::test]
async fn send_session_message_appends_user_message_for_chat_session() {
    redesmyn_logging::init();

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let pool = control_plane.pool().clone();

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    insert_workspace(&pool, workspace_id).await;
    insert_repo(&pool, workspace_id, repo_id).await;

    let session_id = SessionId::new();
    insert_agent_session(
        &pool,
        &AgentSessionRecord {
            session_id,
            created_at_ms: 5,
            updated_at_ms: 5,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: StorageAgentSessionScopeKind::Chat,
            task_id: None,
            agent_kind: StorageAgentKind::Codex,
            interface_mode: StorageAgentInterfaceMode::StructuredExec,
            status: StorageAgentSessionStatus::Stopped,
            external_session_ref: r#"{"type":"none"}"#.to_owned(),
            title: None,
            started_at_ms: None,
            ended_at_ms: None,
            closed_at_ms: None,
        },
    )
    .await
    .expect("insert session");

    let (mut client, mut server) = InProcEndpoint::pair(8);
    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
    let mut shutdown = shutdown_tx.subscribe();
    let server_control_plane = control_plane.clone();
    let server_task = tokio::spawn(async move {
        redesmyn_control_plane::client_api::serve_connection(
            &mut server,
            server_control_plane,
            &mut shutdown,
        )
        .await
        .ok();
    });

    let result = request(
        &mut client,
        RequestPayload::SendSessionMessage(SendSessionMessageRequest {
            session_id,
            message: "hello".to_string(),
            on_conflict: AgentMessageConflictAction::Fail,
        }),
    )
    .await;

    let resp = match result {
        ResponseResult::SendSessionMessage(resp) => resp,
        other => panic!("unexpected response: {other:?}"),
    };
    assert_eq!(resp.session_id, session_id);
    assert_eq!(resp.event.session_id, session_id);
    assert!(matches!(resp.event.kind, SessionEventKind::UserMessage(_)));

    let events = control_plane
        .session_events()
        .get_session_events(session_id, None, 10, &[])
        .await
        .expect("get events")
        .0;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0].kind, SessionEventKind::UserMessage(_)));

    let _ = shutdown_tx.send(());
    let _ = server_task.await;
}

#[tokio::test]
async fn send_session_message_conflicts_on_structured_turn_in_progress() {
    redesmyn_logging::init();

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let pool = control_plane.pool().clone();

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    insert_workspace(&pool, workspace_id).await;
    insert_repo(&pool, workspace_id, repo_id).await;
    let epic_id = insert_epic(&pool, repo_id).await;
    let task_id = insert_task(&pool, epic_id).await;

    let session_id = SessionId::new();
    insert_agent_session(
        &pool,
        &AgentSessionRecord {
            session_id,
            created_at_ms: 5,
            updated_at_ms: 5,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: StorageAgentSessionScopeKind::Task,
            task_id: Some(task_id),
            agent_kind: StorageAgentKind::Codex,
            interface_mode: StorageAgentInterfaceMode::StructuredExec,
            status: StorageAgentSessionStatus::Running,
            external_session_ref: r#"{"type":"none"}"#.to_owned(),
            title: None,
            started_at_ms: None,
            ended_at_ms: None,
            closed_at_ms: None,
        },
    )
    .await
    .expect("insert session");

    let (mut client, mut server) = InProcEndpoint::pair(8);
    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
    let mut shutdown = shutdown_tx.subscribe();
    let server_control_plane = control_plane.clone();
    let server_task = tokio::spawn(async move {
        redesmyn_control_plane::client_api::serve_connection(
            &mut server,
            server_control_plane,
            &mut shutdown,
        )
        .await
        .ok();
    });

    let result = request(
        &mut client,
        RequestPayload::SendSessionMessage(SendSessionMessageRequest {
            session_id,
            message: "hello".to_string(),
            on_conflict: AgentMessageConflictAction::Fail,
        }),
    )
    .await;

    let ResponseResult::Error(err) = result else {
        panic!("expected conflict error, got {result:?}");
    };

    assert_eq!(err.category, redesmyn_protocol::ErrorCategory::Conflict);
    let code = err
        .detail
        .as_ref()
        .and_then(|detail| detail.get("conflict_code"))
        .map(String::as_str);
    assert_eq!(code, Some("structured_turn_in_progress"));

    let result = request(
        &mut client,
        RequestPayload::SendSessionMessage(SendSessionMessageRequest {
            session_id,
            message: "hello".to_string(),
            on_conflict: AgentMessageConflictAction::InterruptTurn,
        }),
    )
    .await;

    match result {
        ResponseResult::SendSessionMessage(resp) => {
            assert_eq!(resp.session_id, session_id);
            assert_eq!(resp.event.session_id, session_id);
            assert!(matches!(resp.event.kind, SessionEventKind::UserMessage(_)));
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let session = get_agent_session(&pool, session_id)
        .await
        .expect("get session")
        .expect("session exists");
    assert_eq!(session.status, StorageAgentSessionStatus::Stopped);

    let _ = shutdown_tx.send(());
    let _ = server_task.await;
}

#[tokio::test]
async fn send_session_message_stop_and_start_new_returns_new_session_id() {
    redesmyn_logging::init();

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let pool = control_plane.pool().clone();

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    insert_workspace(&pool, workspace_id).await;
    insert_repo(&pool, workspace_id, repo_id).await;
    let epic_id = insert_epic(&pool, repo_id).await;
    let task_id = insert_task(&pool, epic_id).await;

    let target_session_id = SessionId::new();
    insert_agent_session(
        &pool,
        &AgentSessionRecord {
            session_id: target_session_id,
            created_at_ms: 5,
            updated_at_ms: 5,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: StorageAgentSessionScopeKind::Task,
            task_id: Some(task_id),
            agent_kind: StorageAgentKind::Codex,
            interface_mode: StorageAgentInterfaceMode::StructuredExec,
            status: StorageAgentSessionStatus::Stopped,
            external_session_ref: r#"{"type":"none"}"#.to_owned(),
            title: None,
            started_at_ms: None,
            ended_at_ms: None,
            closed_at_ms: None,
        },
    )
    .await
    .expect("insert session");

    let conflicting_session_id = SessionId::new();
    insert_agent_session(
        &pool,
        &AgentSessionRecord {
            session_id: conflicting_session_id,
            created_at_ms: 6,
            updated_at_ms: 6,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: StorageAgentSessionScopeKind::Task,
            task_id: Some(task_id),
            agent_kind: StorageAgentKind::Shell,
            interface_mode: StorageAgentInterfaceMode::ShellTmux,
            status: StorageAgentSessionStatus::Running,
            external_session_ref: r#"{"type":"none"}"#.to_owned(),
            title: None,
            started_at_ms: None,
            ended_at_ms: None,
            closed_at_ms: None,
        },
    )
    .await
    .expect("insert session");

    let (mut client, mut server) = InProcEndpoint::pair(8);
    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
    let mut shutdown = shutdown_tx.subscribe();
    let server_control_plane = control_plane.clone();
    let server_task = tokio::spawn(async move {
        redesmyn_control_plane::client_api::serve_connection(
            &mut server,
            server_control_plane,
            &mut shutdown,
        )
        .await
        .ok();
    });

    let result = request(
        &mut client,
        RequestPayload::SendSessionMessage(SendSessionMessageRequest {
            session_id: target_session_id,
            message: "hello".to_string(),
            on_conflict: AgentMessageConflictAction::Fail,
        }),
    )
    .await;

    let ResponseResult::Error(err) = result else {
        panic!("expected conflict error, got {result:?}");
    };
    let code = err
        .detail
        .as_ref()
        .and_then(|detail| detail.get("conflict_code"))
        .map(String::as_str);
    assert_eq!(code, Some("structured_session_conflict"));

    let result = request(
        &mut client,
        RequestPayload::SendSessionMessage(SendSessionMessageRequest {
            session_id: target_session_id,
            message: "hello".to_string(),
            on_conflict: AgentMessageConflictAction::StopSessionAndStartNew,
        }),
    )
    .await;

    let (new_session_id, returned_event) = match result {
        ResponseResult::SendSessionMessage(resp) => (resp.session_id, resp.event),
        other => panic!("unexpected response: {other:?}"),
    };
    assert_ne!(new_session_id, target_session_id);
    assert_eq!(returned_event.session_id, new_session_id);
    assert!(matches!(
        returned_event.kind,
        SessionEventKind::UserMessage(_)
    ));

    let ended_target = get_agent_session(&pool, target_session_id)
        .await
        .expect("get session")
        .expect("session exists");
    assert!(ended_target.ended_at_ms.is_some());

    let ended_conflict = get_agent_session(&pool, conflicting_session_id)
        .await
        .expect("get session")
        .expect("session exists");
    assert!(ended_conflict.ended_at_ms.is_some());

    let new_session = get_agent_session(&pool, new_session_id)
        .await
        .expect("get session")
        .expect("session exists");
    assert!(new_session.ended_at_ms.is_none());

    let events = control_plane
        .session_events()
        .get_session_events(new_session_id, None, 10, &[])
        .await
        .expect("get events")
        .0;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0].kind, SessionEventKind::UserMessage(_)));

    let _ = shutdown_tx.send(());
    let _ = server_task.await;
}
