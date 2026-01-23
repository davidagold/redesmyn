#![cfg(unix)]

use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_ids::{
    EpicId, RepoId, RequestId, SessionEventId, SessionId, SubscriptionId, TaskId, WorkspaceId,
};
use redesmyn_protocol::client::{
    AgentMessageConflictAction, ClientFrame, ClientMessage, CloseChatSessionRequest,
    CreateChatSessionRequest,
    GetEpicPinnedChatSessionRequest, GetSessionEventsRequest, ListChatSessionsRequest,
    ListTaskSessionsRequest, PinChatSessionToEpicRequest, Request, RequestPayload, ResponseResult,
    SendSessionMessageRequest, SessionEventsFilter, Subscribe, SubscriptionEvent, SubscriptionFilter,
};
use redesmyn_protocol::session::{AssistantMessage, SessionEventKind, SessionScope, UserMessage};
use redesmyn_protocol::{ProtocolEnvelope, RepoScope, SessionEvent, Timestamp};
use redesmyn_storage::schema::{
    AgentInterfaceMode as StorageAgentInterfaceMode, AgentKind as StorageAgentKind,
    AgentSessionScopeKind as StorageAgentSessionScopeKind,
    AgentSessionStatus as StorageAgentSessionStatus,
};
use redesmyn_storage::sessions::{AgentSessionRecord, insert_agent_session};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::codec::ProtobufCodec;
use redesmyn_transport::client::framed::FramedEndpoint;

fn ts(seconds: i64) -> Timestamp {
    let dt = time::OffsetDateTime::from_unix_timestamp(seconds).unwrap();
    Timestamp::from_offset_date_time(dt)
}

fn user_event(session_id: SessionId, id: SessionEventId, created_at: Timestamp) -> SessionEvent {
    SessionEvent {
        session_event_id: id,
        created_at,
        scope: SessionScope::Chat,
        session_id,
        turn_id: None,
        kind: SessionEventKind::UserMessage(UserMessage {
            text: "hello".to_string(),
            preview: "hello".to_string(),
            full_text_artifact: None,
        }),
    }
}

fn assistant_event(
    session_id: SessionId,
    id: SessionEventId,
    created_at: Timestamp,
) -> SessionEvent {
    SessionEvent {
        session_event_id: id,
        created_at,
        scope: SessionScope::Chat,
        session_id,
        turn_id: None,
        kind: SessionEventKind::AssistantMessage(AssistantMessage {
            text: "world".to_string(),
            preview: "world".to_string(),
            full_text_artifact: None,
        }),
    }
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

async fn insert_epic(pool: &redesmyn_storage::SqlitePool, repo_id: RepoId, epic_id: EpicId) {
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
}

async fn insert_task(pool: &redesmyn_storage::SqlitePool, epic_id: EpicId, task_id: TaskId) {
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
}

async fn connect_with_retry(path: &std::path::Path) -> tokio::net::UnixStream {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match tokio::net::UnixStream::connect(path).await {
                Ok(stream) => return stream,
                Err(_) => tokio::task::yield_now().await,
            }
        }
    })
    .await
    .expect("connect timeout")
}

async fn recv_frame(
    conn: &mut FramedEndpoint<ProtobufCodec, tokio::net::UnixStream>,
) -> ClientFrame {
    tokio::time::timeout(Duration::from_secs(2), conn.recv())
        .await
        .expect("recv timeout")
        .expect("recv frame")
}

#[tokio::test]
async fn uds_server_supports_session_query_surfaces() {
    redesmyn_logging::init();

    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let pool = control_plane.pool().clone();
    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();
    let task_id = TaskId::new();

    insert_workspace(&pool, workspace_id).await;
    insert_repo(&pool, workspace_id, repo_id).await;
    insert_epic(&pool, repo_id, epic_id).await;
    insert_task(&pool, epic_id, task_id).await;

    let task_session_id = SessionId::new();
    let task_session = AgentSessionRecord {
        session_id: task_session_id,
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
    };
    insert_agent_session(&pool, &task_session)
        .await
        .expect("insert task session");

    let server_socket_path = socket_path.clone();
    let server_control_plane = control_plane.clone();
    let server_shutdown_tx = shutdown_tx.clone();
    let server = tokio::spawn(async move {
        let mut shutdown = server_shutdown_tx.subscribe();
        redesmyn_control_plane::client_api::serve_client_api_uds(
            server_control_plane,
            server_socket_path,
            ClientApiCodec::Protobuf,
            &mut shutdown,
        )
        .await
    });

    let stream = connect_with_retry(&socket_path).await;
    let mut conn = FramedEndpoint::new(stream, ProtobufCodec::new());
    let scope = RepoScope::new(workspace_id, repo_id);

    let list_task_sessions_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: list_task_sessions_request_id,
            payload: RequestPayload::ListTaskSessions(ListTaskSessionsRequest {
                task_id,
                limit: 10,
            }),
        }),
    ))
    .await
    .expect("send list_task_sessions");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    match resp.result {
        ResponseResult::ListTaskSessions(payload) => {
            assert_eq!(resp.request_id, list_task_sessions_request_id);
            assert_eq!(payload.active_session_id, Some(task_session_id));
            assert_eq!(payload.sessions.len(), 1);
            assert_eq!(payload.sessions[0].session_id, task_session_id);
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let create_chat_session_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: create_chat_session_request_id,
            payload: RequestPayload::CreateChatSession(CreateChatSessionRequest {
                title: Some("Chat".to_string()),
            }),
        }),
    ))
    .await
    .expect("send create_chat_session");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    let chat_session_id = match resp.result {
        ResponseResult::CreateChatSession(payload) => {
            assert_eq!(resp.request_id, create_chat_session_request_id);
            payload.session_id
        }
        other => panic!("unexpected response: {other:?}"),
    };

    let list_chat_sessions_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: list_chat_sessions_request_id,
            payload: RequestPayload::ListChatSessions(ListChatSessionsRequest {
                include_closed: false,
                limit: 10,
            }),
        }),
    ))
    .await
    .expect("send list_chat_sessions");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    match resp.result {
        ResponseResult::ListChatSessions(payload) => {
            assert_eq!(resp.request_id, list_chat_sessions_request_id);
            assert_eq!(payload.sessions.len(), 1);
            assert_eq!(payload.sessions[0].session_id, chat_session_id);
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let pin_chat_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: pin_chat_request_id,
            payload: RequestPayload::PinChatSessionToEpic(PinChatSessionToEpicRequest {
                epic_id,
                session_id: chat_session_id,
            }),
        }),
    ))
    .await
    .expect("send pin_chat_session_to_epic");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    match resp.result {
        ResponseResult::PinChatSessionToEpic(_) => {
            assert_eq!(resp.request_id, pin_chat_request_id);
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let get_pinned_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: get_pinned_request_id,
            payload: RequestPayload::GetEpicPinnedChatSession(GetEpicPinnedChatSessionRequest {
                epic_id,
            }),
        }),
    ))
    .await
    .expect("send get_epic_pinned_chat_session");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    match resp.result {
        ResponseResult::GetEpicPinnedChatSession(payload) => {
            assert_eq!(resp.request_id, get_pinned_request_id);
            assert_eq!(payload.session_id, Some(chat_session_id));
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let send_session_message_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: send_session_message_request_id,
            payload: RequestPayload::SendSessionMessage(SendSessionMessageRequest {
                session_id: chat_session_id,
                message: "hello from client".to_string(),
                on_conflict: AgentMessageConflictAction::Fail,
            }),
        }),
    ))
    .await
    .expect("send send_session_message");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    let sent_event_id = match resp.result {
        ResponseResult::SendSessionMessage(payload) => {
            assert_eq!(resp.request_id, send_session_message_request_id);
            assert_eq!(payload.session_id, chat_session_id);
            assert_eq!(payload.event.session_id, chat_session_id);
            match payload.event.kind {
                SessionEventKind::UserMessage(message) => {
                    assert_eq!(message.text, "hello from client");
                    payload.event.session_event_id
                }
                other => panic!("unexpected SendSessionMessage event kind: {other:?}"),
            }
        }
        other => panic!("unexpected response: {other:?}"),
    };

    let close_chat_session_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: close_chat_session_request_id,
            payload: RequestPayload::CloseChatSession(CloseChatSessionRequest {
                session_id: chat_session_id,
            }),
        }),
    ))
    .await
    .expect("send close_chat_session");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    match resp.result {
        ResponseResult::CloseChatSession(_) => {
            assert_eq!(resp.request_id, close_chat_session_request_id);
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let list_open_chats_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: list_open_chats_request_id,
            payload: RequestPayload::ListChatSessions(ListChatSessionsRequest {
                include_closed: false,
                limit: 10,
            }),
        }),
    ))
    .await
    .expect("send list_chat_sessions open");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    match resp.result {
        ResponseResult::ListChatSessions(payload) => {
            assert_eq!(resp.request_id, list_open_chats_request_id);
            assert!(payload.sessions.is_empty());
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let list_all_chats_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: list_all_chats_request_id,
            payload: RequestPayload::ListChatSessions(ListChatSessionsRequest {
                include_closed: true,
                limit: 10,
            }),
        }),
    ))
    .await
    .expect("send list_chat_sessions all");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    match resp.result {
        ResponseResult::ListChatSessions(payload) => {
            assert_eq!(resp.request_id, list_all_chats_request_id);
            assert_eq!(payload.sessions.len(), 1);
            assert_eq!(payload.sessions[0].session_id, chat_session_id);
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let e1 = user_event(chat_session_id, SessionEventId::new(), ts(10));
    let e2 = assistant_event(chat_session_id, SessionEventId::new(), ts(20));
    control_plane
        .session_events()
        .append_session_event(&e1)
        .await
        .expect("append e1");
    control_plane
        .session_events()
        .append_session_event(&e2)
        .await
        .expect("append e2");

    let get_session_events_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Request(Request {
            request_id: get_session_events_request_id,
            payload: RequestPayload::GetSessionEvents(GetSessionEventsRequest {
                session_id: chat_session_id,
                before: None,
                limit: 10,
                kinds: vec![],
            }),
        }),
    ))
    .await
    .expect("send get_session_events");

    let frame = recv_frame(&mut conn).await;
    let ClientMessage::Response(resp) = frame.message else {
        panic!("expected Response, got {:?}", frame.message);
    };
    match resp.result {
        ResponseResult::GetSessionEvents(payload) => {
            assert_eq!(resp.request_id, get_session_events_request_id);
            assert_eq!(payload.next_cursor, None);
            assert_eq!(
                payload
                    .events
                    .iter()
                    .map(|ev| ev.session_event_id)
                    .collect::<Vec<_>>(),
                vec![e1.session_event_id, e2.session_event_id, sent_event_id]
            );
        }
        other => panic!("unexpected response: {other:?}"),
    }

    let subscription_id = SubscriptionId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope.into()),
        ClientMessage::Subscribe(Subscribe {
            subscription_id,
            filter: SubscriptionFilter::SessionEvents(SessionEventsFilter {
                session_id: chat_session_id,
                after: None,
            }),
        }),
    ))
    .await
    .expect("send session_events subscribe");

    loop {
        let frame = recv_frame(&mut conn).await;
        let ClientMessage::Event(event) = frame.message else {
            continue;
        };
        if event.subscription_id != subscription_id {
            continue;
        }
        match event.event {
            SubscriptionEvent::Subscribed(_) => break,
            SubscriptionEvent::EventLog(_) => continue,
            SubscriptionEvent::SessionEvent(_) => continue,
            SubscriptionEvent::Error(err) => panic!("unexpected subscription error: {err:?}"),
        }
    }

    let e3 = user_event(chat_session_id, SessionEventId::new(), ts(30));
    control_plane
        .session_events()
        .append_session_event(&e3)
        .await
        .expect("append e3");

    loop {
        let frame = recv_frame(&mut conn).await;
        let ClientMessage::Event(event) = frame.message else {
            continue;
        };
        if event.subscription_id != subscription_id {
            continue;
        }
        match event.event {
            SubscriptionEvent::SessionEvent(ev) => {
                assert_eq!(ev.session_event_id, e3.session_event_id);
                break;
            }
            SubscriptionEvent::Subscribed(_) => continue,
            SubscriptionEvent::EventLog(_) => continue,
            SubscriptionEvent::Error(err) => panic!("unexpected subscription error: {err:?}"),
        }
    }

    let _ = shutdown_tx.send(());
    server.await.expect("server task").expect("server exit");
}
