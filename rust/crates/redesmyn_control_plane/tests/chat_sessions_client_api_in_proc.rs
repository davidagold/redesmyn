use std::time::Duration;

use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_control_plane::{ControlPlane, ControlPlaneDb, ControlPlaneStartOptions};
use redesmyn_ids::{EpicId, RepoId, RequestId, WorkspaceId};
use redesmyn_protocol::ProtocolEnvelope;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, CloseChatSessionRequest, CreateChatSessionRequest,
    GetEpicPinnedChatSessionRequest, ListChatSessionsRequest, ListEpicsRequest,
    PinChatSessionToEpicRequest, Request, RequestPayload, ResponseResult,
    UnpinChatSessionFromEpicRequest,
};
use redesmyn_transport::client::ClientConnection;

async fn request(
    conn: &mut redesmyn_transport::client::in_proc::InProcEndpoint,
    payload: RequestPayload,
) -> ResponseResult {
    let request_id = RequestId::new();

    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request { request_id, payload }),
    ))
    .await
    .expect("send request");

    loop {
        let frame = tokio::time::timeout(Duration::from_secs(2), conn.recv())
            .await
            .expect("timeout waiting for response")
            .expect("recv response");

        let ClientMessage::Response(resp) = frame.message else {
            continue;
        };

        if resp.request_id == request_id {
            return resp.result;
        }
    }
}

#[tokio::test]
async fn chat_sessions_and_pins_work_over_in_proc_client_api() {
    let db = redesmyn_storage::open_test_sqlite_pool()
        .await
        .expect("open test DB");

    let now_ms: i64 = 1_700_000_000_000;
    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();

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

    let options = ControlPlaneStartOptions {
        db: ControlPlaneDb::Pool(db),
        client_api_socket_path: None,
        client_api_codec: ClientApiCodec::Protobuf,
    };
    let mut control_plane = ControlPlane::start(options).await.expect("start");

    let mut conn = control_plane.connect_in_proc_client(16);

    let list_epics = request(
        &mut conn,
        RequestPayload::ListEpics(ListEpicsRequest {}),
    )
    .await;
    let ResponseResult::ListEpics(list_epics) = list_epics else {
        panic!("expected ListEpics, got {list_epics:?}");
    };
    assert_eq!(list_epics.epics.len(), 1);
    assert_eq!(list_epics.epics[0].slug, "gpui");
    assert_eq!(list_epics.epics[0].epic_id, Some(epic_id));

    let create = request(
        &mut conn,
        RequestPayload::CreateChatSession(CreateChatSessionRequest {
            title: Some("First chat".to_string()),
        }),
    )
    .await;
    let ResponseResult::CreateChatSession(create) = create else {
        panic!("expected CreateChatSession, got {create:?}");
    };
    let first_session_id = create.session_id;

    let pin = request(
        &mut conn,
        RequestPayload::PinChatSessionToEpic(PinChatSessionToEpicRequest {
            session_id: first_session_id,
            epic_id,
        }),
    )
    .await;
    let ResponseResult::PinChatSessionToEpic(pin) = pin else {
        panic!("expected PinChatSessionToEpic, got {pin:?}");
    };
    assert_eq!(pin.replaced_session_id, None);

    let pinned = request(
        &mut conn,
        RequestPayload::GetEpicPinnedChatSession(GetEpicPinnedChatSessionRequest { epic_id }),
    )
    .await;
    let ResponseResult::GetEpicPinnedChatSession(pinned) = pinned else {
        panic!("expected GetEpicPinnedChatSession, got {pinned:?}");
    };
    assert_eq!(pinned.session_id, Some(first_session_id));

    let list_pinned = request(
        &mut conn,
        RequestPayload::ListChatSessions(ListChatSessionsRequest {
            pinned_to_epic_id: Some(epic_id),
            include_closed: true,
            limit: 10,
        }),
    )
    .await;
    let ResponseResult::ListChatSessions(list_pinned) = list_pinned else {
        panic!("expected ListChatSessions, got {list_pinned:?}");
    };
    assert_eq!(list_pinned.sessions.len(), 1);
    assert_eq!(list_pinned.sessions[0].session_id, first_session_id);
    assert!(list_pinned.sessions[0].closed_at.is_none());

    let create = request(
        &mut conn,
        RequestPayload::CreateChatSession(CreateChatSessionRequest { title: None }),
    )
    .await;
    let ResponseResult::CreateChatSession(create) = create else {
        panic!("expected CreateChatSession, got {create:?}");
    };
    let second_session_id = create.session_id;

    let pin = request(
        &mut conn,
        RequestPayload::PinChatSessionToEpic(PinChatSessionToEpicRequest {
            session_id: second_session_id,
            epic_id,
        }),
    )
    .await;
    let ResponseResult::PinChatSessionToEpic(pin) = pin else {
        panic!("expected PinChatSessionToEpic, got {pin:?}");
    };
    assert_eq!(pin.replaced_session_id, Some(first_session_id));

    let pinned = request(
        &mut conn,
        RequestPayload::GetEpicPinnedChatSession(GetEpicPinnedChatSessionRequest { epic_id }),
    )
    .await;
    let ResponseResult::GetEpicPinnedChatSession(pinned) = pinned else {
        panic!("expected GetEpicPinnedChatSession, got {pinned:?}");
    };
    assert_eq!(pinned.session_id, Some(second_session_id));

    let unpin = request(
        &mut conn,
        RequestPayload::UnpinChatSessionFromEpic(UnpinChatSessionFromEpicRequest { epic_id }),
    )
    .await;
    let ResponseResult::UnpinChatSessionFromEpic(unpin) = unpin else {
        panic!("expected UnpinChatSessionFromEpic, got {unpin:?}");
    };
    assert_eq!(unpin.unpinned_session_id, Some(second_session_id));

    let pinned = request(
        &mut conn,
        RequestPayload::GetEpicPinnedChatSession(GetEpicPinnedChatSessionRequest { epic_id }),
    )
    .await;
    let ResponseResult::GetEpicPinnedChatSession(pinned) = pinned else {
        panic!("expected GetEpicPinnedChatSession, got {pinned:?}");
    };
    assert_eq!(pinned.session_id, None);

    let close = request(
        &mut conn,
        RequestPayload::CloseChatSession(CloseChatSessionRequest {
            session_id: first_session_id,
        }),
    )
    .await;
    let ResponseResult::CloseChatSession(_) = close else {
        panic!("expected CloseChatSession, got {close:?}");
    };

    let list_open = request(
        &mut conn,
        RequestPayload::ListChatSessions(ListChatSessionsRequest {
            pinned_to_epic_id: None,
            include_closed: false,
            limit: 100,
        }),
    )
    .await;
    let ResponseResult::ListChatSessions(list_open) = list_open else {
        panic!("expected ListChatSessions, got {list_open:?}");
    };
    assert!(list_open
        .sessions
        .iter()
        .all(|session| session.closed_at.is_none()));

    control_plane.shutdown().await;
}

