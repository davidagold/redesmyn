use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_control_plane::session_events::{
    SessionEventsResyncReason, SessionEventsSubscription, SessionEventsSubscriptionItem,
};
use redesmyn_ids::{EpicId, RepoId, SessionEventId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::client::{SessionEventCursor, SessionEventKindFilter};
use redesmyn_protocol::session::{AssistantMessage, SessionEventKind, SessionScope, UserMessage};
use redesmyn_protocol::{SessionEvent, Timestamp};
use redesmyn_storage::schema::{
    AgentKind as StorageAgentKind, AgentSessionScopeKind as StorageAgentSessionScopeKind,
    AgentSessionStatus as StorageAgentSessionStatus,
};
use redesmyn_storage::sessions::{AgentSessionRecord, insert_agent_session};

async fn insert_repo_and_chat_session(control_plane: &ControlPlane, session_id: SessionId) {
    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let now_ms = 0_i64;

    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?, ?, ?, ?)
        "#,
    )
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("test-workspace")
    .execute(control_plane.pool())
    .await
    .expect("insert workspace");

    sqlx::query(
        r#"
        INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(repo_id)
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("test-repo")
    .bind("Test Repo")
    .execute(control_plane.pool())
    .await
    .expect("insert repository");

    insert_agent_session(
        control_plane.pool(),
        &AgentSessionRecord {
            session_id,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: StorageAgentSessionScopeKind::Chat,
            task_id: None,
            epic_id: None,
            agent_kind: StorageAgentKind::Shell,
            status: StorageAgentSessionStatus::Stopped,
            external_session_ref: r#"{"type":"none"}"#.to_owned(),
            title: None,
            started_at_ms: None,
            ended_at_ms: None,
            archived_at_ms: None,
            repo_name: None,
        },
    )
    .await
    .expect("insert agent session");
}

async fn recv_item(sub: &mut SessionEventsSubscription) -> SessionEventsSubscriptionItem {
    tokio::time::timeout(Duration::from_secs(2), sub.recv())
        .await
        .expect("recv timeout")
        .expect("subscription closed")
}

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
            image_attachments: Vec::new(),
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

#[tokio::test]
async fn append_session_event_creates_missing_task_session_row() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();
    let task_id = TaskId::new();
    let now_ms = 0_i64;

    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?, ?, ?, ?)
        "#,
    )
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("test-workspace")
    .execute(control_plane.pool())
    .await
    .expect("insert workspace");

    sqlx::query(
        r#"
        INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(repo_id)
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("test-repo")
    .bind("Test Repo")
    .execute(control_plane.pool())
    .await
    .expect("insert repository");

    sqlx::query(
        r#"
        INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(epic_id)
    .bind(repo_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("test-epic")
    .bind("Test Epic")
    .execute(control_plane.pool())
    .await
    .expect("insert epic");

    sqlx::query(
        r#"
        INSERT INTO tasks (id, epic_id, created_at_ms, updated_at_ms, title, merge_readiness)
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(task_id)
    .bind(epic_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("Test Task")
    .bind("unknown")
    .execute(control_plane.pool())
    .await
    .expect("insert task");

    let session_id = SessionId::new();
    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Task { task_id },
        session_id,
        turn_id: None,
        kind: SessionEventKind::UserMessage(UserMessage {
            text: "hello".to_string(),
            preview: "hello".to_string(),
            full_text_artifact: None,
            image_attachments: Vec::new(),
        }),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .expect("append session event");

    let session = redesmyn_storage::sessions::get_agent_session(control_plane.pool(), session_id)
        .await
        .expect("get agent session")
        .expect("missing agent session row");
    assert_eq!(session.scope_kind, StorageAgentSessionScopeKind::Task);
    assert_eq!(session.task_id, Some(task_id));
    assert_eq!(session.scope_workspace_id, workspace_id);
    assert_eq!(session.scope_repo_id, repo_id);
}

#[tokio::test]
async fn session_events_pagination_returns_stable_next_cursor() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let session_id = SessionId::new();
    insert_repo_and_chat_session(&control_plane, session_id).await;
    let e1 = user_event(session_id, SessionEventId::new(), ts(1));
    let e2 = user_event(session_id, SessionEventId::new(), ts(2));
    let e3 = user_event(session_id, SessionEventId::new(), ts(3));
    let e4 = user_event(session_id, SessionEventId::new(), ts(4));
    let e5 = user_event(session_id, SessionEventId::new(), ts(5));

    for ev in [&e1, &e2, &e3, &e4, &e5] {
        control_plane
            .session_events()
            .append_session_event(ev)
            .await
            .expect("append session event");
    }

    let (page1, next1) = control_plane
        .session_events()
        .get_session_events(session_id, None, 2, &[])
        .await
        .expect("get session events");
    assert_eq!(
        page1.iter().map(|e| e.session_event_id).collect::<Vec<_>>(),
        vec![e4.session_event_id, e5.session_event_id]
    );
    let expected_next1 = SessionEventCursor {
        created_at: e4.created_at,
        session_event_id: e4.session_event_id,
    };
    assert_eq!(next1, Some(expected_next1));

    let (page2, next2) = control_plane
        .session_events()
        .get_session_events(session_id, next1, 2, &[])
        .await
        .expect("get session events page2");
    assert_eq!(
        page2.iter().map(|e| e.session_event_id).collect::<Vec<_>>(),
        vec![e2.session_event_id, e3.session_event_id]
    );
    let expected_next2 = SessionEventCursor {
        created_at: e2.created_at,
        session_event_id: e2.session_event_id,
    };
    assert_eq!(next2, Some(expected_next2));

    let (page3, next3) = control_plane
        .session_events()
        .get_session_events(session_id, next2, 2, &[])
        .await
        .expect("get session events page3");
    assert_eq!(
        page3.iter().map(|e| e.session_event_id).collect::<Vec<_>>(),
        vec![e1.session_event_id]
    );
    assert_eq!(next3, None);
}

#[tokio::test]
async fn session_events_pagination_supports_kind_filtering() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let session_id = SessionId::new();
    insert_repo_and_chat_session(&control_plane, session_id).await;
    let user = user_event(session_id, SessionEventId::new(), ts(1));
    let assistant = assistant_event(session_id, SessionEventId::new(), ts(2));

    control_plane
        .session_events()
        .append_session_event(&user)
        .await
        .expect("append user event");
    control_plane
        .session_events()
        .append_session_event(&assistant)
        .await
        .expect("append assistant event");

    let (events, next_cursor) = control_plane
        .session_events()
        .get_session_events(session_id, None, 10, &[SessionEventKindFilter::UserMessage])
        .await
        .expect("get session events");

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].session_event_id, user.session_event_id);
    assert_eq!(next_cursor, None);
}

#[tokio::test]
async fn session_events_subscription_requires_resync_when_cursor_missing() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let session_id = SessionId::new();
    let mut sub = control_plane.session_events().subscribe(
        session_id,
        Some(SessionEventCursor {
            created_at: ts(0),
            session_event_id: SessionEventId::new(),
        }),
    );

    match recv_item(&mut sub).await {
        SessionEventsSubscriptionItem::ResyncRequired(resync) => {
            assert_eq!(resync.reason, SessionEventsResyncReason::CursorNotFound);
            assert_eq!(resync.resume_after, None);
            assert_eq!(resync.dropped_events, None);
        }
        other => panic!("expected resync, got {other:?}"),
    }
}

#[tokio::test]
async fn session_events_subscription_resumes_from_cursor_and_delivers_appends() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let session_id = SessionId::new();
    insert_repo_and_chat_session(&control_plane, session_id).await;
    let first = user_event(session_id, SessionEventId::new(), ts(1));
    let second = user_event(session_id, SessionEventId::new(), ts(2));

    control_plane
        .session_events()
        .append_session_event(&first)
        .await
        .expect("append first");
    control_plane
        .session_events()
        .append_session_event(&second)
        .await
        .expect("append second");

    let after = SessionEventCursor {
        created_at: first.created_at,
        session_event_id: first.session_event_id,
    };

    let mut sub = control_plane
        .session_events()
        .subscribe(session_id, Some(after));

    match recv_item(&mut sub).await {
        SessionEventsSubscriptionItem::Event(ev) => {
            assert_eq!(ev.session_event_id, second.session_event_id)
        }
        other => panic!("expected second event, got {other:?}"),
    }

    let third = user_event(session_id, SessionEventId::new(), ts(3));
    control_plane
        .session_events()
        .append_session_event(&third)
        .await
        .expect("append third");

    match recv_item(&mut sub).await {
        SessionEventsSubscriptionItem::Event(ev) => {
            assert_eq!(ev.session_event_id, third.session_event_id)
        }
        other => panic!("expected third event, got {other:?}"),
    }
}
