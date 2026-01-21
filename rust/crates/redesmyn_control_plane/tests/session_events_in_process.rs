use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_control_plane::session_events::{
    SessionEventsResyncReason, SessionEventsSubscription, SessionEventsSubscriptionItem,
};
use redesmyn_ids::{SessionEventId, SessionId};
use redesmyn_protocol::client::{SessionEventCursor, SessionEventKindFilter};
use redesmyn_protocol::session::{AssistantMessage, SessionEventKind, SessionScope, UserMessage};
use redesmyn_protocol::{SessionEvent, Timestamp};

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
async fn session_events_pagination_returns_stable_next_cursor() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let session_id = SessionId::new();
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
