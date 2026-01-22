use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_control_plane::event_log::{EventLogConfig, EventLogSubscriptionItem};
use redesmyn_storage::events::EventScope;

async fn recv_item(
    sub: &mut redesmyn_control_plane::event_log::EventLogSubscription,
) -> EventLogSubscriptionItem {
    tokio::time::timeout(Duration::from_secs(2), sub.recv())
        .await
        .expect("recv timeout")
        .expect("subscription closed")
}

#[tokio::test]
async fn event_log_subscription_delivers_new_events_in_process() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let mut sub = control_plane.event_log().subscribe(EventScope::None, None);

    let payload = br#"{"hello":"world"}"#.to_vec();
    let expected_id = control_plane
        .event_log()
        .append_event(EventScope::None, "test.event", payload.clone())
        .await
        .expect("append event");

    match recv_item(&mut sub).await {
        EventLogSubscriptionItem::Event(ev) => {
            assert_eq!(ev.id, expected_id);
            assert_eq!(ev.kind, "test.event");
            assert_eq!(ev.payload, payload);
        }
        EventLogSubscriptionItem::ResyncRequired(resync) => {
            panic!("unexpected resync: {resync:?}");
        }
    }
}

#[tokio::test]
async fn event_log_subscription_resumes_from_cursor_in_process() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let after_event_id = control_plane
        .event_log()
        .append_event(EventScope::None, "test.first", br#"{"n":1}"#.to_vec())
        .await
        .expect("append first event");
    let payload = br#"{"n":2}"#.to_vec();
    let expected_id = control_plane
        .event_log()
        .append_event(EventScope::None, "test.second", payload.clone())
        .await
        .expect("append second event");

    let mut sub = control_plane
        .event_log()
        .subscribe(EventScope::None, Some(after_event_id));

    match recv_item(&mut sub).await {
        EventLogSubscriptionItem::Event(ev) => {
            assert_eq!(ev.id, expected_id);
            assert_eq!(ev.kind, "test.second");
            assert_eq!(ev.payload, payload);
        }
        EventLogSubscriptionItem::ResyncRequired(resync) => {
            panic!("unexpected resync: {resync:?}");
        }
    }
}

#[tokio::test]
async fn event_log_subscription_emits_resync_when_lagging() {
    let pool = redesmyn_storage::open_test_sqlite_pool()
        .await
        .expect("open test db");
    let control_plane = ControlPlane::new_with_event_log_config(
        pool,
        EventLogConfig {
            hub_buffer: 1,
            subscription_buffer: 1,
            backlog_page_size: 1,
        },
    );

    let mut sub = control_plane.event_log().subscribe(EventScope::None, None);

    for n in 0..10 {
        let _ = control_plane
            .event_log()
            .append_event(
                EventScope::None,
                format!("test.event.{n}"),
                br#"{}"#.to_vec(),
            )
            .await
            .expect("append event");
    }

    loop {
        match recv_item(&mut sub).await {
            EventLogSubscriptionItem::Event(_) => continue,
            EventLogSubscriptionItem::ResyncRequired(resync) => {
                assert_eq!(
                    resync.reason,
                    redesmyn_control_plane::event_log::EventLogResyncReason::Lagged
                );
                assert!(
                    resync.dropped_events.unwrap_or_default() > 0,
                    "expected dropped_events > 0"
                );
                break;
            }
        }
    }
}

