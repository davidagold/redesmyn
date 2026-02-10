use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_control_plane::director_wake::{
    DirectorMode, DirectorWakeError, DirectorWakeNotification, DirectorWakeReason,
};
use redesmyn_control_plane::session_events::TASK_SESSION_TURN_COMPLETED_EVENT;
use redesmyn_ids::{EpicId, EventId, RepoId, SessionEventId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::session::{InterfaceMode, SessionEventKind, SessionScope, TurnCompleted};
use redesmyn_protocol::{SessionEvent, Timestamp};
use redesmyn_storage::events::EventScope;

fn repo_scope(workspace_id: WorkspaceId, repo_id: RepoId) -> EventScope {
    EventScope::Repo {
        workspace_id,
        repo_id,
    }
}

async fn recv_wake(
    rx: &mut tokio::sync::broadcast::Receiver<DirectorWakeNotification>,
) -> redesmyn_control_plane::director_wake::DirectorWakePayload {
    let notification = tokio::time::timeout(Duration::from_secs(3), rx.recv())
        .await
        .expect("wake notification timeout")
        .expect("wake notification channel closed");
    match notification {
        DirectorWakeNotification::WakeDispatched(payload) => payload,
    }
}

async fn assert_no_wake(rx: &mut tokio::sync::broadcast::Receiver<DirectorWakeNotification>) {
    let got = tokio::time::timeout(Duration::from_millis(250), rx.recv()).await;
    assert!(got.is_err(), "unexpected wake notification: {got:?}");
}

async fn seed_epic_task(control_plane: &ControlPlane) -> (WorkspaceId, RepoId, EpicId, TaskId) {
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

    (workspace_id, repo_id, epic_id, task_id)
}

#[tokio::test]
async fn task_turn_completed_triggers_wake() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (_workspace_id, _repo_id, epic_id, task_id) = seed_epic_task(&control_plane).await;
    let mut notifications = control_plane.director_wake().subscribe_notifications();

    control_plane
        .director_wake()
        .start_epic(epic_id)
        .await
        .expect("start epic watcher");
    control_plane
        .director_wake()
        .set_mode(epic_id, DirectorMode::Active)
        .await
        .expect("set active mode");

    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Task { task_id },
        session_id: SessionId::new(),
        turn_id: Some("turn-1".to_owned()),
        kind: SessionEventKind::TurnCompleted(TurnCompleted {
            interface_mode: InterfaceMode::Structured,
            external_session_ref: None,
            exit_code: Some(0),
            error: None,
        }),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .expect("append session turn completed");

    let wake = recv_wake(&mut notifications).await;
    assert_eq!(
        wake.summary.reasons,
        vec![DirectorWakeReason::TaskTurnCompleted]
    );
    assert_eq!(wake.summary.queue_size, 1);
    assert_eq!(wake.chunks.len(), 1);
    assert_eq!(
        wake.chunks[0].events[0].event_type,
        TASK_SESSION_TURN_COMPLETED_EVENT
    );

    control_plane.director_wake().stop_epic(epic_id).await;
}

#[tokio::test]
async fn wake_payload_includes_all_unacked_events_and_chunks() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, epic_id, _task_id) = seed_epic_task(&control_plane).await;
    let scope = repo_scope(workspace_id, repo_id);
    let mut notifications = control_plane.director_wake().subscribe_notifications();

    control_plane
        .director_wake()
        .start_epic(epic_id)
        .await
        .expect("start epic watcher");
    control_plane
        .director_wake()
        .set_mode(epic_id, DirectorMode::Active)
        .await
        .expect("set active mode");

    let mut expected_ids = Vec::new();
    for n in 0..129 {
        let id = control_plane
            .event_log()
            .append_event(
                scope,
                "epic.note",
                serde_json::to_vec(&serde_json::json!({
                    "epic_id": epic_id,
                    "index": n,
                }))
                .unwrap(),
            )
            .await
            .expect("append backlog note");
        expected_ids.push(id);
    }

    let last_id = control_plane
        .event_log()
        .append_event(
            scope,
            "conductor.override.approved",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
                "actor": "conductor",
                "decision": "approved",
            }))
            .unwrap(),
        )
        .await
        .expect("append trigger event");
    expected_ids.push(last_id);

    let wake = recv_wake(&mut notifications).await;
    assert!(wake.chunks.len() > 1, "expected chunking for large backlog");
    assert_eq!(wake.summary.queue_size, expected_ids.len());
    assert_eq!(
        wake.summary.reasons,
        vec![DirectorWakeReason::ConductorOverride]
    );
    assert_eq!(wake.summary.high_water_event_id, last_id);

    let actual_ids: Vec<EventId> = wake
        .chunks
        .iter()
        .flat_map(|chunk| chunk.events.iter().map(|event| event.event_id))
        .collect();
    assert_eq!(actual_ids, expected_ids);

    control_plane.director_wake().stop_epic(epic_id).await;
}

#[tokio::test]
async fn coalesces_while_in_flight_and_dispatches_after_ack() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, epic_id, _task_id) = seed_epic_task(&control_plane).await;
    let scope = repo_scope(workspace_id, repo_id);
    let mut notifications = control_plane.director_wake().subscribe_notifications();

    control_plane
        .director_wake()
        .start_epic(epic_id)
        .await
        .expect("start epic watcher");
    control_plane
        .director_wake()
        .set_mode(epic_id, DirectorMode::Active)
        .await
        .expect("set active mode");

    let first_high_water = control_plane
        .event_log()
        .append_event(
            scope,
            "command.succeeded",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
                "kind": "task.agent.start",
            }))
            .unwrap(),
        )
        .await
        .expect("append first trigger");
    let first_wake = recv_wake(&mut notifications).await;
    assert_eq!(first_wake.summary.high_water_event_id, first_high_water);

    let coalesced_non_significant = control_plane
        .event_log()
        .append_event(
            scope,
            "epic.note",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
                "note": "queued while in flight",
            }))
            .unwrap(),
        )
        .await
        .expect("append coalesced non-significant");
    let coalesced_trigger = control_plane
        .event_log()
        .append_event(
            scope,
            "conductor.override.defer",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
                "actor": "conductor",
            }))
            .unwrap(),
        )
        .await
        .expect("append coalesced trigger");

    assert_no_wake(&mut notifications).await;

    let ack_result = control_plane
        .director_wake()
        .ack_wake(
            epic_id,
            &first_wake.summary.wake_id,
            first_wake.summary.high_water_event_id,
        )
        .await
        .expect("ack first wake");
    assert!(ack_result.replay_payload.is_none());
    let next_wake = ack_result
        .next_payload
        .expect("expected next wake after ack");
    assert_eq!(next_wake.summary.high_water_event_id, coalesced_trigger);
    let next_ids: Vec<EventId> = next_wake
        .chunks
        .iter()
        .flat_map(|chunk| chunk.events.iter().map(|event| event.event_id))
        .collect();
    assert_eq!(next_ids, vec![coalesced_non_significant, coalesced_trigger]);

    control_plane.director_wake().stop_epic(epic_id).await;
}

#[tokio::test]
async fn invalid_ack_is_rejected_without_advancing_cursor() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, epic_id, _task_id) = seed_epic_task(&control_plane).await;
    let scope = repo_scope(workspace_id, repo_id);
    let mut notifications = control_plane.director_wake().subscribe_notifications();

    control_plane
        .director_wake()
        .start_epic(epic_id)
        .await
        .expect("start epic watcher");
    control_plane
        .director_wake()
        .set_mode(epic_id, DirectorMode::Active)
        .await
        .expect("set active mode");

    control_plane
        .event_log()
        .append_event(
            scope,
            "command.succeeded",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
            }))
            .unwrap(),
        )
        .await
        .expect("append initial trigger");
    let wake = recv_wake(&mut notifications).await;

    let out_of_range_ack = control_plane
        .event_log()
        .append_event(
            scope,
            "command.failed",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
            }))
            .unwrap(),
        )
        .await
        .expect("append newer event");

    let err = control_plane
        .director_wake()
        .ack_wake(epic_id, &wake.summary.wake_id, out_of_range_ack)
        .await
        .expect_err("ack should be rejected");
    assert!(matches!(err, DirectorWakeError::InvalidAck { .. }));

    let snapshot = control_plane
        .director_wake()
        .snapshot(epic_id)
        .await
        .expect("snapshot");
    assert_eq!(snapshot.ack_cursor_event_id, None);
    assert_eq!(
        snapshot.in_flight_wake_id,
        Some(wake.summary.wake_id.clone())
    );

    let rejected_count: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM events
        WHERE kind = 'director.wake.ack_rejected'
        "#,
    )
    .fetch_one(control_plane.pool())
    .await
    .expect("count rejected events");
    assert_eq!(rejected_count, 1);

    control_plane.director_wake().stop_epic(epic_id).await;
}

#[tokio::test]
async fn ack_not_in_payload_relevant_set_is_rejected_without_cursor_advance() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, epic_id, _task_id) = seed_epic_task(&control_plane).await;
    let scope = repo_scope(workspace_id, repo_id);
    let mut notifications = control_plane.director_wake().subscribe_notifications();

    control_plane
        .director_wake()
        .start_epic(epic_id)
        .await
        .expect("start epic watcher");
    control_plane
        .director_wake()
        .set_mode(epic_id, DirectorMode::Active)
        .await
        .expect("set active mode");

    let unrelated_epic_id = EpicId::new();
    let unrelated_event_id = control_plane
        .event_log()
        .append_event(
            scope,
            "epic.note",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": unrelated_epic_id,
                "note": "other epic event",
            }))
            .unwrap(),
        )
        .await
        .expect("append unrelated epic event");

    let trigger_event_id = control_plane
        .event_log()
        .append_event(
            scope,
            "command.succeeded",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
            }))
            .unwrap(),
        )
        .await
        .expect("append wake trigger event");
    let wake = recv_wake(&mut notifications).await;
    assert_eq!(wake.summary.high_water_event_id, trigger_event_id);

    // Rationale: rowid alone is insufficient; ack must reference an event present in
    // this wake's relevant payload window for the target epic.
    let payload_ids: Vec<EventId> = wake
        .chunks
        .iter()
        .flat_map(|chunk| chunk.events.iter().map(|event| event.event_id))
        .collect();
    assert_eq!(payload_ids, vec![trigger_event_id]);

    let err = control_plane
        .director_wake()
        .ack_wake(epic_id, &wake.summary.wake_id, unrelated_event_id)
        .await
        .expect_err("ack of unrelated event should be rejected");
    assert!(matches!(err, DirectorWakeError::InvalidAck { .. }));

    let snapshot = control_plane
        .director_wake()
        .snapshot(epic_id)
        .await
        .expect("snapshot");
    assert_eq!(snapshot.ack_cursor_event_id, None);
    assert_eq!(
        snapshot.in_flight_wake_id,
        Some(wake.summary.wake_id.clone())
    );

    let rejected_count: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM events
        WHERE kind = 'director.wake.ack_rejected'
        "#,
    )
    .fetch_one(control_plane.pool())
    .await
    .expect("count rejected events");
    assert_eq!(rejected_count, 1);

    control_plane.director_wake().stop_epic(epic_id).await;
}

#[tokio::test]
async fn wake_id_mismatch_does_not_mutate_in_flight_state() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, epic_id, _task_id) = seed_epic_task(&control_plane).await;
    let scope = repo_scope(workspace_id, repo_id);
    let mut notifications = control_plane.director_wake().subscribe_notifications();

    control_plane
        .director_wake()
        .start_epic(epic_id)
        .await
        .expect("start epic watcher");
    control_plane
        .director_wake()
        .set_mode(epic_id, DirectorMode::Active)
        .await
        .expect("set active mode");

    control_plane
        .event_log()
        .append_event(
            scope,
            "command.succeeded",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
            }))
            .unwrap(),
        )
        .await
        .expect("append initial trigger");
    let wake = recv_wake(&mut notifications).await;

    // Rationale: wake identity is part of deterministic replay; mismatches must not
    // alter cursor or in-flight wake window.
    let err = control_plane
        .director_wake()
        .ack_wake(
            epic_id,
            "wake-id-mismatch",
            wake.summary.high_water_event_id,
        )
        .await
        .expect_err("wake id mismatch should fail");
    assert!(matches!(err, DirectorWakeError::WakeIdMismatch { .. }));

    let snapshot = control_plane
        .director_wake()
        .snapshot(epic_id)
        .await
        .expect("snapshot");
    assert_eq!(snapshot.ack_cursor_event_id, None);
    assert_eq!(
        snapshot.in_flight_wake_id,
        Some(wake.summary.wake_id.clone())
    );
    assert_eq!(
        snapshot.in_flight_high_water_event_id,
        Some(wake.summary.high_water_event_id)
    );

    control_plane.director_wake().stop_epic(epic_id).await;
}

#[tokio::test]
async fn resume_required_blocks_dispatch_until_resume() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, epic_id, _task_id) = seed_epic_task(&control_plane).await;
    let scope = repo_scope(workspace_id, repo_id);
    let mut notifications = control_plane.director_wake().subscribe_notifications();

    control_plane
        .director_wake()
        .start_epic(epic_id)
        .await
        .expect("start epic watcher");
    control_plane
        .director_wake()
        .set_mode(epic_id, DirectorMode::Active)
        .await
        .expect("set active mode");

    let first_id = control_plane
        .event_log()
        .append_event(
            scope,
            "command.succeeded",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
            }))
            .unwrap(),
        )
        .await
        .expect("append initial trigger");
    let first_wake = recv_wake(&mut notifications).await;
    assert_eq!(first_wake.summary.high_water_event_id, first_id);

    control_plane
        .director_wake()
        .mark_interrupted(epic_id)
        .await
        .expect("mark interrupted");

    let resumed_high_water = control_plane
        .event_log()
        .append_event(
            scope,
            "conductor.override.approved",
            serde_json::to_vec(&serde_json::json!({
                "epic_id": epic_id,
                "actor": "conductor",
            }))
            .unwrap(),
        )
        .await
        .expect("append event while resume_required");

    assert_no_wake(&mut notifications).await;

    let resumed_wake = control_plane
        .director_wake()
        .resume(epic_id)
        .await
        .expect("resume")
        .expect("wake after resume");
    assert_eq!(resumed_wake.summary.high_water_event_id, resumed_high_water);

    let snapshot = control_plane
        .director_wake()
        .snapshot(epic_id)
        .await
        .expect("snapshot");
    assert_eq!(snapshot.director_mode, DirectorMode::Active);

    control_plane.director_wake().stop_epic(epic_id).await;
}
