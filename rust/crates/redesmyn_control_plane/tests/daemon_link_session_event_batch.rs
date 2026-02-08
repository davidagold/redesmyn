use std::time::Duration;

use redesmyn_control_plane::{ControlPlane, DaemonLinkHandle};
use redesmyn_ids::{HostId, HostInstanceId, RepoId, SessionEventId, WorkspaceId};
use redesmyn_protocol::daemon::{
    DaemonFrame, DaemonHello, DaemonMessage, RepoAttach, SessionEventBatch,
};
use redesmyn_protocol::session::{SessionEventKind, SessionScope, UserMessage};
use redesmyn_protocol::{
    ProtocolEnvelope, ProtocolVersion, RepoScope, Scope, SessionEvent, Timestamp,
};
use redesmyn_storage::schema::AgentKind as StorageAgentKind;
use redesmyn_transport::in_proc::InProcEndpoint;

#[tokio::test]
async fn daemon_link_persists_session_event_batches() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let repo_scope = RepoScope::new(workspace_id, repo_id);
    let now_ms = 0_i64;

    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, ?4)
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
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
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

    let session_id = redesmyn_storage::sessions::create_chat_session(
        control_plane.pool(),
        workspace_id,
        repo_id,
        None,
        StorageAgentKind::Codex,
        None,
    )
    .await
    .expect("create chat session");

    let (control_plane_conn, mut daemon_conn) = InProcEndpoint::pair(64);
    let daemon_link = DaemonLinkHandle::start(
        &tokio::runtime::Handle::current(),
        control_plane.clone(),
        control_plane_conn,
    );

    let host_id = HostId::new();
    let host_instance_id = HostInstanceId::new();
    daemon_conn
        .send_frame(DaemonFrame::new(
            ProtocolEnvelope::new(),
            DaemonMessage::DaemonHello(DaemonHello {
                host_id,
                host_instance_id,
                capabilities: vec!["test".to_string()],
                supported_protocol: ProtocolVersion::CURRENT,
            }),
        ))
        .await
        .expect("send daemon hello");

    let ack = daemon_conn.recv_frame().await.expect("recv hello ack");
    let DaemonMessage::ControlPlaneHelloAck(ack) = ack.message else {
        panic!("expected ControlPlaneHelloAck, got {:?}", ack.message);
    };
    let accepted = ack.accepted_protocol;

    let mut attach_envelope = ProtocolEnvelope::new().with_scope(Scope::from(repo_scope));
    attach_envelope.protocol_major = accepted.major;
    attach_envelope.protocol_minor = accepted.minor;
    daemon_conn
        .send_frame(DaemonFrame::new(
            attach_envelope,
            DaemonMessage::RepoAttach(RepoAttach {
                repo_scope,
                repo_root_hint: None,
            }),
        ))
        .await
        .expect("send repo attach");

    let mut sub = control_plane.session_events().subscribe(session_id, None);

    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Chat,
        session_id,
        turn_id: None,
        kind: SessionEventKind::UserMessage(UserMessage {
            text: "hello".to_string(),
            preview: "hello".to_string(),
            full_text_artifact: None,
            image_attachments: Vec::new(),
        }),
    };

    let mut batch_envelope = ProtocolEnvelope::new().with_scope(Scope::from(repo_scope));
    batch_envelope.protocol_major = accepted.major;
    batch_envelope.protocol_minor = accepted.minor;
    daemon_conn
        .send_frame(DaemonFrame::new(
            batch_envelope,
            DaemonMessage::SessionEventBatch(SessionEventBatch {
                events: vec![event.clone()],
            }),
        ))
        .await
        .expect("send session event batch");

    let recv = tokio::time::timeout(Duration::from_secs(2), sub.recv())
        .await
        .expect("subscription timeout")
        .expect("subscription closed");

    match recv {
        redesmyn_control_plane::session_events::SessionEventsSubscriptionItem::Event(received) => {
            assert_eq!(received.session_event_id, event.session_event_id);
        }
        other => panic!("expected SessionEvent, got {other:?}"),
    }

    let (persisted, _next) = control_plane
        .session_events()
        .get_session_events(session_id, None, 10, &[])
        .await
        .expect("query session events");
    assert!(
        persisted
            .iter()
            .any(|e| e.session_event_id == event.session_event_id),
        "expected daemon-emitted session event to be persisted",
    );

    daemon_link.shutdown().await;
}
