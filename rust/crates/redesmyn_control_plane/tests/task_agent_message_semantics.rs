#![cfg(unix)]

use std::collections::HashMap;
use std::time::Duration;

use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_control_plane::{ControlPlane, DaemonLinkHandle};
use redesmyn_ids::{CommandId, EpicId, RepoId, RequestId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::agent_commands::{
    ResumeByIdTaskAgentTurnCommand, SESSION_AGENT_RESUME_BY_ID_TURN, StartTaskAgentSessionCommand,
    TASK_AGENT_START,
};
use redesmyn_protocol::client::{
    AgentKind, AgentMessageConflictAction, ClientFrame, ClientMessage, Request, RequestPayload,
    ResponseResult, SendTaskAgentMessageRequest, WaitForCommandRequest,
};
use redesmyn_protocol::daemon::{
    CommandDispatch, CommandState as DaemonCommandState, CommandUpdate as DaemonCommandUpdate,
    DaemonFrame, DaemonHello, DaemonMessage, RepoAttach, SessionEventBatch,
};
use redesmyn_protocol::session::{
    AssistantMessage, ExternalSessionRef, InterfaceMode, SessionEventKind, SessionScope,
    TurnCompleted, TurnStarted, UserMessage,
};
use redesmyn_protocol::{
    ErrorCategory, ProtocolEnvelope, ProtocolVersion, RepoScope, Scope, SessionEvent, Timestamp,
};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::codec::ProtobufCodec;
use redesmyn_transport::client::framed::FramedEndpoint;
use redesmyn_transport::in_proc::InProcEndpoint;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
struct DispatchObserved {
    command_kind: String,
    json_payload: Vec<u8>,
}

fn ts_from_ms(ms: i64) -> Timestamp {
    let nanos = i128::from(ms).saturating_mul(1_000_000);
    let dt = time::OffsetDateTime::from_unix_timestamp_nanos(nanos)
        .unwrap_or(time::OffsetDateTime::UNIX_EPOCH);
    Timestamp::from_offset_date_time(dt)
}

async fn seed_repo_and_task(control_plane: &ControlPlane) -> (RepoScope, TaskId) {
    let pool = control_plane.pool();

    let now_ms = 1_i64;
    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();
    let task_id = TaskId::new();
    let branch_name = format!("rn/task/{}", task_id);

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
    .bind("repo")
    .bind("Repo")
    .execute(pool)
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
    .bind("epic")
    .bind("Epic")
    .execute(pool)
    .await
    .expect("insert epic");

    sqlx::query(
        r#"
        INSERT INTO tasks (id, epic_id, created_at_ms, updated_at_ms, title, branch_name, merge_readiness)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'unknown')
        "#,
    )
    .bind(task_id)
    .bind(epic_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("Task")
    .bind(branch_name)
    .execute(pool)
    .await
    .expect("insert task");

    (RepoScope::new(workspace_id, repo_id), task_id)
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

async fn control_plane_request(
    socket_path: &std::path::Path,
    scope: Scope,
    payload: RequestPayload,
) -> ResponseResult {
    let stream = connect_with_retry(socket_path).await;
    let mut conn = FramedEndpoint::new(stream, ProtobufCodec::new());

    let request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope),
        ClientMessage::Request(Request {
            request_id,
            payload,
        }),
    ))
    .await
    .expect("send request");

    loop {
        let frame = conn.recv().await.expect("recv frame");
        let ClientMessage::Response(resp) = frame.message else {
            continue;
        };
        if resp.request_id == request_id {
            return resp.result;
        }
    }
}

async fn send_daemon_command_update(
    conn: &InProcEndpoint,
    accepted: ProtocolVersion,
    scope: RepoScope,
    command_id: CommandId,
    state: DaemonCommandState,
) {
    let mut envelope = ProtocolEnvelope::new().with_scope(Scope::from(scope));
    envelope.protocol_major = accepted.major;
    envelope.protocol_minor = accepted.minor;

    let update = DaemonCommandUpdate {
        command_id,
        state,
        message: None,
        progress: None,
        detail: None,
        error: None,
    };

    conn.send_frame(DaemonFrame::new(
        envelope,
        DaemonMessage::CommandUpdate(update),
    ))
    .await
    .expect("send command update");
}

async fn send_session_event(
    conn: &InProcEndpoint,
    accepted: ProtocolVersion,
    scope: RepoScope,
    event: SessionEvent,
) {
    let mut envelope = ProtocolEnvelope::new().with_scope(Scope::from(scope));
    envelope.protocol_major = accepted.major;
    envelope.protocol_minor = accepted.minor;

    conn.send_frame(DaemonFrame::new(
        envelope,
        DaemonMessage::SessionEventBatch(SessionEventBatch {
            events: vec![event],
        }),
    ))
    .await
    .expect("send session event batch");
}

async fn run_mock_daemon(
    mut conn: InProcEndpoint,
    repo_scope: RepoScope,
    dispatch_tx: mpsc::Sender<DispatchObserved>,
) {
    let hello = DaemonHello {
        host_id: redesmyn_ids::HostId::new(),
        host_instance_id: redesmyn_ids::HostInstanceId::new(),
        capabilities: vec!["test".to_string()],
        supported_protocol: ProtocolVersion::CURRENT,
    };

    let envelope = ProtocolEnvelope::new();
    conn.send_frame(DaemonFrame::new(
        envelope,
        DaemonMessage::DaemonHello(hello),
    ))
    .await
    .expect("send daemon hello");

    let accepted = loop {
        let frame = conn.recv_frame().await.expect("recv hello ack");
        if let DaemonMessage::ControlPlaneHelloAck(ack) = frame.message {
            break ack.accepted_protocol;
        }
    };

    let mut attach_envelope = ProtocolEnvelope::new().with_scope(Scope::from(repo_scope));
    attach_envelope.protocol_major = accepted.major;
    attach_envelope.protocol_minor = accepted.minor;
    conn.send_frame(DaemonFrame::new(
        attach_envelope,
        DaemonMessage::RepoAttach(RepoAttach {
            repo_scope,
            repo_root_hint: None,
        }),
    ))
    .await
    .expect("send repo attach");

    let mut session_task_ids: HashMap<SessionId, TaskId> = HashMap::new();
    let mut logical_ms: i64 = 1_700_000_000_000;
    let mut turn_in_progress: HashMap<SessionId, bool> = HashMap::new();

    loop {
        let frame = match conn.recv_frame().await {
            Ok(frame) => frame,
            Err(_) => break,
        };

        let DaemonMessage::CommandDispatch(CommandDispatch {
            command_id,
            scope,
            command_kind,
            json_payload,
        }) = frame.message
        else {
            continue;
        };

        let _ = dispatch_tx
            .send(DispatchObserved {
                command_kind: command_kind.clone(),
                json_payload: json_payload.clone(),
            })
            .await;

        send_daemon_command_update(
            &conn,
            accepted,
            scope,
            command_id,
            DaemonCommandState::Accepted,
        )
        .await;
        send_daemon_command_update(
            &conn,
            accepted,
            scope,
            command_id,
            DaemonCommandState::Running,
        )
        .await;

        match command_kind.as_str() {
            TASK_AGENT_START => {
                let payload: StartTaskAgentSessionCommand =
                    serde_json::from_slice(&json_payload).expect("decode start payload");
                session_task_ids.insert(payload.session_id, payload.task_id);

                logical_ms += 1;
                let external_session_ref = ExternalSessionRef::CodexThread {
                    thread_id: "thread-1".to_string(),
                    turn_id: None,
                };

                send_session_event(
                    &conn,
                    accepted,
                    scope,
                    SessionEvent {
                        session_event_id: redesmyn_ids::SessionEventId::new(),
                        created_at: ts_from_ms(logical_ms),
                        scope: SessionScope::Task {
                            task_id: payload.task_id,
                        },
                        session_id: payload.session_id,
                        turn_id: None,
                        kind: SessionEventKind::TurnStarted(TurnStarted {
                            interface_mode: InterfaceMode::Structured,
                            external_session_ref: Some(external_session_ref.clone()),
                            idempotency_key: None,
                            log_offset_bytes: None,
                        }),
                    },
                )
                .await;

                logical_ms += 1;
                send_session_event(
                    &conn,
                    accepted,
                    scope,
                    SessionEvent {
                        session_event_id: redesmyn_ids::SessionEventId::new(),
                        created_at: ts_from_ms(logical_ms),
                        scope: SessionScope::Task {
                            task_id: payload.task_id,
                        },
                        session_id: payload.session_id,
                        turn_id: None,
                        kind: SessionEventKind::AssistantMessage(AssistantMessage {
                            text: "hello from daemon".to_string(),
                            preview: "hello from daemon".to_string(),
                            full_text_artifact: None,
                        }),
                    },
                )
                .await;

                logical_ms += 1;
                send_session_event(
                    &conn,
                    accepted,
                    scope,
                    SessionEvent {
                        session_event_id: redesmyn_ids::SessionEventId::new(),
                        created_at: ts_from_ms(logical_ms),
                        scope: SessionScope::Task {
                            task_id: payload.task_id,
                        },
                        session_id: payload.session_id,
                        turn_id: None,
                        kind: SessionEventKind::TurnCompleted(TurnCompleted {
                            interface_mode: InterfaceMode::Structured,
                            external_session_ref: Some(external_session_ref),
                            exit_code: Some(0),
                            error: None,
                        }),
                    },
                )
                .await;

                turn_in_progress.insert(payload.session_id, false);

                send_daemon_command_update(
                    &conn,
                    accepted,
                    scope,
                    command_id,
                    DaemonCommandState::Succeeded,
                )
                .await;
            }
            SESSION_AGENT_RESUME_BY_ID_TURN => {
                let payload: ResumeByIdTaskAgentTurnCommand =
                    serde_json::from_slice(&json_payload).expect("decode resume payload");

                let task_id = *session_task_ids
                    .get(&payload.session_id)
                    .expect("session exists");
                let in_progress = *turn_in_progress.get(&payload.session_id).unwrap_or(&false);

                let external_session_ref = ExternalSessionRef::CodexThread {
                    thread_id: "thread-1".to_string(),
                    turn_id: None,
                };

                if payload.interrupt_turn && in_progress {
                    logical_ms += 1;
                    send_session_event(
                        &conn,
                        accepted,
                        scope,
                        SessionEvent {
                            session_event_id: redesmyn_ids::SessionEventId::new(),
                            created_at: ts_from_ms(logical_ms),
                            scope: SessionScope::Task { task_id },
                            session_id: payload.session_id,
                            turn_id: None,
                            kind: SessionEventKind::TurnCompleted(TurnCompleted {
                                interface_mode: InterfaceMode::Structured,
                                external_session_ref: Some(external_session_ref.clone()),
                                exit_code: None,
                                error: None,
                            }),
                        },
                    )
                    .await;
                }

                logical_ms += 1;
                send_session_event(
                    &conn,
                    accepted,
                    scope,
                    SessionEvent {
                        session_event_id: redesmyn_ids::SessionEventId::new(),
                        created_at: ts_from_ms(logical_ms),
                        scope: SessionScope::Task { task_id },
                        session_id: payload.session_id,
                        turn_id: None,
                        kind: SessionEventKind::TurnStarted(TurnStarted {
                            interface_mode: InterfaceMode::Structured,
                            external_session_ref: Some(external_session_ref.clone()),
                            idempotency_key: None,
                            log_offset_bytes: None,
                        }),
                    },
                )
                .await;

                logical_ms += 1;
                send_session_event(
                    &conn,
                    accepted,
                    scope,
                    SessionEvent {
                        session_event_id: redesmyn_ids::SessionEventId::new(),
                        created_at: ts_from_ms(logical_ms),
                        scope: SessionScope::Task { task_id },
                        session_id: payload.session_id,
                        turn_id: None,
                        kind: SessionEventKind::UserMessage(UserMessage {
                            text: payload.prompt.clone(),
                            preview: payload.prompt.clone(),
                            full_text_artifact: None,
                            image_attachments: Vec::new(),
                        }),
                    },
                )
                .await;

                if payload.interrupt_turn && in_progress {
                    logical_ms += 1;
                    send_session_event(
                        &conn,
                        accepted,
                        scope,
                        SessionEvent {
                            session_event_id: redesmyn_ids::SessionEventId::new(),
                            created_at: ts_from_ms(logical_ms),
                            scope: SessionScope::Task { task_id },
                            session_id: payload.session_id,
                            turn_id: None,
                            kind: SessionEventKind::TurnCompleted(TurnCompleted {
                                interface_mode: InterfaceMode::Structured,
                                external_session_ref: Some(external_session_ref),
                                exit_code: Some(0),
                                error: None,
                            }),
                        },
                    )
                    .await;

                    turn_in_progress.insert(payload.session_id, false);
                    send_daemon_command_update(
                        &conn,
                        accepted,
                        scope,
                        command_id,
                        DaemonCommandState::Succeeded,
                    )
                    .await;
                } else {
                    turn_in_progress.insert(payload.session_id, true);
                    // Keep the command in-flight (no terminal update) to exercise conflict logic.
                }
            }
            _ => {
                send_daemon_command_update(
                    &conn,
                    accepted,
                    scope,
                    command_id,
                    DaemonCommandState::Succeeded,
                )
                .await;
            }
        }
    }
}

async fn wait_for_turn_in_progress(pool: &sqlx::SqlitePool, session_id: SessionId) -> bool {
    for _ in 0..50 {
        let started: Option<(i64, redesmyn_ids::SessionEventId)> = sqlx::query_as(
            r#"
            SELECT created_at_ms, id
            FROM session_events
            WHERE session_id = ?1 AND kind = 'turn_started'
            ORDER BY created_at_ms DESC, id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_optional(pool)
        .await
        .expect("query last turn_started");

        let completed: Option<(i64, redesmyn_ids::SessionEventId)> = sqlx::query_as(
            r#"
            SELECT created_at_ms, id
            FROM session_events
            WHERE session_id = ?1 AND kind = 'turn_completed'
            ORDER BY created_at_ms DESC, id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_optional(pool)
        .await
        .expect("query last turn_completed");

        let in_progress = match (started, completed) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some((s_ms, s_id)), Some((c_ms, c_id))) => {
                s_ms > c_ms || (s_ms == c_ms && s_id > c_id)
            }
        };

        if in_progress {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}

#[tokio::test]
async fn send_task_agent_message_structured_resume_conflict_interrupt() {
    redesmyn_logging::init();

    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (repo_scope, task_id) = seed_repo_and_task(&control_plane).await;

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
    let server = tokio::spawn({
        let socket_path = socket_path.clone();
        let control_plane = control_plane.clone();
        async move {
            redesmyn_control_plane::client_api::serve_client_api_uds(
                control_plane,
                socket_path,
                ClientApiCodec::Protobuf,
                &mut shutdown_rx,
            )
            .await
        }
    });

    let (cp_conn, daemon_conn) = InProcEndpoint::pair(64);
    let daemon_link = DaemonLinkHandle::start(
        &tokio::runtime::Handle::current(),
        control_plane.clone(),
        cp_conn,
    );

    let (dispatch_tx, mut dispatch_rx) = mpsc::channel::<DispatchObserved>(32);
    tokio::spawn(run_mock_daemon(daemon_conn, repo_scope, dispatch_tx));

    let scope = Scope::from(repo_scope);

    // 1) First message starts a structured session.
    let first = control_plane_request(
        &socket_path,
        scope,
        RequestPayload::SendTaskAgentMessage(SendTaskAgentMessageRequest {
            task_id,
            message: "hello".to_string(),
            intent: None,
            on_conflict: AgentMessageConflictAction::Fail,
            interrupt: None,
            agent_kind: AgentKind::Codex,
        }),
    )
    .await;

    let ResponseResult::SendTaskAgentMessage(first) = first else {
        panic!("unexpected response: {first:?}");
    };
    assert_eq!(
        first.delivery,
        redesmyn_protocol::client::TaskAgentMessageDelivery::StructuredStarted
    );
    assert_eq!(
        first.conversation_continuity,
        redesmyn_protocol::client::TaskAgentMessageConversationContinuity::Broken
    );

    let dispatch = dispatch_rx.recv().await.expect("dispatch");
    assert_eq!(dispatch.command_kind, TASK_AGENT_START);

    let started_payload: StartTaskAgentSessionCommand =
        serde_json::from_slice(&dispatch.json_payload).expect("decode start payload");
    assert_eq!(started_payload.session_id, first.session_id);

    // Wait for the start command to complete.
    let wait = control_plane_request(
        &socket_path,
        scope,
        RequestPayload::WaitForCommand(WaitForCommandRequest {
            command_id: first.command.command_id,
            terminal_states: Vec::new(),
            timeout_ms: 2_000,
        }),
    )
    .await;
    assert!(matches!(wait, ResponseResult::WaitForCommand(_)));

    // 2) Second message resumes via resume-by-id turn and leaves the turn in progress.
    let second = control_plane_request(
        &socket_path,
        scope,
        RequestPayload::SendTaskAgentMessage(SendTaskAgentMessageRequest {
            task_id,
            message: "resume".to_string(),
            intent: None,
            on_conflict: AgentMessageConflictAction::Fail,
            interrupt: None,
            agent_kind: AgentKind::Codex,
        }),
    )
    .await;
    let ResponseResult::SendTaskAgentMessage(second) = second else {
        panic!("unexpected response: {second:?}");
    };
    assert_eq!(
        second.delivery,
        redesmyn_protocol::client::TaskAgentMessageDelivery::StructuredResumed
    );
    assert_eq!(
        second.conversation_continuity,
        redesmyn_protocol::client::TaskAgentMessageConversationContinuity::Kept
    );
    assert_eq!(second.session_id, first.session_id);

    let dispatch = dispatch_rx.recv().await.expect("dispatch");
    assert_eq!(dispatch.command_kind, SESSION_AGENT_RESUME_BY_ID_TURN);

    // Wait until the control plane sees an in-progress turn.
    assert!(
        wait_for_turn_in_progress(control_plane.pool(), first.session_id).await,
        "expected a turn to be in progress"
    );

    let (persisted_events, _cursor) = control_plane
        .session_events()
        .get_session_events(first.session_id, None, 32, &[])
        .await
        .expect("query persisted session events");
    assert!(
        persisted_events
            .iter()
            .any(|event| matches!(event.kind, SessionEventKind::TaskAgentMessageSent(_))),
        "expected task_agent_message_sent durable event"
    );

    // 3) A conflicting message with on_conflict=fail returns 409 and does not dispatch.
    let conflict = control_plane_request(
        &socket_path,
        scope,
        RequestPayload::SendTaskAgentMessage(SendTaskAgentMessageRequest {
            task_id,
            message: "should conflict".to_string(),
            intent: None,
            on_conflict: AgentMessageConflictAction::Fail,
            interrupt: None,
            agent_kind: AgentKind::Codex,
        }),
    )
    .await;

    match conflict {
        ResponseResult::Error(err) => {
            assert_eq!(err.category, ErrorCategory::Conflict);
            let code = err
                .detail
                .as_ref()
                .and_then(|m| m.get("conflict_code"))
                .cloned()
                .unwrap_or_default();
            assert_eq!(code, "structured_turn_in_progress");
        }
        other => panic!("expected conflict error, got {other:?}"),
    }

    let no_dispatch = tokio::time::timeout(Duration::from_millis(200), dispatch_rx.recv()).await;
    assert!(
        no_dispatch.is_err(),
        "expected no daemon dispatch for conflict"
    );

    // 4) on_conflict=interrupt_turn dispatches with interrupt_turn=true.
    let fourth = control_plane_request(
        &socket_path,
        scope,
        RequestPayload::SendTaskAgentMessage(SendTaskAgentMessageRequest {
            task_id,
            message: "interrupt".to_string(),
            intent: None,
            on_conflict: AgentMessageConflictAction::InterruptTurn,
            interrupt: None,
            agent_kind: AgentKind::Codex,
        }),
    )
    .await;

    let ResponseResult::SendTaskAgentMessage(fourth) = fourth else {
        panic!("unexpected response: {fourth:?}");
    };
    assert_eq!(
        fourth.delivery,
        redesmyn_protocol::client::TaskAgentMessageDelivery::StructuredResumed
    );

    let dispatch = dispatch_rx.recv().await.expect("dispatch");
    assert_eq!(dispatch.command_kind, SESSION_AGENT_RESUME_BY_ID_TURN);
    let payload: ResumeByIdTaskAgentTurnCommand =
        serde_json::from_slice(&dispatch.json_payload).expect("decode resume payload");
    assert!(payload.interrupt_turn, "expected interrupt_turn=true");

    // Clean shutdown.
    daemon_link.shutdown().await;
    drop(shutdown_tx);
    server.abort();
}
