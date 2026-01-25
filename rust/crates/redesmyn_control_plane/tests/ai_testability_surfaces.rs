#![cfg(unix)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_control_plane::{ControlPlane, DaemonLinkHandle};
use redesmyn_ids::{CommandId, EventId, HostId, HostInstanceId, RepoId, RequestId, WorkspaceId};
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, CommandState as ClientCommandState, CreateCommandRequest,
    GetCommandRequest, Request, RequestPayload, ResponseResult, WaitForCommandRequest,
    WaitForEventRequest, WaitForIdleRequest,
};
use redesmyn_protocol::daemon::{
    CommandState as DaemonCommandState, CommandUpdate as DaemonCommandUpdate, DaemonFrame,
    DaemonHello, DaemonMessage, RepoAttach,
};
use redesmyn_protocol::ui_driver::{
    GetUiSnapshotRequest, GetUiSnapshotResponse, OpenEpicRequest, TriggerMergeRequest,
    UiComposerState, UiDriverRequest, UiDriverRequestPayload, UiDriverResponse,
    UiDriverResponseResult, UiInFlightAction, UiLeftPaneState, UiPrimaryView, UiSelectionState,
    UiSnapshot,
};
use redesmyn_protocol::{ProtocolEnvelope, ProtocolVersion, RepoScope, Scope};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::codec::ProtobufCodec;
use redesmyn_transport::client::framed::FramedEndpoint;
use redesmyn_transport::in_proc::InProcEndpoint;
use std::sync::Arc;
use tokio::sync::{Mutex, watch};

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

async fn seed_repo_scope(control_plane: &ControlPlane) -> RepoScope {
    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let now_ms = now_ms();

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
    .bind("repo")
    .bind("Test Repo")
    .execute(control_plane.pool())
    .await
    .expect("insert repository");

    RepoScope::new(workspace_id, repo_id)
}

async fn run_mock_daemon(
    mut conn: InProcEndpoint,
    repo_scope: RepoScope,
    expected_dispatch_payload: Vec<u8>,
) {
    let host_id = HostId::new();
    let host_instance_id = HostInstanceId::new();

    let hello = DaemonHello {
        host_id,
        host_instance_id,
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
        let frame = conn
            .recv_frame()
            .await
            .expect("recv control plane hello ack");
        if let DaemonMessage::ControlPlaneHelloAck(ack) = frame.message {
            break ack.accepted_protocol;
        }
    };

    let mut attach_envelope = ProtocolEnvelope::new().with_scope(Scope::from(repo_scope));
    attach_envelope.protocol_major = accepted.major;
    attach_envelope.protocol_minor = accepted.minor;

    let _ = conn
        .send_frame(DaemonFrame::new(
            attach_envelope,
            DaemonMessage::RepoAttach(RepoAttach {
                repo_scope,
                repo_root_hint: None,
            }),
        ))
        .await;

    loop {
        let frame = match conn.recv_frame().await {
            Ok(frame) => frame,
            Err(_) => break,
        };

        let DaemonMessage::CommandDispatch(dispatch) = frame.message else {
            continue;
        };

        assert_eq!(
            dispatch.json_payload, expected_dispatch_payload,
            "expected command payload bytes to be delivered to the daemon"
        );

        for state in [
            DaemonCommandState::Accepted,
            DaemonCommandState::Running,
            DaemonCommandState::Succeeded,
        ] {
            let mut update_envelope =
                ProtocolEnvelope::new().with_scope(Scope::from(dispatch.scope));
            update_envelope.protocol_major = accepted.major;
            update_envelope.protocol_minor = accepted.minor;

            let update = DaemonCommandUpdate {
                command_id: dispatch.command_id,
                state,
                message: Some(format!("state={state:?}")),
                progress: None,
                detail: None,
                error: None,
            };

            if conn
                .send_frame(DaemonFrame::new(
                    update_envelope,
                    DaemonMessage::CommandUpdate(update),
                ))
                .await
                .is_err()
            {
                break;
            }
        }
    }
}

async fn wait_for_socket(path: &Path) {
    for _ in 0..50 {
        if path.exists() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("socket file was not created: {}", path.display());
}

async fn control_plane_request(
    socket_path: &Path,
    scope: Scope,
    payload: RequestPayload,
) -> ResponseResult {
    let stream = tokio::net::UnixStream::connect(socket_path)
        .await
        .expect("connect control plane");
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
    .expect("send control plane request");

    loop {
        let frame = conn.recv().await.expect("recv control plane response");
        let ClientMessage::Response(resp) = frame.message else {
            continue;
        };
        if resp.request_id == request_id {
            return resp.result;
        }
    }
}

async fn create_noop_command(socket_path: &Path, scope: Scope) -> CommandId {
    let result = control_plane_request(
        socket_path,
        scope,
        RequestPayload::CreateCommand(CreateCommandRequest {
            kind: "noop".to_string(),
            target_task_id: None,
            idempotency_key: None,
            created_by: None,
            json_payload: Vec::new(),
        }),
    )
    .await;

    match result {
        ResponseResult::CreateCommand(resp) => resp.command.command_id,
        other => panic!("unexpected CreateCommand result: {other:?}"),
    }
}

async fn wait_for_command(
    socket_path: &Path,
    scope: Scope,
    command_id: CommandId,
) -> ClientCommandState {
    let result = control_plane_request(
        socket_path,
        scope,
        RequestPayload::WaitForCommand(WaitForCommandRequest {
            command_id,
            terminal_states: Vec::new(),
            timeout_ms: 2_000,
        }),
    )
    .await;

    match result {
        ResponseResult::WaitForCommand(resp) => resp.command.state,
        other => panic!("unexpected WaitForCommand result: {other:?}"),
    }
}

async fn get_command_state(
    socket_path: &Path,
    scope: Scope,
    command_id: CommandId,
) -> ClientCommandState {
    let result = control_plane_request(
        socket_path,
        scope,
        RequestPayload::GetCommand(GetCommandRequest { command_id }),
    )
    .await;

    match result {
        ResponseResult::GetCommand(resp) => resp.command.state,
        other => panic!("unexpected GetCommand result: {other:?}"),
    }
}

async fn wait_for_event(
    socket_path: &Path,
    scope: Scope,
    event_type_prefix: &str,
    after_event_id: Option<EventId>,
) -> String {
    let result = control_plane_request(
        socket_path,
        scope,
        RequestPayload::WaitForEvent(WaitForEventRequest {
            filter: redesmyn_protocol::client::EventWaitFilter {
                event_type_prefix: event_type_prefix.to_string(),
                after_event_id,
            },
            timeout_ms: 2_000,
        }),
    )
    .await;

    match result {
        ResponseResult::WaitForEvent(resp) => resp.event_log.event_type,
        other => panic!("unexpected WaitForEvent result: {other:?}"),
    }
}

async fn wait_for_idle(socket_path: &Path, scope: Scope) {
    let result = control_plane_request(
        socket_path,
        scope,
        RequestPayload::WaitForIdle(WaitForIdleRequest {
            scope: None,
            timeout_ms: 2_000,
            quiescence_ms: 10,
        }),
    )
    .await;

    match result {
        ResponseResult::WaitForIdle(_) => {}
        other => panic!("unexpected WaitForIdle result: {other:?}"),
    }
}

#[derive(Debug)]
struct UiDriverHarness {
    control_plane_socket_path: PathBuf,
    scope: Scope,
    state: Arc<Mutex<UiDriverState>>,
    updated_tx: watch::Sender<u64>,
}

#[derive(Debug)]
struct UiDriverState {
    epic_slug: String,
    selected_task_slug: String,
    in_flight: HashMap<CommandId, String>,
    left_pane: UiLeftPaneState,
    errors: Vec<String>,
}

impl UiDriverHarness {
    fn new(control_plane_socket_path: PathBuf, scope: Scope) -> Self {
        let (updated_tx, _updated_rx) = watch::channel(0_u64);
        Self {
            control_plane_socket_path,
            scope,
            state: Arc::new(Mutex::new(UiDriverState {
                epic_slug: String::new(),
                selected_task_slug: String::new(),
                in_flight: HashMap::new(),
                left_pane: UiLeftPaneState {
                    visible: true,
                    collapsed: false,
                    width: 320,
                },
                errors: Vec::new(),
            })),
            updated_tx,
        }
    }

    fn touch(&self) {
        let next = *self.updated_tx.borrow() + 1;
        let _ = self.updated_tx.send(next);
    }

    async fn snapshot(&self) -> UiSnapshot {
        let state = self.state.lock().await;
        UiSnapshot {
            captured_at: redesmyn_protocol::Timestamp::now_utc(),
            primary_view: if state.epic_slug.is_empty() {
                UiPrimaryView::EpicSelector
            } else {
                UiPrimaryView::EpicWorkspace
            },
            left_pane: state.left_pane.clone(),
            selection: UiSelectionState {
                epic_id: None,
                epic_slug: state.epic_slug.clone(),
                task_id: None,
                task_slug: state.selected_task_slug.clone(),
                edge_id: None,
            },
            graph: redesmyn_protocol::ui_driver::UiGraphState::default(),
            in_flight: state
                .in_flight
                .iter()
                .map(|(command_id, label)| UiInFlightAction {
                    label: label.clone(),
                    command_id: Some(*command_id),
                })
                .collect(),
            errors: state
                .errors
                .iter()
                .map(|message| redesmyn_protocol::ui_driver::UiErrorCallout {
                    message: message.clone(),
                })
                .collect(),
            pinned_chat_session_id: None,
            pinned_chat_composer: UiComposerState::default(),
            graph: Default::default(),
        }
    }

    async fn wait_for_no_inflight(&self, timeout: Duration) -> UiSnapshot {
        let mut rx = self.updated_tx.subscribe();
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            let snapshot = self.snapshot().await;
            if snapshot.in_flight.is_empty() {
                return snapshot;
            }

            tokio::select! {
                changed = rx.changed() => {
                    if changed.is_err() {
                        return snapshot;
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    panic!("timed out waiting for UI to clear in_flight actions");
                }
            }
        }
    }

    async fn handle(&self, request: UiDriverRequest) -> UiDriverResponse {
        let request_id = request.request_id;
        match request.payload {
            UiDriverRequestPayload::GetSnapshot(_) => UiDriverResponse {
                request_id,
                result: UiDriverResponseResult::GetSnapshot(GetUiSnapshotResponse {
                    snapshot: self.snapshot().await,
                }),
            },
            UiDriverRequestPayload::OpenEpic(req) => {
                let mut state = self.state.lock().await;
                state.epic_slug = req.epic_slug;
                drop(state);
                self.touch();

                UiDriverResponse {
                    request_id,
                    result: UiDriverResponseResult::OpenEpic(
                        redesmyn_protocol::ui_driver::OpenEpicResponse {},
                    ),
                }
            }
            UiDriverRequestPayload::TriggerMerge(_req) => {
                let command_id =
                    create_noop_command(&self.control_plane_socket_path, self.scope).await;

                {
                    let mut state = self.state.lock().await;
                    state.in_flight.insert(command_id, "merge".to_string());
                }
                self.touch();

                let socket_path = self.control_plane_socket_path.clone();
                let scope = self.scope;
                let updated_tx = self.updated_tx.clone();
                let state = Arc::clone(&self.state);
                tokio::spawn(async move {
                    let _ = wait_for_command(&socket_path, scope, command_id).await;
                    {
                        let mut state = state.lock().await;
                        state.in_flight.remove(&command_id);
                    }
                    let next = *updated_tx.borrow() + 1;
                    let _ = updated_tx.send(next);
                });

                UiDriverResponse {
                    request_id,
                    result: UiDriverResponseResult::TriggerMerge(
                        redesmyn_protocol::ui_driver::TriggerMergeResponse {
                            command_id: Some(command_id),
                        },
                    ),
                }
            }
            other => UiDriverResponse {
                request_id,
                result: UiDriverResponseResult::Error(redesmyn_protocol::ErrorEnvelope::new(
                    redesmyn_protocol::ErrorCategory::InvalidRequest,
                    format!("unimplemented UI driver request: {other:?}"),
                )),
            },
        }
    }
}

#[tokio::test]
async fn ai_testability_surfaces_cover_actions_model_and_ui_snapshot() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let repo_scope = seed_repo_scope(&control_plane).await;
    let scope = Scope::from(repo_scope);

    let (control_plane_conn, daemon_conn) = InProcEndpoint::pair(64);
    let daemon_link = DaemonLinkHandle::start(
        &tokio::runtime::Handle::current(),
        control_plane.clone(),
        control_plane_conn,
    );
    let daemon_task = tokio::spawn(run_mock_daemon(daemon_conn, repo_scope, Vec::new()));

    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
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

    wait_for_socket(&socket_path).await;

    let ui = UiDriverHarness::new(socket_path.clone(), scope);

    let _ = ui
        .handle(UiDriverRequest {
            request_id: RequestId::new(),
            payload: UiDriverRequestPayload::OpenEpic(OpenEpicRequest {
                epic_slug: "gpui".to_string(),
            }),
        })
        .await;

    let response = ui
        .handle(UiDriverRequest {
            request_id: RequestId::new(),
            payload: UiDriverRequestPayload::TriggerMerge(TriggerMergeRequest {
                task_slug: String::new(),
            }),
        })
        .await;

    let command_id = match response.result {
        UiDriverResponseResult::TriggerMerge(resp) => resp.command_id.expect("command_id"),
        other => panic!("unexpected TriggerMerge response: {other:?}"),
    };

    // UI reflects "no silent actions": in-flight is visible immediately.
    let snapshot = match ui
        .handle(UiDriverRequest {
            request_id: RequestId::new(),
            payload: UiDriverRequestPayload::GetSnapshot(GetUiSnapshotRequest {}),
        })
        .await
        .result
    {
        UiDriverResponseResult::GetSnapshot(GetUiSnapshotResponse { snapshot }) => snapshot,
        other => panic!("unexpected GetSnapshot result: {other:?}"),
    };

    assert_eq!(snapshot.primary_view, UiPrimaryView::EpicWorkspace);
    assert_eq!(snapshot.selection.epic_slug, "gpui");
    assert!(
        snapshot
            .in_flight
            .iter()
            .any(|action| action.command_id == Some(command_id)),
        "expected command_id to appear in UI snapshot in_flight"
    );

    // Deterministic waiting primitive: wait for completion without sleeps.
    let terminal_state = wait_for_command(&socket_path, scope, command_id).await;
    assert_eq!(terminal_state, ClientCommandState::Succeeded);

    // Queryable model state: validate command lifecycle result.
    let state = get_command_state(&socket_path, scope, command_id).await;
    assert_eq!(state, ClientCommandState::Succeeded);

    // UI snapshot reflects completion (in-flight cleared).
    let settled = ui.wait_for_no_inflight(Duration::from_secs(2)).await;
    assert!(settled.in_flight.is_empty());

    let _ = shutdown_tx.send(());
    server.await.expect("server task").expect("server exit");
    daemon_link.shutdown().await;
    daemon_task.await.expect("daemon task");
}

#[tokio::test]
async fn wait_primitives_support_events_and_idle() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let repo_scope = seed_repo_scope(&control_plane).await;
    let scope = Scope::from(repo_scope);

    let (control_plane_conn, daemon_conn) = InProcEndpoint::pair(64);
    let daemon_link = DaemonLinkHandle::start(
        &tokio::runtime::Handle::current(),
        control_plane.clone(),
        control_plane_conn,
    );
    let daemon_task = tokio::spawn(run_mock_daemon(daemon_conn, repo_scope, Vec::new()));

    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
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

    wait_for_socket(&socket_path).await;

    let baseline_event_id = control_plane
        .event_log()
        .append_event(
            redesmyn_storage::events::EventScope::Repo {
                workspace_id: repo_scope.workspace_id,
                repo_id: repo_scope.repo_id,
            },
            "test.baseline",
            Vec::new(),
        )
        .await
        .expect("append baseline event");

    let command_id = create_noop_command(&socket_path, scope).await;
    let event_type = wait_for_event(
        &socket_path,
        scope,
        "command.succeeded",
        Some(baseline_event_id),
    )
    .await;
    assert_eq!(event_type, "command.succeeded");

    let terminal_state = wait_for_command(&socket_path, scope, command_id).await;
    assert_eq!(terminal_state, ClientCommandState::Succeeded);

    wait_for_idle(&socket_path, scope).await;

    let _ = shutdown_tx.send(());
    server.await.expect("server task").expect("server exit");
    daemon_link.shutdown().await;
    daemon_task.await.expect("daemon task");
}

#[tokio::test]
async fn create_command_payload_is_dispatched_and_idempotency_payload_mismatch_conflicts() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let repo_scope = seed_repo_scope(&control_plane).await;
    let scope = Scope::from(repo_scope);

    let expected_payload = br#"{"hello":"world"}"#.to_vec();

    let (control_plane_conn, daemon_conn) = InProcEndpoint::pair(64);
    let daemon_link = DaemonLinkHandle::start(
        &tokio::runtime::Handle::current(),
        control_plane.clone(),
        control_plane_conn,
    );
    let daemon_task = tokio::spawn(run_mock_daemon(
        daemon_conn,
        repo_scope,
        expected_payload.clone(),
    ));

    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
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

    wait_for_socket(&socket_path).await;

    let idempotency_key = Some("test-key".to_string());

    let command_id = match control_plane_request(
        &socket_path,
        scope,
        RequestPayload::CreateCommand(CreateCommandRequest {
            kind: "noop".to_string(),
            target_task_id: None,
            idempotency_key: idempotency_key.clone(),
            created_by: None,
            json_payload: expected_payload,
        }),
    )
    .await
    {
        ResponseResult::CreateCommand(resp) => resp.command.command_id,
        other => panic!("unexpected CreateCommand result: {other:?}"),
    };

    let terminal_state = wait_for_command(&socket_path, scope, command_id).await;
    assert_eq!(terminal_state, ClientCommandState::Succeeded);

    let result = control_plane_request(
        &socket_path,
        scope,
        RequestPayload::CreateCommand(CreateCommandRequest {
            kind: "noop".to_string(),
            target_task_id: None,
            idempotency_key,
            created_by: None,
            json_payload: br#"{"hello":"different"}"#.to_vec(),
        }),
    )
    .await;

    let error = match result {
        ResponseResult::Error(err) => err,
        other => panic!("expected Conflict error, got {other:?}"),
    };

    assert_eq!(error.category, redesmyn_protocol::ErrorCategory::Conflict);
    let detail = error.detail.expect("conflict detail");
    let command_id_str = command_id.to_string();

    assert_eq!(
        detail.get("conflict_code").map(String::as_str),
        Some("command_idempotency_payload_mismatch")
    );
    assert_eq!(
        detail.get("idempotency_key").map(String::as_str),
        Some("test-key")
    );
    assert_eq!(
        detail.get("existing_command_id").map(String::as_str),
        Some(command_id_str.as_str())
    );

    let _ = shutdown_tx.send(());
    server.await.expect("server task").expect("server exit");
    daemon_link.shutdown().await;
    daemon_task.await.expect("daemon task");
}
