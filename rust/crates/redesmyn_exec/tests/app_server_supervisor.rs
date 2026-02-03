use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use redesmyn_domain::agent::AppServerTurnIntent;
use redesmyn_exec::app_server::{
    AppServerClient, AppServerConnection, AppServerEvent, AppServerProcess, AppServerProcessError,
    AppServerRequest, AppServerRequestError, AppServerResponse, AppServerSessionSpec,
    AppServerSupervisor, AppServerSupervisorConfig, BoxFuture, SessionControlError,
    StartSessionError,
};
use redesmyn_exec::artifact_store::LocalArtifactStore;
use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::daemon::DaemonMessage;
use redesmyn_protocol::session::{SessionEvent, SessionEventKind, SessionScope};
use redesmyn_protocol::session_live::SessionLiveEvent;
use tokio::sync::mpsc;

#[derive(Clone)]
struct FakeAppServerProcess;

impl AppServerProcess for FakeAppServerProcess {
    fn start(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }

    fn connect(&self) -> BoxFuture<'_, Result<AppServerConnection, AppServerProcessError>> {
        Box::pin(async {
            let (events_tx, events_rx) = mpsc::channel::<AppServerEvent>(16);
            Ok(AppServerConnection {
                client: std::sync::Arc::new(FakeAppServerClient { events_tx }),
                events: events_rx,
            })
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }
}

struct FakeAppServerClient {
    events_tx: mpsc::Sender<AppServerEvent>,
}

impl AppServerClient for FakeAppServerClient {
    fn request(
        &self,
        request: AppServerRequest,
    ) -> BoxFuture<'_, Result<AppServerResponse, AppServerRequestError>> {
        let events_tx = self.events_tx.clone();
        Box::pin(async move {
            match request {
                AppServerRequest::SendMessage { intent } => {
                    let prompt = match intent {
                        AppServerTurnIntent::StartNew { prompt } => prompt,
                        AppServerTurnIntent::Resume { prompt, .. } => prompt,
                    };

                    let _ = events_tx
                        .send(AppServerEvent::AssistantMessage {
                            text: format!("echo: {prompt}"),
                        })
                        .await;
                    Ok(AppServerResponse::MessageAccepted)
                }
                AppServerRequest::Interrupt => Ok(AppServerResponse::Interrupted),
                AppServerRequest::SetPermissionsMode { .. } => {
                    Ok(AppServerResponse::PermissionsModeSet)
                }
                AppServerRequest::SetCodexApprovalPolicy { .. } => {
                    Ok(AppServerResponse::CodexApprovalPolicySet)
                }
                AppServerRequest::SetCodexSandboxPolicy { .. } => {
                    Ok(AppServerResponse::CodexSandboxPolicySet)
                }
                AppServerRequest::RespondPermissionRequest { .. } => {
                    Ok(AppServerResponse::PermissionRequestResponded)
                }
            }
        })
    }
}

#[derive(Clone)]
struct DeltaAppServerProcess;

impl AppServerProcess for DeltaAppServerProcess {
    fn start(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }

    fn connect(&self) -> BoxFuture<'_, Result<AppServerConnection, AppServerProcessError>> {
        Box::pin(async {
            let (events_tx, events_rx) = mpsc::channel::<AppServerEvent>(16);
            Ok(AppServerConnection {
                client: Arc::new(DeltaAppServerClient { events_tx }),
                events: events_rx,
            })
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }
}

struct DeltaAppServerClient {
    events_tx: mpsc::Sender<AppServerEvent>,
}

impl AppServerClient for DeltaAppServerClient {
    fn request(
        &self,
        request: AppServerRequest,
    ) -> BoxFuture<'_, Result<AppServerResponse, AppServerRequestError>> {
        let events_tx = self.events_tx.clone();
        Box::pin(async move {
            match request {
                AppServerRequest::SendMessage { .. } => {
                    let item_id = Some("test_item".to_owned());
                    let _ = events_tx
                        .send(AppServerEvent::AssistantMessageDelta {
                            turn_id: Some("turn_1".to_owned()),
                            item_id: item_id.clone(),
                            delta: "hello ".to_owned(),
                        })
                        .await;
                    let _ = events_tx
                        .send(AppServerEvent::AssistantMessageDelta {
                            turn_id: Some("turn_1".to_owned()),
                            item_id,
                            delta: "world".to_owned(),
                        })
                        .await;
                    let _ = events_tx
                        .send(AppServerEvent::AssistantMessage {
                            text: "hello world".to_owned(),
                        })
                        .await;
                    Ok(AppServerResponse::MessageAccepted)
                }
                AppServerRequest::Interrupt => Ok(AppServerResponse::Interrupted),
                AppServerRequest::SetPermissionsMode { .. } => {
                    Ok(AppServerResponse::PermissionsModeSet)
                }
                AppServerRequest::SetCodexApprovalPolicy { .. } => {
                    Ok(AppServerResponse::CodexApprovalPolicySet)
                }
                AppServerRequest::SetCodexSandboxPolicy { .. } => {
                    Ok(AppServerResponse::CodexSandboxPolicySet)
                }
                AppServerRequest::RespondPermissionRequest { .. } => {
                    Ok(AppServerResponse::PermissionRequestResponded)
                }
            }
        })
    }
}

#[derive(Default)]
struct FlakyConnectProcess {
    connect_calls: AtomicUsize,
}

impl AppServerProcess for FlakyConnectProcess {
    fn start(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }

    fn connect(&self) -> BoxFuture<'_, Result<AppServerConnection, AppServerProcessError>> {
        let connect_call = self.connect_calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            if connect_call > 0 {
                return Err(AppServerProcessError::ConnectFailed {
                    reason: "forced reconnect failure".to_owned(),
                });
            }

            let (events_tx, events_rx) = mpsc::channel::<AppServerEvent>(16);
            Ok(AppServerConnection {
                client: Arc::new(FakeAppServerClient { events_tx }),
                events: events_rx,
            })
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }
}

struct FailingConnectProcess {
    shutdown_calls: Arc<AtomicUsize>,
}

impl AppServerProcess for FailingConnectProcess {
    fn start(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }

    fn connect(&self) -> BoxFuture<'_, Result<AppServerConnection, AppServerProcessError>> {
        Box::pin(async {
            Err(AppServerProcessError::ConnectFailed {
                reason: "forced connect failure".to_owned(),
            })
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        let shutdown_calls = Arc::clone(&self.shutdown_calls);
        Box::pin(async move {
            shutdown_calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }
}

async fn collect_session_records(
    rx: &mut mpsc::Receiver<redesmyn_protocol::daemon::DaemonFrame>,
    session_id: redesmyn_ids::SessionId,
    timeout: Duration,
) -> Vec<SessionEvent> {
    let frame = tokio::time::timeout(timeout, rx.recv())
        .await
        .expect("timeout waiting for daemon frame")
        .expect("frame channel closed");

    match frame.message {
        DaemonMessage::SessionEventBatch(batch) => batch
            .events
            .into_iter()
            .filter(|event| event.session_id == session_id)
            .collect(),
        _ => Vec::new(),
    }
}

async fn collect_live_events(
    rx: &mut mpsc::Receiver<redesmyn_protocol::daemon::DaemonFrame>,
    session_id: redesmyn_ids::SessionId,
    timeout: Duration,
) -> Vec<SessionLiveEvent> {
    tokio::time::timeout(timeout, async {
        loop {
            let frame = rx.recv().await.expect("frame channel closed");
            if let DaemonMessage::SessionLiveEventBatch(batch) = frame.message {
                let events: Vec<_> = batch
                    .events
                    .into_iter()
                    .filter(|event| event.session_id == session_id)
                    .collect();
                if !events.is_empty() {
                    return events;
                }
            }
        }
    })
    .await
    .expect("timeout waiting for live session events")
}

async fn collect_until_session_ended(
    rx: &mut mpsc::Receiver<redesmyn_protocol::daemon::DaemonFrame>,
    session_id: redesmyn_ids::SessionId,
    timeout: Duration,
) -> Vec<SessionEvent> {
    let mut out = Vec::new();
    loop {
        let records = collect_session_records(rx, session_id, timeout).await;
        let ended = records
            .iter()
            .any(|event| matches!(event.kind, SessionEventKind::SessionEnded(_)));
        out.extend(records);
        if ended {
            return out;
        }
    }
}

async fn wait_for_session_cleanup(
    supervisor: &AppServerSupervisor,
    session_id: redesmyn_ids::SessionId,
) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match supervisor.stop_session(session_id).await {
                Err(SessionControlError::UnknownSession { .. }) => break,
                Err(SessionControlError::SessionClosed { .. }) | Ok(()) => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }
    })
    .await
    .expect("timeout waiting for supervisor cleanup");
}

#[tokio::test]
async fn send_message_round_trips_and_emits_structured_event() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let supervisor = AppServerSupervisor::new(
        AppServerSupervisorConfig::default(),
        artifact_store,
        frames_tx,
    )
    .await
    .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let session_id = SessionId::new();
    let session_id = supervisor
        .start_session(
            session_id,
            Some(task_id),
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process: std::sync::Arc::new(FakeAppServerProcess),
            },
        )
        .await
        .expect("start_session");

    let response = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::StartNew {
                prompt: "hi".to_owned(),
            },
        )
        .await
        .expect("send_message");
    assert_eq!(response, AppServerResponse::MessageAccepted);

    let mut saw_assistant_message = false;
    for _ in 0..10 {
        let records =
            collect_session_records(&mut frames_rx, session_id, Duration::from_secs(2)).await;
        for event in records {
            if matches!(event.kind, SessionEventKind::AssistantMessage(_)) {
                saw_assistant_message = true;
                break;
            }
        }
        if saw_assistant_message {
            break;
        }
    }
    assert!(saw_assistant_message, "expected an assistant message event");

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}

#[tokio::test]
async fn send_message_emits_live_assistant_deltas() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let supervisor = AppServerSupervisor::new(
        AppServerSupervisorConfig::default(),
        artifact_store,
        frames_tx,
    )
    .await
    .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let session_id = SessionId::new();
    let session_id = supervisor
        .start_session(
            session_id,
            Some(task_id),
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process: Arc::new(DeltaAppServerProcess),
            },
        )
        .await
        .expect("start_session");

    let response = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::StartNew {
                prompt: "hi".to_owned(),
            },
        )
        .await
        .expect("send_message");
    assert_eq!(response, AppServerResponse::MessageAccepted);

    let live_events = collect_live_events(&mut frames_rx, session_id, Duration::from_secs(2)).await;
    assert!(live_events
        .iter()
        .all(|event| event.item_id.as_deref() == Some("test_item")));
    assert_eq!(
        live_events
            .into_iter()
            .filter_map(|event| match event.kind {
                redesmyn_protocol::session_live::SessionLiveEventKind::AssistantMessageDelta(
                    delta,
                ) => Some(delta.delta),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec!["hello ".to_owned(), "world".to_owned()]
    );
}

#[tokio::test]
async fn concurrent_sessions_do_not_clear_active_task_marker() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let supervisor = AppServerSupervisor::new(
        AppServerSupervisorConfig::default(),
        artifact_store,
        frames_tx,
    )
    .await
    .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let session_1_id = SessionId::new();
    let session_1 = supervisor
        .start_session(
            session_1_id,
            Some(task_id),
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: true,
                process: Arc::new(FakeAppServerProcess),
            },
        )
        .await
        .expect("start_session 1");

    let session_2_id = SessionId::new();
    let session_2 = supervisor
        .start_session(
            session_2_id,
            Some(task_id),
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: true,
                process: Arc::new(FakeAppServerProcess),
            },
        )
        .await
        .expect("start_session 2");

    supervisor
        .stop_session(session_1)
        .await
        .expect("stop_session 1");

    let _records =
        collect_until_session_ended(&mut frames_rx, session_1, Duration::from_secs(5)).await;
    wait_for_session_cleanup(&supervisor, session_1).await;

    let err = supervisor
        .start_session(
            SessionId::new(),
            Some(task_id),
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process: Arc::new(FakeAppServerProcess),
            },
        )
        .await
        .unwrap_err();

    assert!(
        matches!(err, StartSessionError::TaskHasActiveSession { .. }),
        "unexpected error: {err:?}"
    );

    supervisor
        .stop_session(session_2)
        .await
        .expect("stop_session 2");
    let _records =
        collect_until_session_ended(&mut frames_rx, session_2, Duration::from_secs(5)).await;
    wait_for_session_cleanup(&supervisor, session_2).await;

    supervisor.shutdown().await;
}

#[tokio::test]
async fn reconnect_failure_does_not_kill_event_forwarding() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let supervisor = AppServerSupervisor::new(
        AppServerSupervisorConfig::default(),
        artifact_store,
        frames_tx,
    )
    .await
    .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let session_id = SessionId::new();
    let session_id = supervisor
        .start_session(
            session_id,
            Some(task_id),
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process: Arc::new(FlakyConnectProcess::default()),
            },
        )
        .await
        .expect("start_session");

    assert!(supervisor.reconnect_session(session_id).await.is_err());

    let response = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::StartNew {
                prompt: "hi".to_owned(),
            },
        )
        .await
        .expect("send_message");
    assert_eq!(response, AppServerResponse::MessageAccepted);

    let mut saw_assistant_message = false;
    for _ in 0..10 {
        let records =
            collect_session_records(&mut frames_rx, session_id, Duration::from_secs(2)).await;
        for event in records {
            if matches!(event.kind, SessionEventKind::AssistantMessage(_)) {
                saw_assistant_message = true;
                break;
            }
        }
        if saw_assistant_message {
            break;
        }
    }
    assert!(saw_assistant_message, "expected an assistant message event");

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}

#[tokio::test]
async fn connect_failure_triggers_shutdown() {
    let (frames_tx, _frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let supervisor = AppServerSupervisor::new(
        AppServerSupervisorConfig::default(),
        artifact_store,
        frames_tx,
    )
    .await
    .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let shutdown_calls = Arc::new(AtomicUsize::new(0));
    let err = supervisor
        .start_session(
            SessionId::new(),
            Some(task_id),
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process: Arc::new(FailingConnectProcess {
                    shutdown_calls: Arc::clone(&shutdown_calls),
                }),
            },
        )
        .await
        .unwrap_err();

    assert!(
        matches!(
            err,
            StartSessionError::Process(AppServerProcessError::ConnectFailed { .. })
        ),
        "unexpected error: {err:?}"
    );
    assert_eq!(shutdown_calls.load(Ordering::Relaxed), 1);

    supervisor.shutdown().await;
}
