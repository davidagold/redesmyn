use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use redesmyn_domain::agent::{AppServerTurnIntent, ExternalSessionRef as DomainExternalSessionRef};
use redesmyn_exec::app_server::{
    AppServerConnection, AppServerProcess, AppServerProcessError, AppServerSessionSpec,
    AppServerSupervisor, AppServerSupervisorConfig, BoxFuture,
};
use redesmyn_exec::artifact_store::LocalArtifactStore;
use redesmyn_exec::codex_app_server::{
    CodexAppServerProcess, CodexAppServerProcessConfig, JsonRpcWireFormat,
};
use redesmyn_ids::TaskId;
use redesmyn_protocol::daemon::DaemonMessage;
use redesmyn_protocol::session::{
    ExternalSessionRef, SessionEvent, SessionEventKind, SessionScope,
};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::sync::{Mutex, mpsc};

#[derive(Default)]
struct FakeCodexCounters {
    thread_start_calls: AtomicUsize,
    thread_resume_calls: AtomicUsize,
    turn_interrupt_calls: AtomicUsize,
}

struct FakeCodexState {
    counters: Arc<FakeCodexCounters>,
    inflight: Mutex<Option<InflightTurn>>,
    next_turn: AtomicUsize,
}

struct InflightTurn {
    turn_id: String,
    interrupt: Arc<tokio::sync::Notify>,
}

impl FakeCodexState {
    fn new(counters: Arc<FakeCodexCounters>) -> Arc<Self> {
        Arc::new(Self {
            counters,
            inflight: Mutex::new(None),
            next_turn: AtomicUsize::new(1),
        })
    }

    fn alloc_turn_id(&self) -> String {
        let n = self.next_turn.fetch_add(1, Ordering::Relaxed);
        format!("turn_{n}")
    }
}

async fn fake_codex_server(stream: tokio::io::DuplexStream, state: Arc<FakeCodexState>) {
    let thread_id: &str = "thr_test";
    let (mut reader, writer) = tokio::io::split(stream);
    let writer: Writer = Arc::new(Mutex::new(writer));

    let mut buf = Vec::<u8>::new();
    loop {
        let mut chunk = [0u8; 4096];
        let read = match reader.read(&mut chunk).await {
            Ok(0) => return,
            Ok(n) => n,
            Err(_) => return,
        };
        buf.extend_from_slice(&chunk[..read]);

        while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
            let line = buf.drain(..pos + 1).collect::<Vec<u8>>();
            let line = String::from_utf8_lossy(&line);
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let msg: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let Some(method) = msg.get("method").and_then(|m| m.as_str()) else {
                continue;
            };
            let id = msg.get("id").cloned();
            let params = msg
                .get("params")
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            match (method, id) {
                ("initialize", Some(id)) => {
                    send_response(&writer, id, serde_json::json!({"userAgent": "fake"})).await;
                }
                ("thread/start", Some(id)) => {
                    state
                        .counters
                        .thread_start_calls
                        .fetch_add(1, Ordering::Relaxed);
                    send_response(
                        &writer,
                        id.clone(),
                        serde_json::json!({"thread": { "id": thread_id }}),
                    )
                    .await;
                    send_notification(
                        &writer,
                        "thread/started",
                        serde_json::json!({"thread": { "id": thread_id }}),
                    )
                    .await;
                }
                ("thread/resume", Some(id)) => {
                    state
                        .counters
                        .thread_resume_calls
                        .fetch_add(1, Ordering::Relaxed);
                    send_response(
                        &writer,
                        id.clone(),
                        serde_json::json!({"thread": { "id": thread_id }}),
                    )
                    .await;
                    send_notification(
                        &writer,
                        "thread/started",
                        serde_json::json!({"thread": { "id": thread_id }}),
                    )
                    .await;
                }
                ("turn/start", Some(id)) => {
                    let prompt = params
                        .get("input")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.first())
                        .and_then(|v| v.get("text"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned();

                    let turn_id = state.alloc_turn_id();
                    send_response(
                        &writer,
                        id.clone(),
                        serde_json::json!({"turn": { "id": turn_id }}),
                    )
                    .await;

                    send_notification(
                        &writer,
                        "turn/started",
                        serde_json::json!({
                            "threadId": thread_id,
                            "turn": { "id": turn_id, "status": "inProgress", "items": [], "error": null }
                        }),
                    )
                    .await;

                    let interrupt = Arc::new(tokio::sync::Notify::new());
                    {
                        let mut inflight = state.inflight.lock().await;
                        *inflight = Some(InflightTurn {
                            turn_id: turn_id.clone(),
                            interrupt: Arc::clone(&interrupt),
                        });
                    }

                    let writer_for_turn = Arc::clone(&writer);
                    let state_for_turn = Arc::clone(&state);
                    tokio::spawn(async move {
                        let notified = interrupt.notified();
                        let interrupted = if prompt.contains("block") {
                            tokio::select! {
                                _ = tokio::time::sleep(Duration::from_secs(60)) => false,
                                _ = notified => true,
                            }
                        } else {
                            tokio::select! {
                                _ = tokio::time::sleep(Duration::from_millis(20)) => false,
                                _ = notified => true,
                            }
                        };

                        if interrupted {
                            send_notification(
                                &writer_for_turn,
                                "turn/completed",
                                serde_json::json!({
                                    "threadId": thread_id,
                                    "turn": { "id": turn_id, "status": "interrupted", "items": [], "error": null }
                                }),
                            )
                            .await;
                        } else {
                            send_notification(
                                &writer_for_turn,
                                "item/completed",
                                serde_json::json!({
                                    "threadId": thread_id,
                                    "turnId": turn_id,
                                    "item": { "type": "userMessage", "id": "user_1", "content": [{ "type": "text", "text": prompt, "textElements": [] }] }
                                }),
                            )
                            .await;
                            send_notification(
                                &writer_for_turn,
                                "item/completed",
                                serde_json::json!({
                                    "threadId": thread_id,
                                    "turnId": turn_id,
                                    "item": { "type": "agentMessage", "id": "agent_1", "text": format!("echo: {prompt}") }
                                }),
                            )
                            .await;
                            send_notification(
                                &writer_for_turn,
                                "turn/completed",
                                serde_json::json!({
                                    "threadId": thread_id,
                                    "turn": { "id": turn_id, "status": "completed", "items": [], "error": null }
                                }),
                            )
                            .await;
                        }

                        let mut inflight = state_for_turn.inflight.lock().await;
                        *inflight = None;
                    });
                }
                ("turn/interrupt", Some(id)) => {
                    state
                        .counters
                        .turn_interrupt_calls
                        .fetch_add(1, Ordering::Relaxed);
                    let turn_id = params.get("turnId").and_then(|v| v.as_str());
                    let inflight = state.inflight.lock().await;
                    if let (Some(turn_id), Some(inflight_turn)) = (turn_id, inflight.as_ref()) {
                        if inflight_turn.turn_id == turn_id {
                            inflight_turn.interrupt.notify_waiters();
                        }
                    }
                    send_response(&writer, id, serde_json::json!({})).await;
                }
                ("initialized", None) => {}
                (_other, Some(id)) => {
                    send_error(&writer, id, -32601, "method not supported").await;
                }
                (_other, None) => {}
            }
        }
    }
}

type Writer = Arc<Mutex<tokio::io::WriteHalf<tokio::io::DuplexStream>>>;

async fn send_notification(writer: &Writer, method: &str, params: serde_json::Value) {
    let msg = serde_json::json!({ "method": method, "params": params });
    let line = format!("{}\n", serde_json::to_string(&msg).expect("json"));
    let mut w = writer.lock().await;
    let _ = w.write_all(line.as_bytes()).await;
    let _ = w.flush().await;
}

async fn send_response(writer: &Writer, id: serde_json::Value, result: serde_json::Value) {
    let msg = serde_json::json!({ "id": id, "result": result });
    let line = format!("{}\n", serde_json::to_string(&msg).expect("json"));
    let mut w = writer.lock().await;
    let _ = w.write_all(line.as_bytes()).await;
    let _ = w.flush().await;
}

async fn send_error(writer: &Writer, id: serde_json::Value, code: i64, message: &str) {
    let msg = serde_json::json!({ "id": id, "error": { "code": code, "message": message }});
    let line = format!("{}\n", serde_json::to_string(&msg).expect("json"));
    let mut w = writer.lock().await;
    let _ = w.write_all(line.as_bytes()).await;
    let _ = w.flush().await;
}

struct InProcCodexProcess {
    codex: CodexAppServerProcess,
    server_state: Arc<FakeCodexState>,
    client_stream: Mutex<Option<tokio::io::DuplexStream>>,
    server_join: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl InProcCodexProcess {
    fn new(cwd: PathBuf, counters: Arc<FakeCodexCounters>) -> Self {
        let mut config = CodexAppServerProcessConfig::codex_default(cwd);
        config.wire_format = JsonRpcWireFormat::JsonLines;
        config.argv = vec!["fake-codex".to_owned()];

        Self {
            codex: CodexAppServerProcess::new(config),
            server_state: FakeCodexState::new(counters),
            client_stream: Mutex::new(None),
            server_join: Mutex::new(None),
        }
    }
}

impl AppServerProcess for InProcCodexProcess {
    fn start(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async move {
            let mut join_guard = self.server_join.lock().await;
            if join_guard.is_some() {
                return Ok(());
            }

            let (client, server) = tokio::io::duplex(64 * 1024);
            *self.client_stream.lock().await = Some(client);

            let server_state = Arc::clone(&self.server_state);
            let join = tokio::spawn(async move {
                fake_codex_server(server, server_state).await;
            });
            *join_guard = Some(join);

            Ok(())
        })
    }

    fn connect(&self) -> BoxFuture<'_, Result<AppServerConnection, AppServerProcessError>> {
        Box::pin(async move {
            let client = self.client_stream.lock().await.take().ok_or_else(|| {
                AppServerProcessError::ConnectFailed {
                    reason: "client stream already taken".to_owned(),
                }
            })?;
            let (read, write) = tokio::io::split(client);
            self.codex
                .connect_stream(Box::new(read), Box::new(write))
                .await
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async move {
            if let Some(join) = self.server_join.lock().await.take() {
                join.abort();
            }
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

async fn collect_until_turn_completed(
    rx: &mut mpsc::Receiver<redesmyn_protocol::daemon::DaemonFrame>,
    session_id: redesmyn_ids::SessionId,
    timeout: Duration,
) -> Vec<SessionEvent> {
    let mut out = Vec::new();
    loop {
        let records = collect_session_records(rx, session_id, timeout).await;
        let done = records
            .iter()
            .any(|e| matches!(e.kind, SessionEventKind::TurnCompleted(_)));
        out.extend(records);
        if done {
            return out;
        }
    }
}

async fn collect_until_turn_started(
    rx: &mut mpsc::Receiver<redesmyn_protocol::daemon::DaemonFrame>,
    session_id: redesmyn_ids::SessionId,
    timeout: Duration,
) -> Vec<SessionEvent> {
    let mut out = Vec::new();
    loop {
        let records = collect_session_records(rx, session_id, timeout).await;
        let started = records
            .iter()
            .any(|e| matches!(e.kind, SessionEventKind::TurnStarted(_)));
        out.extend(records);
        if started {
            return out;
        }
    }
}

fn find_latest_codex_thread(events: &[SessionEvent]) -> Option<(String, Option<String>)> {
    events.iter().rev().find_map(|ev| match &ev.kind {
        SessionEventKind::TurnStarted(ts) => match ts.external_session_ref.as_ref()? {
            ExternalSessionRef::CodexThread { thread_id, turn_id } => {
                Some((thread_id.clone(), turn_id.clone()))
            }
            _ => None,
        },
        _ => None,
    })
}

#[tokio::test]
async fn captures_thread_id_and_emits_message_events() {
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

    let counters = Arc::new(FakeCodexCounters::default());
    let process = Arc::new(InProcCodexProcess::new(tmp.path().to_path_buf(), counters));

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let session_id = supervisor
        .start_session(
            task_id,
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process,
            },
        )
        .await
        .expect("start_session");

    let _ = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::StartNew {
                prompt: "hi".to_owned(),
            },
        )
        .await
        .expect("send_message");

    let events =
        collect_until_turn_completed(&mut frames_rx, session_id, Duration::from_secs(5)).await;

    let (thread_id, turn_id) = find_latest_codex_thread(&events).expect("thread id");
    assert_eq!(thread_id, "thr_test");
    assert!(turn_id.is_some(), "expected turn id to be present");

    assert!(
        events.iter().any(
            |e| matches!(&e.kind, SessionEventKind::AssistantMessage(m) if m.text == "echo: hi")
        )
    );

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}

#[tokio::test]
async fn resumes_existing_thread_with_external_session_ref() {
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

    let counters = Arc::new(FakeCodexCounters::default());
    let process = Arc::new(InProcCodexProcess::new(
        tmp.path().to_path_buf(),
        Arc::clone(&counters),
    ));

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let session_id = supervisor
        .start_session(
            task_id,
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process,
            },
        )
        .await
        .expect("start_session");

    let _ = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::StartNew {
                prompt: "first".to_owned(),
            },
        )
        .await
        .expect("send_message");

    let first_events =
        collect_until_turn_completed(&mut frames_rx, session_id, Duration::from_secs(5)).await;
    let (thread_id, _turn_id) = find_latest_codex_thread(&first_events).expect("thread id");

    let _ = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::Resume {
                external: DomainExternalSessionRef::CodexThread {
                    thread_id: thread_id.clone(),
                    turn_id: None,
                },
                prompt: "again".to_owned(),
            },
        )
        .await
        .expect("send_message");

    let second_events =
        collect_until_turn_completed(&mut frames_rx, session_id, Duration::from_secs(5)).await;
    assert!(second_events.iter().any(|e| matches!(
        &e.kind,
        SessionEventKind::AssistantMessage(m) if m.text == "echo: again"
    )));

    assert_eq!(counters.thread_start_calls.load(Ordering::Relaxed), 1);
    assert_eq!(counters.thread_resume_calls.load(Ordering::Relaxed), 1);

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}

#[tokio::test]
async fn interrupt_requests_turn_interrupt_and_emits_turn_completed() {
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

    let counters = Arc::new(FakeCodexCounters::default());
    let process = Arc::new(InProcCodexProcess::new(
        tmp.path().to_path_buf(),
        Arc::clone(&counters),
    ));

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let session_id = supervisor
        .start_session(
            task_id,
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process,
            },
        )
        .await
        .expect("start_session");

    let _ = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::StartNew {
                prompt: "block please".to_owned(),
            },
        )
        .await
        .expect("send_message");

    // Wait for turn started so the process has an active turn id.
    let _events =
        collect_until_turn_started(&mut frames_rx, session_id, Duration::from_secs(5)).await;

    // The fake server completes the blocked turn only on interrupt; trigger it.
    let _ = supervisor
        .interrupt_session(session_id)
        .await
        .expect("interrupt_session");

    let completed =
        collect_until_turn_completed(&mut frames_rx, session_id, Duration::from_secs(5)).await;
    assert!(
        completed
            .iter()
            .any(|e| matches!(&e.kind, SessionEventKind::TurnCompleted(tc) if tc.error.is_none()))
    );

    assert_eq!(counters.turn_interrupt_calls.load(Ordering::Relaxed), 1);

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}
