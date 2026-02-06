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
use redesmyn_exec::codex_app_server::{CodexAppServerProcess, CodexAppServerProcessConfig};
use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::daemon::DaemonMessage;
use redesmyn_protocol::session::{
    ExternalSessionRef, SessionEvent, SessionEventKind, SessionScope,
};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::sync::{Mutex, mpsc};

type Writer = Arc<Mutex<tokio::io::WriteHalf<tokio::io::DuplexStream>>>;

async fn send_frame(writer: &Writer, value: serde_json::Value, chunk_size: usize) {
    let mut bytes = serde_json::to_vec(&value).expect("json");
    bytes.push(b'\n');

    let mut w = writer.lock().await;
    if chunk_size == 0 {
        let _ = w.write_all(&bytes).await;
    } else {
        for chunk in bytes.chunks(chunk_size) {
            let _ = w.write_all(chunk).await;
        }
    }
    let _ = w.flush().await;
}

async fn send_notification(writer: &Writer, method: &str, params: serde_json::Value) {
    send_frame(
        writer,
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }),
        3, // force chunked writes for framing robustness
    )
    .await;
}

async fn send_response(writer: &Writer, id: serde_json::Value, result: serde_json::Value) {
    send_frame(
        writer,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }),
        0,
    )
    .await;
}

async fn send_error(writer: &Writer, id: serde_json::Value, code: i64, message: &str) {
    send_frame(
        writer,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": code, "message": message },
        }),
        0,
    )
    .await;
}

#[derive(Default)]
struct FakeV2Counters {
    initialize_calls: AtomicUsize,
    thread_start_calls: AtomicUsize,
    thread_resume_calls: AtomicUsize,
    turn_start_calls: AtomicUsize,
    turn_interrupt_calls: AtomicUsize,
}

struct FakeV2State {
    counters: Arc<FakeV2Counters>,
    next_turn: AtomicUsize,
    inflight: Mutex<Option<InflightTurn>>,
    observed_turn_start_thread_ids: Mutex<Vec<String>>,
}

struct InflightTurn {
    cancel: Arc<tokio::sync::Notify>,
}

impl FakeV2State {
    fn new(counters: Arc<FakeV2Counters>) -> Arc<Self> {
        Arc::new(Self {
            counters,
            next_turn: AtomicUsize::new(0),
            inflight: Mutex::new(None),
            observed_turn_start_thread_ids: Mutex::new(Vec::new()),
        })
    }

    fn alloc_turn_id(&self) -> String {
        let n = self.next_turn.fetch_add(1, Ordering::Relaxed);
        format!("turn_{n}")
    }
}

async fn fake_v2_app_server(stream: tokio::io::DuplexStream, state: Arc<FakeV2State>) {
    let thread_id: &str = "session_test";

    let (reader, writer) = tokio::io::split(stream);
    let writer: Writer = Arc::new(Mutex::new(writer));

    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let read = match reader.read_line(&mut line).await {
            Ok(0) => return,
            Ok(n) => n,
            Err(_) => return,
        };

        if read == 0 {
            continue;
        }

        let frame = line.trim_end_matches(|c| c == '\n' || c == '\r');
        if frame.is_empty() {
            continue;
        }

        let msg: serde_json::Value = serde_json::from_str(frame).unwrap_or(serde_json::Value::Null);

        let Some(method) = msg.get("method").and_then(|m| m.as_str()) else {
            continue;
        };
        let id = msg.get("id").cloned();
        let params = msg
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        match method {
            "initialize" => {
                let Some(id) = id else { continue };
                state
                    .counters
                    .initialize_calls
                    .fetch_add(1, Ordering::Relaxed);
                send_response(&writer, id, serde_json::json!({ "ok": true })).await;
            }
            "thread/start" => {
                let Some(id) = id else { continue };
                state
                    .counters
                    .thread_start_calls
                    .fetch_add(1, Ordering::Relaxed);
                send_response(
                    &writer,
                    id,
                    serde_json::json!({ "thread": { "id": thread_id } }),
                )
                .await;
            }
            "thread/resume" => {
                let Some(id) = id else { continue };
                state
                    .counters
                    .thread_resume_calls
                    .fetch_add(1, Ordering::Relaxed);

                let resolved_thread_id = params
                    .get("threadId")
                    .and_then(|v| v.as_str())
                    .unwrap_or(thread_id);

                send_response(
                    &writer,
                    id,
                    serde_json::json!({ "thread": { "id": resolved_thread_id } }),
                )
                .await;
            }
            "turn/start" => {
                let Some(id) = id else { continue };
                state
                    .counters
                    .turn_start_calls
                    .fetch_add(1, Ordering::Relaxed);

                let request_thread_id = params
                    .get("threadId")
                    .and_then(|v| v.as_str())
                    .unwrap_or(thread_id)
                    .to_owned();

                state
                    .observed_turn_start_thread_ids
                    .lock()
                    .await
                    .push(request_thread_id.clone());

                let prompt = params
                    .get("input")
                    .and_then(|v| v.as_array())
                    .and_then(|items| {
                        items.iter().find_map(|item| {
                            if item.get("type")?.as_str()? != "text" {
                                return None;
                            }
                            item.get("text")?.as_str().map(ToOwned::to_owned)
                        })
                    })
                    .unwrap_or_default();

                let turn_id = state.alloc_turn_id();
                send_response(
                    &writer,
                    id,
                    serde_json::json!({ "turn": { "id": turn_id } }),
                )
                .await;

                let cancel = Arc::new(tokio::sync::Notify::new());

                {
                    let mut inflight = state.inflight.lock().await;
                    *inflight = Some(InflightTurn {
                        cancel: Arc::clone(&cancel),
                    });
                }

                let writer_for_turn = Arc::clone(&writer);
                let state_for_turn = Arc::clone(&state);
                tokio::spawn(async move {
                    let user_message_id = format!("user_{turn_id}");
                    let assistant_message_id = format!("assistant_{turn_id}");

                    send_notification(
                        &writer_for_turn,
                        "turn/started",
                        serde_json::json!({
                            "threadId": request_thread_id.clone(),
                            "turn": { "id": turn_id.clone(), "status": "inProgress", "items": [] },
                        }),
                    )
                    .await;

                    send_notification(
                        &writer_for_turn,
                        "item/started",
                        serde_json::json!({
                            "threadId": request_thread_id.clone(),
                            "turnId": turn_id.clone(),
                            "item": {
                                "type": "userMessage",
                                "id": user_message_id,
                                "content": [
                                    { "type": "text", "text": prompt.clone() }
                                ]
                            }
                        }),
                    )
                    .await;

                    if prompt.contains("block") {
                        cancel.notified().await;
                        send_notification(
                            &writer_for_turn,
                            "turn/completed",
                            serde_json::json!({
                                "threadId": request_thread_id.clone(),
                                "turn": { "id": turn_id.clone(), "status": "interrupted", "error": null }
                            }),
                        )
                        .await;
                    } else {
                        send_notification(
                            &writer_for_turn,
                            "item/completed",
                            serde_json::json!({
                                "threadId": request_thread_id.clone(),
                                "turnId": turn_id.clone(),
                                "item": {
                                    "type": "agentMessage",
                                    "id": assistant_message_id,
                                    "text": format!("echo: {prompt}")
                                }
                            }),
                        )
                        .await;
                        send_notification(
                            &writer_for_turn,
                            "turn/completed",
                            serde_json::json!({
                                "threadId": request_thread_id,
                                "turn": { "id": turn_id, "status": "completed", "error": null }
                            }),
                        )
                        .await;
                    }

                    let mut inflight = state_for_turn.inflight.lock().await;
                    *inflight = None;
                });
            }
            "turn/interrupt" => {
                state
                    .counters
                    .turn_interrupt_calls
                    .fetch_add(1, Ordering::Relaxed);

                if let Some(id) = id {
                    send_response(&writer, id, serde_json::json!({})).await;
                }

                let inflight = state.inflight.lock().await;
                if let Some(turn) = inflight.as_ref() {
                    turn.cancel.notify_waiters();
                }
            }
            "commandExecutionApproval" => {
                let Some(id) = id else { continue };
                send_response(&writer, id, serde_json::json!({})).await;
            }
            "fileChangeApproval" => {
                let Some(id) = id else { continue };
                send_response(&writer, id, serde_json::json!({})).await;
            }
            "initialized" | "exit" => {}
            _ => {
                if let Some(id) = id {
                    send_error(&writer, id, -32601, "method not supported").await;
                }
            }
        }
    }
}

struct InProcV2CodexProcess {
    codex: CodexAppServerProcess,
    server_state: Arc<FakeV2State>,
    client_stream: Mutex<Option<tokio::io::DuplexStream>>,
    server_join: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl InProcV2CodexProcess {
    fn new(cwd: PathBuf, counters: Arc<FakeV2Counters>) -> Self {
        let mut config = CodexAppServerProcessConfig::codex_default(cwd);
        config.argv = vec!["fake-codex".to_owned()];

        Self {
            codex: CodexAppServerProcess::new(config),
            server_state: FakeV2State::new(counters),
            client_stream: Mutex::new(None),
            server_join: Mutex::new(None),
        }
    }
}

impl AppServerProcess for InProcV2CodexProcess {
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
                fake_v2_app_server(server, server_state).await;
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

fn find_latest_codex_session(events: &[SessionEvent]) -> Option<(String, Option<String>)> {
    events.iter().rev().find_map(|ev| match &ev.kind {
        SessionEventKind::TurnStarted(ts) => match ts.external_session_ref.as_ref()? {
            ExternalSessionRef::CodexThread { thread_id, turn_id } => {
                Some((thread_id.clone(), turn_id.clone()))
            }
            ExternalSessionRef::CodexSession {
                session_id,
                turn_id,
            } => Some((session_id.clone(), turn_id.clone())),
            _ => None,
        },
        _ => None,
    })
}

#[tokio::test]
async fn initialize_new_session_user_message_emits_codex_session_ref() {
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

    let counters = Arc::new(FakeV2Counters::default());
    let process = Arc::new(InProcV2CodexProcess::new(
        tmp.path().to_path_buf(),
        counters,
    ));

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let daemon_session_id = SessionId::new();
    let daemon_session_id = supervisor
        .start_session(
            daemon_session_id,
            Some(task_id),
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
            daemon_session_id,
            AppServerTurnIntent::StartNew {
                prompt: "hi".to_owned(),
            },
        )
        .await
        .expect("send_message");

    let events =
        collect_until_turn_completed(&mut frames_rx, daemon_session_id, Duration::from_secs(5))
            .await;

    let (codex_session_id, turn_id) = find_latest_codex_session(&events).expect("session id");
    assert_eq!(codex_session_id, "session_test");
    assert!(turn_id.is_some(), "expected turn id to be present");

    assert!(events.iter().any(|e| matches!(
        &e.kind,
        SessionEventKind::AssistantMessage(m) if m.text == "echo: hi"
    )));

    supervisor
        .stop_session(daemon_session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}

#[tokio::test]
async fn resume_reuses_conversation_and_skips_reinitialize() {
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

    let counters = Arc::new(FakeV2Counters::default());
    let process = Arc::new(InProcV2CodexProcess::new(
        tmp.path().to_path_buf(),
        Arc::clone(&counters),
    ));
    let server_state = Arc::clone(&process.server_state);

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let daemon_session_id = SessionId::new();
    let daemon_session_id = supervisor
        .start_session(
            daemon_session_id,
            Some(task_id),
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
            daemon_session_id,
            AppServerTurnIntent::StartNew {
                prompt: "first".to_owned(),
            },
        )
        .await
        .expect("send_message");

    let first_events =
        collect_until_turn_completed(&mut frames_rx, daemon_session_id, Duration::from_secs(5))
            .await;
    let (codex_session_id, turn_id) = find_latest_codex_session(&first_events).expect("session id");

    let _ = supervisor
        .send_message(
            daemon_session_id,
            AppServerTurnIntent::Resume {
                external: DomainExternalSessionRef::CodexThread {
                    thread_id: codex_session_id.clone(),
                    turn_id,
                },
                prompt: "again".to_owned(),
            },
        )
        .await
        .expect("send_message");

    let second_events =
        collect_until_turn_completed(&mut frames_rx, daemon_session_id, Duration::from_secs(5))
            .await;
    assert!(second_events.iter().any(|e| matches!(
        &e.kind,
        SessionEventKind::AssistantMessage(m) if m.text == "echo: again"
    )));

    let threads = server_state
        .observed_turn_start_thread_ids
        .lock()
        .await
        .clone();
    assert_eq!(threads.len(), 2);
    assert_eq!(threads[0], codex_session_id);
    assert_eq!(threads[1], codex_session_id);

    assert_eq!(counters.thread_start_calls.load(Ordering::Relaxed), 1);
    assert_eq!(counters.thread_resume_calls.load(Ordering::Relaxed), 0);
    assert_eq!(counters.turn_start_calls.load(Ordering::Relaxed), 2);

    supervisor
        .stop_session(daemon_session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}

#[tokio::test]
async fn cancel_mid_turn_emits_turn_completed() {
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

    let counters = Arc::new(FakeV2Counters::default());
    let process = Arc::new(InProcV2CodexProcess::new(
        tmp.path().to_path_buf(),
        Arc::clone(&counters),
    ));

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

    // Wait for turn started so the runner has an active turn id.
    let _events =
        collect_until_turn_started(&mut frames_rx, session_id, Duration::from_secs(5)).await;

    let _ = supervisor
        .interrupt_session(session_id)
        .await
        .expect("interrupt_session");

    let completed =
        collect_until_turn_completed(&mut frames_rx, session_id, Duration::from_secs(5)).await;
    assert!(
        completed
            .iter()
            .any(|e| matches!(&e.kind, SessionEventKind::TurnCompleted(_)))
    );

    assert_eq!(counters.turn_interrupt_calls.load(Ordering::Relaxed), 1);

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}
