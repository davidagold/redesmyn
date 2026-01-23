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
use redesmyn_ids::TaskId;
use redesmyn_protocol::daemon::DaemonMessage;
use redesmyn_protocol::session::{
    ExternalSessionRef, SessionEvent, SessionEventKind, SessionScope,
};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::sync::{Mutex, mpsc};

#[derive(Debug, Default)]
struct ContentLengthDecoder {
    buffer: Vec<u8>,
    expected_len: Option<usize>,
}

impl ContentLengthDecoder {
    fn push(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        self.buffer.extend_from_slice(chunk);
        let mut out = Vec::new();

        loop {
            if self.expected_len.is_none() {
                let Some((header_end, consumed)) = find_header_terminator(&self.buffer) else {
                    break;
                };
                let header = std::str::from_utf8(&self.buffer[..header_end]).expect("ascii header");
                let len = parse_content_length(header);
                self.expected_len = Some(len);
                self.buffer.drain(..consumed);
            }

            let Some(expected) = self.expected_len else {
                break;
            };
            if self.buffer.len() < expected {
                break;
            }

            let payload = self.buffer.drain(..expected).collect::<Vec<u8>>();
            self.expected_len = None;
            out.push(payload);
        }

        out
    }
}

fn find_header_terminator(buf: &[u8]) -> Option<(usize, usize)> {
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some((pos, pos + 4));
    }
    if let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
        return Some((pos, pos + 2));
    }
    None
}

fn parse_content_length(headers: &str) -> usize {
    for line in headers.lines() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        if k.trim().eq_ignore_ascii_case("Content-Length") {
            return v.trim().parse::<usize>().expect("Content-Length int");
        }
    }
    panic!("missing Content-Length header");
}

type Writer = Arc<Mutex<tokio::io::WriteHalf<tokio::io::DuplexStream>>>;

async fn send_frame(writer: &Writer, value: serde_json::Value, chunk_size: usize) {
    let json = serde_json::to_vec(&value).expect("json");
    let header = format!("Content-Length: {}\r\n\r\n", json.len());

    let mut bytes = header.into_bytes();
    bytes.extend_from_slice(&json);

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
    new_conversation_calls: AtomicUsize,
    user_message_calls: AtomicUsize,
    cancel_calls: AtomicUsize,
}

struct FakeV2State {
    counters: Arc<FakeV2Counters>,
    next_turn: AtomicUsize,
    inflight: Mutex<Option<InflightTurn>>,
    observed_user_message_resumes: Mutex<Vec<Option<String>>>,
}

struct InflightTurn {
    cancel: Arc<tokio::sync::Notify>,
}

impl FakeV2State {
    fn new(counters: Arc<FakeV2Counters>) -> Arc<Self> {
        Arc::new(Self {
            counters,
            next_turn: AtomicUsize::new(1),
            inflight: Mutex::new(None),
            observed_user_message_resumes: Mutex::new(Vec::new()),
        })
    }

    fn alloc_turn_id(&self) -> String {
        let n = self.next_turn.fetch_add(1, Ordering::Relaxed);
        format!("turn_{n}")
    }
}

async fn fake_v2_app_server(stream: tokio::io::DuplexStream, state: Arc<FakeV2State>) {
    let conversation_id: &str = "conv_test";

    let (mut reader, writer) = tokio::io::split(stream);
    let writer: Writer = Arc::new(Mutex::new(writer));

    let mut decoder = ContentLengthDecoder::default();
    let mut buf = vec![0u8; 4096];

    loop {
        let read = match reader.read(&mut buf).await {
            Ok(0) => return,
            Ok(n) => n,
            Err(_) => return,
        };

        let frames = decoder.push(&buf[..read]);
        for frame in frames {
            let msg: serde_json::Value =
                serde_json::from_slice(&frame).unwrap_or(serde_json::Value::Null);

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
                    state
                        .counters
                        .initialize_calls
                        .fetch_add(1, Ordering::Relaxed);
                    send_response(&writer, id, serde_json::json!({ "ok": true })).await;
                }
                ("newConversation", Some(id)) => {
                    state
                        .counters
                        .new_conversation_calls
                        .fetch_add(1, Ordering::Relaxed);
                    send_response(
                        &writer,
                        id,
                        serde_json::json!({ "conversationId": conversation_id }),
                    )
                    .await;
                }
                ("userMessage", Some(id)) => {
                    state
                        .counters
                        .user_message_calls
                        .fetch_add(1, Ordering::Relaxed);

                    let prompt = params
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned();
                    let resume = params
                        .get("resume")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned);

                    state
                        .observed_user_message_resumes
                        .lock()
                        .await
                        .push(resume.clone());

                    send_response(&writer, id, serde_json::json!({})).await;

                    let turn_id = state.alloc_turn_id();
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
                            "updateConversation",
                            serde_json::json!({
                                "diff": [
                                    { "type": "newConversation", "conversationId": conversation_id },
                                    { "type": "newTurn", "turnId": turn_id },
                                    { "type": "newMessage", "messageId": user_message_id, "role": "user", "content": prompt, "done": true }
                                ]
                            }),
                        )
                        .await;

                        if prompt.contains("block") {
                            cancel.notified().await;
                            send_notification(
                                &writer_for_turn,
                                "updateConversation",
                                serde_json::json!({
                                    "diff": [
                                        { "type": "turnCompleted", "turnId": turn_id, "status": "canceled" }
                                    ]
                                }),
                            )
                            .await;
                        } else {
                            send_notification(
                                &writer_for_turn,
                                "updateConversation",
                                serde_json::json!({
                                    "diff": [
                                        { "type": "newMessage", "messageId": assistant_message_id, "role": "assistant", "content": format!("echo: {prompt}"), "done": true },
                                        { "type": "turnCompleted", "turnId": turn_id, "status": "completed" }
                                    ]
                                }),
                            )
                            .await;
                        }

                        let mut inflight = state_for_turn.inflight.lock().await;
                        *inflight = None;
                    });
                }
                ("cancel", Some(id)) => {
                    state.counters.cancel_calls.fetch_add(1, Ordering::Relaxed);
                    send_response(&writer, id, serde_json::json!({})).await;

                    let inflight = state.inflight.lock().await;
                    if let Some(turn) = inflight.as_ref() {
                        turn.cancel.notify_waiters();
                    }
                }
                ("commandExecutionApproval", Some(id)) => {
                    send_response(&writer, id, serde_json::json!({})).await;
                }
                ("fileChangeApproval", Some(id)) => {
                    send_response(&writer, id, serde_json::json!({})).await;
                }
                ("initialized", None) => {}
                ("exit", None) => {}
                (_other, Some(id)) => {
                    send_error(&writer, id, -32601, "method not supported").await;
                }
                (_other, None) => {}
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

fn find_latest_codex_conversation(events: &[SessionEvent]) -> Option<(String, Option<String>)> {
    events.iter().rev().find_map(|ev| match &ev.kind {
        SessionEventKind::TurnStarted(ts) => match ts.external_session_ref.as_ref()? {
            ExternalSessionRef::CodexConversation {
                conversation_id,
                turn_id,
            } => Some((conversation_id.clone(), turn_id.clone())),
            _ => None,
        },
        _ => None,
    })
}

#[tokio::test]
async fn initialize_new_conversation_user_message_emits_codex_conversation_ref() {
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

    let (conversation_id, turn_id) =
        find_latest_codex_conversation(&events).expect("conversation id");
    assert_eq!(conversation_id, "conv_test");
    assert!(turn_id.is_some(), "expected turn id to be present");

    assert!(events.iter().any(|e| matches!(
        &e.kind,
        SessionEventKind::AssistantMessage(m) if m.text == "echo: hi"
    )));

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}

#[tokio::test]
async fn resume_sets_user_message_resume_field() {
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
    let (conversation_id, _turn_id) =
        find_latest_codex_conversation(&first_events).expect("conversation id");

    let _ = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::Resume {
                external: DomainExternalSessionRef::CodexConversation {
                    conversation_id: conversation_id.clone(),
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

    let resumes = server_state
        .observed_user_message_resumes
        .lock()
        .await
        .clone();
    assert_eq!(resumes.len(), 2);
    assert_eq!(resumes[0], None);
    assert_eq!(resumes[1], Some(conversation_id));

    assert_eq!(counters.new_conversation_calls.load(Ordering::Relaxed), 1);
    assert_eq!(counters.user_message_calls.load(Ordering::Relaxed), 2);

    supervisor
        .stop_session(session_id)
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

    assert_eq!(counters.cancel_calls.load(Ordering::Relaxed), 1);

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}
