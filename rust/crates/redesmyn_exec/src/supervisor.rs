use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use redesmyn_ids::{SessionEventId, SessionId, TaskId};
use redesmyn_logging::tracing;
use redesmyn_protocol::daemon::{DaemonFrame, DaemonMessage, SessionEventBatch};
use redesmyn_protocol::session::{
    ArtifactEmitted, InterfaceMode, SessionEnded, SessionEvent, SessionEventKind, SessionScope,
    SessionStarted, StatusUpdate, TurnCompleted, TurnStarted, TurnState,
};
use redesmyn_protocol::{ProtocolEnvelope, Timestamp};
use tokio::io::AsyncReadExt as _;
use tokio::sync::{Mutex, mpsc};

use crate::artifact_store::{ArtifactStoreError, LocalArtifactStore};
use crate::parser::{OutputStream, SessionOutputParser};

#[derive(Debug, Clone)]
pub struct ExecSessionSupervisorConfig {
    pub output_read_chunk_bytes: usize,
    pub parser_input_capacity: usize,
    pub max_message_chars: usize,
    pub max_preview_chars: usize,
    pub emit_log_artifact_threshold_bytes: u64,
    pub stop_grace_period: std::time::Duration,
}

impl Default for ExecSessionSupervisorConfig {
    fn default() -> Self {
        Self {
            output_read_chunk_bytes: 8 * 1024,
            parser_input_capacity: 128,
            max_message_chars: 4_000,
            max_preview_chars: 240,
            emit_log_artifact_threshold_bytes: 0,
            stop_grace_period: std::time::Duration::from_secs(2),
        }
    }
}

pub struct ExecSessionSpec {
    pub scope: SessionScope,
    pub interface_mode: InterfaceMode,
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
    pub parser: Option<Box<dyn SessionOutputParser>>,
    pub allow_concurrent_for_task: bool,
}

impl std::fmt::Debug for ExecSessionSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecSessionSpec")
            .field("scope", &self.scope)
            .field("interface_mode", &self.interface_mode)
            .field("argv_len", &self.argv.len())
            .field("env_len", &self.env.len())
            .field("parser_present", &self.parser.is_some())
            .field("allow_concurrent_for_task", &self.allow_concurrent_for_task)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StartSessionError {
    #[error("exec argv is empty")]
    EmptyArgv,
    #[error("invalid exec session scope: {reason}")]
    InvalidScope { reason: String },
    #[error("task already has an active {interface_mode:?} session: {session_id}")]
    TaskHasActiveSession {
        session_id: SessionId,
        interface_mode: InterfaceMode,
    },
    #[error("supervisor is shutting down")]
    ShuttingDown,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Artifacts(#[from] ArtifactStoreError),
}

#[derive(Debug, thiserror::Error)]
pub enum SessionControlError {
    #[error("unknown session: {session_id}")]
    UnknownSession { session_id: SessionId },
    #[error("session control channel closed: {session_id}")]
    SessionClosed { session_id: SessionId },
}

#[derive(Debug)]
pub struct ExecSessionSupervisor {
    config: ExecSessionSupervisorConfig,
    artifact_store: LocalArtifactStore,
    frames_tx: mpsc::Sender<DaemonFrame>,
    state: Arc<Mutex<SupervisorState>>,
    cleanup_tx: mpsc::Sender<SessionId>,
}

#[derive(Debug, Default)]
struct SupervisorState {
    is_shutting_down: bool,
    sessions: HashMap<SessionId, SessionEntry>,
    active_by_task: HashMap<(TaskId, InterfaceMode), HashSet<SessionId>>,
}

#[derive(Debug)]
struct SessionEntry {
    task_id: TaskId,
    interface_mode: InterfaceMode,
    control_tx: mpsc::Sender<SessionCommand>,
    join: tokio::task::JoinHandle<()>,
}

#[derive(Debug)]
enum SessionCommand {
    Interrupt,
    Stop,
}

#[derive(Debug)]
struct OutputChunk {
    stream: OutputStream,
    bytes: Vec<u8>,
}

impl ExecSessionSupervisor {
    pub async fn new(
        config: ExecSessionSupervisorConfig,
        artifact_store: LocalArtifactStore,
        frames_tx: mpsc::Sender<DaemonFrame>,
    ) -> Result<Self, StartSessionError> {
        artifact_store.ensure_dirs().await?;
        let (cleanup_tx, mut cleanup_rx) = mpsc::channel::<SessionId>(256);

        let state = Arc::new(Mutex::new(SupervisorState::default()));
        let state_for_cleanup = Arc::clone(&state);
        tokio::spawn(async move {
            while let Some(session_id) = cleanup_rx.recv().await {
                let mut st = state_for_cleanup.lock().await;
                if let Some(entry) = st.sessions.remove(&session_id) {
                    let key = (entry.task_id, entry.interface_mode);
                    if let Some(active) = st.active_by_task.get_mut(&key) {
                        active.remove(&session_id);
                        if active.is_empty() {
                            st.active_by_task.remove(&key);
                        }
                    }
                }
            }
        });

        Ok(Self {
            config,
            artifact_store,
            frames_tx,
            state,
            cleanup_tx,
        })
    }

    pub async fn start_session(
        &self,
        task_id: TaskId,
        spec: ExecSessionSpec,
    ) -> Result<SessionId, StartSessionError> {
        if spec.argv.is_empty() {
            return Err(StartSessionError::EmptyArgv);
        }

        validate_task_scope(task_id, spec.scope)?;

        let session_id = SessionId::new();

        let mut st = self.state.lock().await;
        if st.is_shutting_down {
            return Err(StartSessionError::ShuttingDown);
        }

        if !spec.allow_concurrent_for_task {
            if let Some(existing) = st
                .active_by_task
                .get(&(task_id, spec.interface_mode))
                .and_then(|active| active.iter().next())
            {
                return Err(StartSessionError::TaskHasActiveSession {
                    session_id: *existing,
                    interface_mode: spec.interface_mode,
                });
            }
        }

        let interface_mode = spec.interface_mode;
        let (control_tx, control_rx) = mpsc::channel::<SessionCommand>(8);

        let config = self.config.clone();
        let artifact_store = self.artifact_store.clone();
        let frames_tx = self.frames_tx.clone();
        let cleanup_tx = self.cleanup_tx.clone();

        let join = tokio::spawn(async move {
            let span = redesmyn_logging::redesmyn_info_span!(
                "exec.session",
                session_id = %session_id,
                interface_mode = ?spec.interface_mode,
            );
            redesmyn_logging::span::record_task_id(&span, task_id);

            let _guard = span.enter();
            if let Err(err) = run_session(
                config,
                artifact_store,
                frames_tx,
                task_id,
                session_id,
                spec,
                control_rx,
            )
            .await
            {
                tracing::error!(error = %err, "exec session crashed");
            }

            let _ = cleanup_tx.send(session_id).await;
        });

        st.sessions.insert(
            session_id,
            SessionEntry {
                task_id,
                interface_mode,
                control_tx: control_tx.clone(),
                join,
            },
        );

        st.active_by_task
            .entry((task_id, interface_mode))
            .or_default()
            .insert(session_id);

        Ok(session_id)
    }

    pub async fn interrupt_session(
        &self,
        session_id: SessionId,
    ) -> Result<(), SessionControlError> {
        self.send_command(session_id, SessionCommand::Interrupt)
            .await
    }

    pub async fn stop_session(&self, session_id: SessionId) -> Result<(), SessionControlError> {
        self.send_command(session_id, SessionCommand::Stop).await
    }

    pub async fn shutdown(&self) {
        let sessions = {
            let mut st = self.state.lock().await;
            st.is_shutting_down = true;
            st.active_by_task.clear();
            std::mem::take(&mut st.sessions)
        };

        for (session_id, entry) in &sessions {
            let _ = entry.control_tx.send(SessionCommand::Stop).await;
            tracing::info!(%session_id, "shutdown requested stop");
        }

        for (_session_id, entry) in sessions {
            let _ = entry.join.await;
        }
    }

    async fn send_command(
        &self,
        session_id: SessionId,
        cmd: SessionCommand,
    ) -> Result<(), SessionControlError> {
        let tx = {
            let st = self.state.lock().await;
            let entry = st
                .sessions
                .get(&session_id)
                .ok_or(SessionControlError::UnknownSession { session_id })?;
            entry.control_tx.clone()
        };

        tx.send(cmd)
            .await
            .map_err(|_| SessionControlError::SessionClosed { session_id })
    }
}

fn validate_task_scope(task_id: TaskId, scope: SessionScope) -> Result<(), StartSessionError> {
    match scope {
        SessionScope::Task {
            task_id: scope_task_id,
        } if scope_task_id == task_id => Ok(()),
        SessionScope::Task {
            task_id: scope_task_id,
        } => Err(StartSessionError::InvalidScope {
            reason: format!("scope.task_id must match task_id ({task_id}); got {scope_task_id}"),
        }),
        _ => Err(StartSessionError::InvalidScope {
            reason: "expected task scope".to_owned(),
        }),
    }
}

async fn run_session(
    config: ExecSessionSupervisorConfig,
    artifact_store: LocalArtifactStore,
    frames_tx: mpsc::Sender<DaemonFrame>,
    task_id: TaskId,
    session_id: SessionId,
    spec: ExecSessionSpec,
    mut control_rx: mpsc::Receiver<SessionCommand>,
) -> Result<(), RunSessionError> {
    emit_event(
        &frames_tx,
        session_id,
        spec.scope,
        SessionEventKind::SessionStarted(SessionStarted {}),
    )
    .await?;
    emit_event(
        &frames_tx,
        session_id,
        spec.scope,
        SessionEventKind::TurnStarted(TurnStarted {
            interface_mode: spec.interface_mode,
            external_session_ref: None,
            idempotency_key: None,
            log_offset_bytes: None,
        }),
    )
    .await?;

    let mut cmd = tokio::process::Command::new(&spec.argv[0]);
    if spec.argv.len() > 1 {
        cmd.args(&spec.argv[1..]);
    }
    cmd.current_dir(&spec.cwd);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }

    let mut child = cmd.spawn()?;
    let pid = child.id();

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "missing stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "missing stderr"))?;

    let (parser_tx, parser_rx) = mpsc::channel::<OutputChunk>(config.parser_input_capacity);

    let stdout_writer = artifact_store
        .create_writer(
            redesmyn_protocol::artifacts::ArtifactKind::Log,
            Some("text/plain".to_owned()),
        )
        .await?;
    let stderr_writer = artifact_store
        .create_writer(
            redesmyn_protocol::artifacts::ArtifactKind::Log,
            Some("text/plain".to_owned()),
        )
        .await?;

    let dropped_stdout = Arc::new(AtomicU64::new(0));
    let dropped_stderr = Arc::new(AtomicU64::new(0));

    let stdout_task = tokio::spawn(read_stream(
        OutputStream::Stdout,
        config.output_read_chunk_bytes,
        parser_tx.clone(),
        dropped_stdout.clone(),
        stdout,
        stdout_writer,
    ));

    let stderr_task = tokio::spawn(read_stream(
        OutputStream::Stderr,
        config.output_read_chunk_bytes,
        parser_tx.clone(),
        dropped_stderr.clone(),
        stderr,
        stderr_writer,
    ));

    drop(parser_tx);

    let parser_task = tokio::spawn(run_parser(
        frames_tx.clone(),
        artifact_store.clone(),
        config.clone(),
        session_id,
        spec.scope,
        spec.parser,
        parser_rx,
    ));

    tracing::info!(pid = ?pid, %task_id, "exec session spawned");

    let exit_status = loop {
        tokio::select! {
            status = child.wait() => break status?,
            cmd = control_rx.recv() => {
                match cmd {
                    Some(SessionCommand::Interrupt) => {
                        emit_event(
                            &frames_tx,
                            session_id,
                            spec.scope,
                            SessionEventKind::StatusUpdate(StatusUpdate {
                                turn_state: TurnState::Running,
                                blocking: None,
                                progress_percent: None,
                                message: Some("interrupt_requested".to_owned()),
                            }),
                        )
                        .await?;
                        if let Some(pid) = pid {
                            best_effort_signal(pid, libc::SIGINT)?;
                        }
                    }
                    Some(SessionCommand::Stop) | None => {
                        emit_event(
                            &frames_tx,
                            session_id,
                            spec.scope,
                            SessionEventKind::StatusUpdate(StatusUpdate {
                                turn_state: TurnState::Running,
                                blocking: None,
                                progress_percent: None,
                                message: Some("stop_requested".to_owned()),
                            }),
                        )
                        .await?;

                        if let Some(pid) = pid {
                            best_effort_signal(pid, libc::SIGTERM)?;
                        }

                        match tokio::time::timeout(config.stop_grace_period, child.wait()).await {
                            Ok(status) => break status?,
                            Err(_) => {
                                tracing::warn!("graceful stop timed out; killing");
                                let _ = child.kill().await;
                                break child.wait().await?;
                            }
                        }
                    }
                }
            }
        }
    };

    let stdout_writer = stdout_task.await??;
    let stderr_writer = stderr_task.await??;

    let dropped_stdout_bytes = dropped_stdout.load(Ordering::Relaxed);
    let dropped_stderr_bytes = dropped_stderr.load(Ordering::Relaxed);
    if dropped_stdout_bytes > 0 || dropped_stderr_bytes > 0 {
        emit_event(
            &frames_tx,
            session_id,
            spec.scope,
            SessionEventKind::StatusUpdate(StatusUpdate {
                turn_state: TurnState::Running,
                blocking: None,
                progress_percent: None,
                message: Some(format!(
                    "output_dropped: dropped bytes due to backpressure (stdout={}, stderr={})",
                    dropped_stdout_bytes, dropped_stderr_bytes
                )),
            }),
        )
        .await?;
    }

    let stdout_ref = stdout_writer.finish();
    let stderr_ref = stderr_writer.finish();

    let stdout_len = stdout_ref.byte_len.unwrap_or_default();
    if stdout_len == 0 || stdout_len < config.emit_log_artifact_threshold_bytes {
        artifact_store
            .remove_artifact(stdout_ref.artifact_id)
            .await?;
    } else {
        emit_event(
            &frames_tx,
            session_id,
            spec.scope,
            SessionEventKind::ArtifactEmitted(ArtifactEmitted {
                artifact: stdout_ref.clone(),
                label: Some("stdout".to_owned()),
            }),
        )
        .await?;
    }

    let stderr_len = stderr_ref.byte_len.unwrap_or_default();
    if stderr_len == 0 || stderr_len < config.emit_log_artifact_threshold_bytes {
        artifact_store
            .remove_artifact(stderr_ref.artifact_id)
            .await?;
    } else {
        emit_event(
            &frames_tx,
            session_id,
            spec.scope,
            SessionEventKind::ArtifactEmitted(ArtifactEmitted {
                artifact: stderr_ref.clone(),
                label: Some("stderr".to_owned()),
            }),
        )
        .await?;
    }

    let _ = parser_task.await;

    let exit_code = exit_status.code();
    emit_event(
        &frames_tx,
        session_id,
        spec.scope,
        SessionEventKind::TurnCompleted(TurnCompleted {
            interface_mode: spec.interface_mode,
            external_session_ref: None,
            exit_code,
            error: None,
        }),
    )
    .await?;
    emit_event(
        &frames_tx,
        session_id,
        spec.scope,
        SessionEventKind::SessionEnded(SessionEnded {}),
    )
    .await?;

    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum RunSessionError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Artifacts(#[from] ArtifactStoreError),
    #[error(transparent)]
    Emit(#[from] EmitEventError),
    #[error("task join failed")]
    Join(#[from] tokio::task::JoinError),
}

async fn read_stream(
    stream: OutputStream,
    chunk_bytes: usize,
    parser_tx: mpsc::Sender<OutputChunk>,
    dropped_bytes: Arc<AtomicU64>,
    mut reader: impl tokio::io::AsyncRead + Unpin + Send + 'static,
    mut writer: crate::artifact_store::LocalArtifactWriter,
) -> Result<crate::artifact_store::LocalArtifactWriter, RunSessionError> {
    let mut buf = vec![0_u8; chunk_bytes.max(1)];

    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }

        writer.write_all(&buf[..n]).await?;

        if parser_tx
            .try_send(OutputChunk {
                stream,
                bytes: buf[..n].to_vec(),
            })
            .is_err()
        {
            dropped_bytes.fetch_add(n as u64, Ordering::Relaxed);
        }
    }

    Ok(writer)
}

#[derive(Debug, thiserror::Error)]
enum EmitEventError {
    #[error("control plane session event stream closed")]
    StreamClosed,
}

async fn emit_event(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    session_id: SessionId,
    scope: SessionScope,
    kind: SessionEventKind,
) -> Result<(), EmitEventError> {
    let record = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope,
        session_id,
        turn_id: None,
        kind,
    };
    let batch = SessionEventBatch {
        events: vec![record],
    };
    let frame = DaemonFrame::new(
        ProtocolEnvelope::new(),
        DaemonMessage::SessionEventBatch(batch),
    );

    frames_tx
        .send(frame)
        .await
        .map_err(|_| EmitEventError::StreamClosed)
}

fn best_effort_signal(pid: u32, signal: i32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let pid_i32 = i32::try_from(pid)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "pid out of range"))?;
        let rc = unsafe { libc::kill(pid_i32, signal) };
        if rc == 0 {
            return Ok(());
        }

        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::NotFound {
            return Ok(());
        }
        return Err(err);
    }

    #[cfg(not(unix))]
    {
        let _ = pid;
        let _ = signal;
        Ok(())
    }
}

async fn run_parser(
    frames_tx: mpsc::Sender<DaemonFrame>,
    artifact_store: LocalArtifactStore,
    config: ExecSessionSupervisorConfig,
    session_id: SessionId,
    scope: SessionScope,
    mut parser: Option<Box<dyn SessionOutputParser>>,
    mut rx: mpsc::Receiver<OutputChunk>,
) -> Result<(), RunSessionError> {
    while let Some(chunk) = rx.recv().await {
        let Some(parser) = parser.as_mut() else {
            continue;
        };
        let text = String::from_utf8_lossy(&chunk.bytes);
        let events = parser.push_chunk(chunk.stream, &text);
        for event in events {
            emit_limited_event(
                &frames_tx,
                &artifact_store,
                &config,
                session_id,
                scope,
                event,
            )
            .await?;
        }
    }

    if let Some(parser) = parser.as_mut() {
        for event in parser.flush() {
            emit_limited_event(
                &frames_tx,
                &artifact_store,
                &config,
                session_id,
                scope,
                event,
            )
            .await?;
        }
    }

    Ok(())
}

async fn emit_limited_event(
    frames_tx: &mpsc::Sender<DaemonFrame>,
    artifact_store: &LocalArtifactStore,
    config: &ExecSessionSupervisorConfig,
    session_id: SessionId,
    scope: SessionScope,
    mut kind: SessionEventKind,
) -> Result<(), RunSessionError> {
    match &mut kind {
        SessionEventKind::UserMessage(ev) => {
            limit_message_event(
                artifact_store,
                config,
                &mut ev.text,
                &mut ev.preview,
                &mut ev.full_text_artifact,
            )
            .await?;
        }
        SessionEventKind::AssistantMessage(ev) => {
            limit_message_event(
                artifact_store,
                config,
                &mut ev.text,
                &mut ev.preview,
                &mut ev.full_text_artifact,
            )
            .await?;
        }
        _ => {}
    }

    emit_event(frames_tx, session_id, scope, kind).await?;
    Ok(())
}

async fn limit_message_event(
    artifact_store: &LocalArtifactStore,
    config: &ExecSessionSupervisorConfig,
    text: &mut String,
    preview: &mut String,
    full_text_artifact: &mut Option<redesmyn_protocol::artifacts::ArtifactRef>,
) -> Result<(), RunSessionError> {
    let normalized_preview = normalize_preview(text);
    *preview = truncate_chars(&normalized_preview, config.max_preview_chars);

    if text.chars().count() <= config.max_message_chars {
        return Ok(());
    }

    let artifact = artifact_store
        .store_bytes(
            redesmyn_protocol::artifacts::ArtifactKind::Log,
            Some("text/plain".to_owned()),
            text.as_bytes(),
        )
        .await?;
    *full_text_artifact = Some(artifact);
    *text = truncate_chars(text, config.max_message_chars);
    Ok(())
}

fn normalize_preview(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let mut out = String::new();
    for (idx, ch) in text.chars().enumerate() {
        if idx >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}
