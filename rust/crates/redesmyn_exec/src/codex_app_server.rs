use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use redesmyn_domain::agent::{AppServerTurnIntent, ExternalSessionRef as DomainExternalSessionRef};
use redesmyn_logging::tracing;
use redesmyn_protocol::session::ExternalSessionRef;
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::app_server::{
    AppServerClient, AppServerConnection, AppServerEvent, AppServerProcess, AppServerProcessError,
    AppServerRequest, AppServerRequestError, AppServerResponse, BoxFuture,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonRpcWireFormat {
    /// Codex app-server uses JSONL over stdio (one JSON-RPC message per line).
    JsonLines,
    /// Optional LSP-style `Content-Length` framing.
    ContentLength,
}

impl Default for JsonRpcWireFormat {
    fn default() -> Self {
        Self::JsonLines
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JsonRpcId {
    Number(i64),
    String(String),
}

impl JsonRpcId {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Number(n) => serde_json::Value::Number((*n).into()),
            Self::String(s) => serde_json::Value::String(s.clone()),
        }
    }

    fn try_from_json(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::Number(n) => n.as_i64().map(Self::Number),
            serde_json::Value::String(s) => Some(Self::String(s.clone())),
            _ => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum JsonRpcError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid utf-8")]
    Utf8,
    #[error("invalid json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("jsonrpc error response: {message}")]
    RemoteError { message: String },
}

#[derive(Debug, thiserror::Error)]
enum FramingError {
    #[error("missing Content-Length header")]
    MissingContentLength,
    #[error("invalid Content-Length value")]
    InvalidContentLength,
    #[error("header bytes are not valid ascii")]
    NonAsciiHeader,
}

#[derive(Debug)]
struct JsonRpcFramingDecoder {
    wire_format: JsonRpcWireFormat,
    buffer: Vec<u8>,
    expected_len: Option<usize>,
}

impl JsonRpcFramingDecoder {
    fn new(wire_format: JsonRpcWireFormat) -> Self {
        Self {
            wire_format,
            buffer: Vec::new(),
            expected_len: None,
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Result<Vec<Vec<u8>>, FramingError> {
        self.buffer.extend_from_slice(chunk);
        match self.wire_format {
            JsonRpcWireFormat::JsonLines => Ok(self.drain_json_lines()),
            JsonRpcWireFormat::ContentLength => self.drain_content_length_frames(),
        }
    }

    fn drain_json_lines(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(pos) = self.buffer.iter().position(|b| *b == b'\n') {
            let mut line = self.buffer.drain(..pos + 1).collect::<Vec<u8>>();
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }

            if line.iter().all(|b| b.is_ascii_whitespace()) {
                continue;
            }

            out.push(line);
        }
        out
    }

    fn drain_content_length_frames(&mut self) -> Result<Vec<Vec<u8>>, FramingError> {
        let mut out = Vec::new();
        loop {
            if self.expected_len.is_none() {
                let Some((header_end, consumed)) = find_header_terminator(&self.buffer) else {
                    break;
                };

                let header_bytes = &self.buffer[..header_end];
                let header_str =
                    std::str::from_utf8(header_bytes).map_err(|_| FramingError::NonAsciiHeader)?;
                let len = parse_content_length(header_str)?;
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

        Ok(out)
    }
}

fn find_header_terminator(buf: &[u8]) -> Option<(usize, usize)> {
    // Prefer strict LSP `\r\n\r\n`.
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some((pos, pos + 4));
    }
    // Accept `\n\n` as a fallback.
    if let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
        return Some((pos, pos + 2));
    }
    None
}

fn parse_content_length(headers: &str) -> Result<usize, FramingError> {
    for line in headers.lines() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        if k.trim().eq_ignore_ascii_case("Content-Length") {
            let value = v.trim();
            let len = value
                .parse::<usize>()
                .map_err(|_| FramingError::InvalidContentLength)?;
            return Ok(len);
        }
    }
    Err(FramingError::MissingContentLength)
}

struct JsonRpcConnection {
    wire_format: JsonRpcWireFormat,
    writer: Mutex<Box<dyn AsyncWrite + Unpin + Send>>,
    next_id: AtomicU64,
    pending: Mutex<HashMap<JsonRpcId, oneshot::Sender<Result<serde_json::Value, JsonRpcError>>>>,
}

impl std::fmt::Debug for JsonRpcConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonRpcConnection")
            .field("wire_format", &self.wire_format)
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .field("writer", &"<writer>")
            .field("pending", &"<pending>")
            .finish()
    }
}

impl JsonRpcConnection {
    fn new(wire_format: JsonRpcWireFormat, writer: Box<dyn AsyncWrite + Unpin + Send>) -> Self {
        Self {
            wire_format,
            writer: Mutex::new(writer),
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
        }
    }

    fn next_request_id(&self) -> JsonRpcId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        JsonRpcId::Number(id as i64)
    }

    async fn send_value(&self, value: &serde_json::Value) -> Result<(), JsonRpcError> {
        let json = serde_json::to_vec(value)?;
        let mut writer = self.writer.lock().await;
        match self.wire_format {
            JsonRpcWireFormat::JsonLines => {
                writer.write_all(&json).await?;
                writer.write_all(b"\n").await?;
            }
            JsonRpcWireFormat::ContentLength => {
                let header = format!("Content-Length: {}\r\n\r\n", json.len());
                writer.write_all(header.as_bytes()).await?;
                writer.write_all(&json).await?;
            }
        }
        writer.flush().await?;
        Ok(())
    }

    async fn request(
        &self,
        method: &'static str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, AppServerRequestError> {
        let request_id = self.next_request_id();
        let (tx, rx) = oneshot::channel();

        {
            let mut pending = self.pending.lock().await;
            pending.insert(request_id.clone(), tx);
        }

        let mut payload = serde_json::json!({
            "method": method,
            "id": request_id.to_json(),
        });
        if let Some(params) = params {
            payload["params"] = params;
        }

        if let Err(err) = self.send_value(&payload).await {
            let mut pending = self.pending.lock().await;
            pending.remove(&request_id);
            return Err(AppServerRequestError::Failed {
                reason: format!("send failed: {err}"),
            });
        }

        match rx.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(err)) => Err(AppServerRequestError::Failed {
                reason: format!("request failed: {err}"),
            }),
            Err(_) => Err(AppServerRequestError::Failed {
                reason: "request response channel closed".to_owned(),
            }),
        }
    }

    async fn notify(
        &self,
        method: &'static str,
        params: Option<serde_json::Value>,
    ) -> Result<(), AppServerRequestError> {
        let mut payload = serde_json::json!({ "method": method });
        if let Some(params) = params {
            payload["params"] = params;
        }
        self.send_value(&payload)
            .await
            .map_err(|err| AppServerRequestError::Failed {
                reason: format!("send failed: {err}"),
            })
    }

    async fn deliver_response(
        &self,
        id: JsonRpcId,
        payload: Result<serde_json::Value, JsonRpcError>,
    ) {
        let tx = {
            let mut pending = self.pending.lock().await;
            pending.remove(&id)
        };
        if let Some(tx) = tx {
            let _ = tx.send(payload);
        } else {
            tracing::debug!(?id, "dropping response for unknown request id");
        }
    }
}

#[derive(Debug)]
struct CodexAppServerState {
    thread_id: Mutex<Option<String>>,
    active_turn_id: Mutex<Option<String>>,
}

impl CodexAppServerState {
    fn new() -> Self {
        Self {
            thread_id: Mutex::new(None),
            active_turn_id: Mutex::new(None),
        }
    }

    async fn set_thread_id(&self, thread_id: String) {
        *self.thread_id.lock().await = Some(thread_id);
    }

    async fn set_active_turn_id(&self, turn_id: Option<String>) {
        *self.active_turn_id.lock().await = turn_id;
    }

    async fn thread_id(&self) -> Option<String> {
        self.thread_id.lock().await.clone()
    }

    async fn active_turn_id(&self) -> Option<String> {
        self.active_turn_id.lock().await.clone()
    }
}

#[derive(Debug)]
pub struct CodexAppServerClient {
    conn: Arc<JsonRpcConnection>,
    state: Arc<CodexAppServerState>,
    cwd: PathBuf,
    client_name: String,
    client_title: String,
    client_version: String,
}

impl CodexAppServerClient {
    fn new(
        conn: Arc<JsonRpcConnection>,
        state: Arc<CodexAppServerState>,
        cwd: PathBuf,
        client_name: String,
        client_title: String,
        client_version: String,
    ) -> Self {
        Self {
            conn,
            state,
            cwd,
            client_name,
            client_title,
            client_version,
        }
    }

    async fn initialize_handshake(&self) -> Result<(), AppServerProcessError> {
        let params = serde_json::json!({
            "clientInfo": {
                "name": self.client_name,
                "title": self.client_title,
                "version": self.client_version,
            }
        });

        let _ = self
            .conn
            .request("initialize", Some(params))
            .await
            .map_err(|err| AppServerProcessError::ConnectFailed {
                reason: format!("initialize failed: {err}"),
            })?;

        self.conn.notify("initialized", None).await.map_err(|err| {
            AppServerProcessError::ConnectFailed {
                reason: format!("initialized notify failed: {err}"),
            }
        })?;

        Ok(())
    }

    async fn ensure_thread_for_intent(
        &self,
        intent: &AppServerTurnIntent,
    ) -> Result<String, AppServerRequestError> {
        match intent {
            AppServerTurnIntent::StartNew { .. } => {
                let thread_id = self.start_thread().await?;
                self.state.set_thread_id(thread_id.clone()).await;
                Ok(thread_id)
            }
            AppServerTurnIntent::Resume { external, .. } => {
                let thread_id = match external {
                    DomainExternalSessionRef::CodexThread { thread_id, .. } => thread_id.clone(),
                    DomainExternalSessionRef::None => {
                        return Err(AppServerRequestError::Failed {
                            reason: "cannot resume: ExternalSessionRef::None".to_owned(),
                        });
                    }
                    other => {
                        return Err(AppServerRequestError::Failed {
                            reason: format!(
                                "cannot resume: unsupported external session ref: {other:?}"
                            ),
                        });
                    }
                };

                let _ = self
                    .conn
                    .request(
                        "thread/resume",
                        Some(serde_json::json!({ "threadId": thread_id })),
                    )
                    .await?;

                self.state.set_thread_id(thread_id.clone()).await;
                Ok(thread_id)
            }
        }
    }

    async fn start_thread(&self) -> Result<String, AppServerRequestError> {
        let cwd = self.cwd.to_string_lossy().to_string();
        let result = self
            .conn
            .request("thread/start", Some(serde_json::json!({ "cwd": cwd })))
            .await?;
        parse_thread_id_from_thread_result(&result).ok_or_else(|| AppServerRequestError::Failed {
            reason: "thread/start response missing result.thread.id".to_owned(),
        })
    }

    async fn start_turn(
        &self,
        thread_id: &str,
        prompt: &str,
    ) -> Result<String, AppServerRequestError> {
        let params = serde_json::json!({
            "threadId": thread_id,
            "input": [{
                "type": "text",
                "text": prompt,
                "textElements": [],
            }],
        });

        let result = self.conn.request("turn/start", Some(params)).await?;
        parse_turn_id_from_turn_result(&result).ok_or_else(|| AppServerRequestError::Failed {
            reason: "turn/start response missing result.turn.id".to_owned(),
        })
    }

    async fn interrupt_active_turn(&self) -> Result<(), AppServerRequestError> {
        let thread_id =
            self.state
                .thread_id()
                .await
                .ok_or_else(|| AppServerRequestError::Failed {
                    reason: "turn/interrupt failed: missing thread id".to_owned(),
                })?;

        let turn_id =
            self.state
                .active_turn_id()
                .await
                .ok_or_else(|| AppServerRequestError::Failed {
                    reason: "turn/interrupt failed: missing active turn id".to_owned(),
                })?;

        let params = serde_json::json!({ "threadId": thread_id, "turnId": turn_id });
        let _ = self.conn.request("turn/interrupt", Some(params)).await?;
        Ok(())
    }
}

impl AppServerClient for CodexAppServerClient {
    fn request(
        &self,
        request: AppServerRequest,
    ) -> BoxFuture<'_, Result<AppServerResponse, AppServerRequestError>> {
        Box::pin(async move {
            match request {
                AppServerRequest::SendMessage { intent } => {
                    let thread_id = self.ensure_thread_for_intent(&intent).await?;
                    let prompt = match intent {
                        AppServerTurnIntent::StartNew { prompt } => prompt,
                        AppServerTurnIntent::Resume { prompt, .. } => prompt,
                    };

                    let turn_id = self.start_turn(&thread_id, &prompt).await?;
                    self.state.set_active_turn_id(Some(turn_id)).await;
                    Ok(AppServerResponse::MessageAccepted)
                }
                AppServerRequest::Interrupt => {
                    self.interrupt_active_turn().await?;
                    Ok(AppServerResponse::Interrupted)
                }
            }
        })
    }
}

fn parse_thread_id_from_thread_result(result: &serde_json::Value) -> Option<String> {
    result
        .get("thread")?
        .get("id")?
        .as_str()
        .map(ToOwned::to_owned)
}

fn parse_turn_id_from_turn_result(result: &serde_json::Value) -> Option<String> {
    result
        .get("turn")?
        .get("id")?
        .as_str()
        .map(ToOwned::to_owned)
}

#[derive(Debug)]
pub struct CodexAppServerProcessConfig {
    pub argv: Vec<String>,
    pub cwd: PathBuf,
    pub env: Vec<(String, String)>,
    pub wire_format: JsonRpcWireFormat,
    pub client_name: String,
    pub client_title: String,
    pub client_version: String,
}

impl CodexAppServerProcessConfig {
    #[must_use]
    pub fn codex_default(cwd: PathBuf) -> Self {
        Self {
            argv: vec!["codex".to_owned(), "app-server".to_owned()],
            cwd,
            env: Vec::new(),
            wire_format: JsonRpcWireFormat::default(),
            client_name: "redesmyn".to_owned(),
            client_title: "Redesmyn".to_owned(),
            client_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

pub struct CodexAppServerProcess {
    config: CodexAppServerProcessConfig,
    child: Mutex<Option<tokio::process::Child>>,
}

impl CodexAppServerProcess {
    #[must_use]
    pub fn new(config: CodexAppServerProcessConfig) -> Self {
        Self {
            config,
            child: Mutex::new(None),
        }
    }

    pub async fn connect_stream(
        &self,
        reader: Box<dyn AsyncRead + Unpin + Send>,
        writer: Box<dyn AsyncWrite + Unpin + Send>,
    ) -> Result<AppServerConnection, AppServerProcessError> {
        let state = Arc::new(CodexAppServerState::new());
        let conn = Arc::new(JsonRpcConnection::new(self.config.wire_format, writer));

        let (events_tx, events_rx) = mpsc::channel::<AppServerEvent>(256);
        spawn_reader_loop(
            self.config.wire_format,
            reader,
            Arc::clone(&conn),
            Arc::clone(&state),
            events_tx,
        );

        let client = Arc::new(CodexAppServerClient::new(
            Arc::clone(&conn),
            state,
            self.config.cwd.clone(),
            self.config.client_name.clone(),
            self.config.client_title.clone(),
            self.config.client_version.clone(),
        ));

        client.initialize_handshake().await?;

        Ok(AppServerConnection {
            client,
            events: events_rx,
        })
    }
}

impl AppServerProcess for CodexAppServerProcess {
    fn start(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async move {
            let mut child_guard = self.child.lock().await;
            if child_guard.is_some() {
                return Ok(());
            }

            let argv0 =
                self.config
                    .argv
                    .first()
                    .ok_or_else(|| AppServerProcessError::StartFailed {
                        reason: "argv is empty".to_owned(),
                    })?;

            let mut cmd = Command::new(argv0);
            if self.config.argv.len() > 1 {
                cmd.args(&self.config.argv[1..]);
            }
            cmd.current_dir(&self.config.cwd);
            cmd.stdin(std::process::Stdio::piped());
            cmd.stdout(std::process::Stdio::piped());
            cmd.stderr(std::process::Stdio::inherit());
            for (k, v) in &self.config.env {
                cmd.env(k, v);
            }

            let child = cmd
                .spawn()
                .map_err(|err| AppServerProcessError::StartFailed {
                    reason: err.to_string(),
                })?;

            *child_guard = Some(child);
            Ok(())
        })
    }

    fn connect(&self) -> BoxFuture<'_, Result<AppServerConnection, AppServerProcessError>> {
        Box::pin(async move {
            let mut child_guard = self.child.lock().await;
            let child =
                child_guard
                    .as_mut()
                    .ok_or_else(|| AppServerProcessError::ConnectFailed {
                        reason: "process not started".to_owned(),
                    })?;

            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| AppServerProcessError::ConnectFailed {
                    reason: "codex app-server stdin unavailable".to_owned(),
                })?;
            let stdout =
                child
                    .stdout
                    .take()
                    .ok_or_else(|| AppServerProcessError::ConnectFailed {
                        reason: "codex app-server stdout unavailable".to_owned(),
                    })?;

            self.connect_stream(Box::new(stdout), Box::new(stdin)).await
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async move {
            let mut child_guard = self.child.lock().await;
            let mut child = child_guard.take();
            let Some(child) = child.as_mut() else {
                return Ok(());
            };

            if let Err(err) = child.kill().await {
                return Err(AppServerProcessError::ShutdownFailed {
                    reason: err.to_string(),
                });
            }
            let _ = child.wait().await;
            Ok(())
        })
    }
}

fn spawn_reader_loop(
    wire_format: JsonRpcWireFormat,
    mut reader: Box<dyn AsyncRead + Unpin + Send>,
    conn: Arc<JsonRpcConnection>,
    state: Arc<CodexAppServerState>,
    events_tx: mpsc::Sender<AppServerEvent>,
) {
    tokio::spawn(async move {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.reader");
        let _guard = span.enter();

        let mut decoder = JsonRpcFramingDecoder::new(wire_format);
        let mut buf = vec![0u8; 8 * 1024];

        loop {
            let read = match reader.read(&mut buf).await {
                Ok(0) => return,
                Ok(n) => n,
                Err(err) => {
                    tracing::warn!(error = %err, "codex app-server read failed");
                    return;
                }
            };

            let frames = match decoder.push(&buf[..read]) {
                Ok(frames) => frames,
                Err(err) => {
                    tracing::warn!(error = %err, "codex app-server framing error");
                    return;
                }
            };

            for frame in frames {
                if let Err(err) = handle_incoming_frame(&conn, &state, &events_tx, frame).await {
                    tracing::warn!(error = %err, "codex app-server frame handling error");
                }
            }
        }
    });
}

async fn handle_incoming_frame(
    conn: &Arc<JsonRpcConnection>,
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    frame: Vec<u8>,
) -> Result<(), JsonRpcError> {
    let text = std::str::from_utf8(&frame).map_err(|_| JsonRpcError::Utf8)?;
    let value: serde_json::Value = serde_json::from_str(text)?;

    if let Some(method) = value.get("method").and_then(|m| m.as_str()) {
        if let Some(id_value) = value.get("id") {
            if let Some(id) = JsonRpcId::try_from_json(id_value) {
                handle_server_request(conn, id, method).await;
            }
            return Ok(());
        }

        let params = value
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        handle_notification(state, events_tx, method, &params).await;
        return Ok(());
    }

    if let Some(id_value) = value.get("id") {
        if let Some(id) = JsonRpcId::try_from_json(id_value) {
            if let Some(result) = value.get("result") {
                conn.deliver_response(id, Ok(result.clone())).await;
                return Ok(());
            }
            if let Some(err) = value.get("error") {
                conn.deliver_response(
                    id,
                    Err(JsonRpcError::RemoteError {
                        message: err.to_string(),
                    }),
                )
                .await;
                return Ok(());
            }
        }
    }

    Ok(())
}

async fn handle_server_request(conn: &Arc<JsonRpcConnection>, id: JsonRpcId, method: &str) {
    tracing::warn!(
        ?id,
        method,
        "codex app-server sent an unsupported server request"
    );
    let response = serde_json::json!({
        "id": id.to_json(),
        "error": { "code": -32601, "message": "method not supported" }
    });
    if let Err(err) = conn.send_value(&response).await {
        tracing::warn!(error = %err, "failed to send jsonrpc error response");
    }
}

async fn handle_notification(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    method: &str,
    params: &serde_json::Value,
) {
    match method {
        "thread/started" => {
            if let Some(thread_id) = params
                .get("thread")
                .and_then(|t| t.get("id"))
                .and_then(|v| v.as_str())
            {
                state.set_thread_id(thread_id.to_owned()).await;
            }
        }
        "turn/started" => {
            let Some(thread_id) = params.get("threadId").and_then(|v| v.as_str()) else {
                return;
            };
            let Some(turn_id) = params
                .get("turn")
                .and_then(|t| t.get("id"))
                .and_then(|v| v.as_str())
            else {
                return;
            };

            state.set_thread_id(thread_id.to_owned()).await;
            state.set_active_turn_id(Some(turn_id.to_owned())).await;

            let external_session_ref = ExternalSessionRef::CodexThread {
                thread_id: thread_id.to_owned(),
                turn_id: Some(turn_id.to_owned()),
            };

            let _ = events_tx
                .send(AppServerEvent::TurnStarted {
                    turn_id: Some(turn_id.to_owned()),
                    external_session_ref: Some(external_session_ref),
                })
                .await;
        }
        "turn/completed" => {
            let Some(thread_id) = params.get("threadId").and_then(|v| v.as_str()) else {
                return;
            };
            let Some(turn) = params.get("turn") else {
                return;
            };
            let Some(turn_id) = turn.get("id").and_then(|v| v.as_str()) else {
                return;
            };
            let status = turn.get("status").and_then(|v| v.as_str());
            let error = match status {
                Some("failed") => turn
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|message| {
                        ErrorEnvelope::new(ErrorCategory::Unavailable, message.to_owned())
                    }),
                _ => None,
            };

            state.set_thread_id(thread_id.to_owned()).await;
            state.set_active_turn_id(None).await;

            let external_session_ref = ExternalSessionRef::CodexThread {
                thread_id: thread_id.to_owned(),
                turn_id: Some(turn_id.to_owned()),
            };

            let _ = events_tx
                .send(AppServerEvent::TurnCompleted {
                    turn_id: Some(turn_id.to_owned()),
                    external_session_ref: Some(external_session_ref),
                    error,
                })
                .await;
        }
        "item/completed" => {
            let Some(item) = params.get("item") else {
                return;
            };
            let Some(item_type) = item.get("type").and_then(|v| v.as_str()) else {
                return;
            };

            match item_type {
                "agentMessage" => {
                    let Some(text) = item.get("text").and_then(|v| v.as_str()) else {
                        return;
                    };
                    let _ = events_tx
                        .send(AppServerEvent::AssistantMessage {
                            text: text.to_owned(),
                        })
                        .await;
                }
                "userMessage" => {
                    let text = item
                        .get("content")
                        .and_then(|v| v.as_array())
                        .map(|content| join_user_input_text(content))
                        .unwrap_or_default();
                    if text.is_empty() {
                        return;
                    }
                    let _ = events_tx.send(AppServerEvent::UserMessage { text }).await;
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn join_user_input_text(content: &[serde_json::Value]) -> String {
    let mut parts = Vec::new();
    for block in content {
        let Some(block_type) = block.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        parts.push(text.to_owned());
                    }
                }
            }
            _ => {}
        }
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_length_decoder_handles_chunked_reads() {
        let payload = br#"{"method":"ping","id":1}"#;
        let framed = format!("Content-Length: {}\r\n\r\n", payload.len());
        let mut bytes = framed.into_bytes();
        bytes.extend_from_slice(payload);

        let mut decoder = JsonRpcFramingDecoder::new(JsonRpcWireFormat::ContentLength);
        let mut out = Vec::new();

        for chunk in bytes.chunks(3) {
            let frames = decoder.push(chunk).expect("push");
            out.extend(frames);
        }

        assert_eq!(out.len(), 1);
        assert_eq!(out[0], payload);
    }

    #[test]
    fn jsonl_decoder_handles_multiple_messages() {
        let mut decoder = JsonRpcFramingDecoder::new(JsonRpcWireFormat::JsonLines);
        let frames = decoder
            .push(b"{\"method\":\"a\"}\n{\"method\":\"b\"}\n")
            .expect("push");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], br#"{"method":"a"}"#);
        assert_eq!(frames[1], br#"{"method":"b"}"#);
    }
}
