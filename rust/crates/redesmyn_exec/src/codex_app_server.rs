use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use redesmyn_domain::agent::{AppServerTurnIntent, ExternalSessionRef as DomainExternalSessionRef};
use redesmyn_logging::tracing;
use redesmyn_protocol::session::ExternalSessionRef;
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};
use tokio::process::Command;
use tokio::sync::{Mutex, mpsc};

use crate::app_server::{
    AppServerClient, AppServerConnection, AppServerEvent, AppServerProcess, AppServerProcessError,
    AppServerRequest, AppServerRequestError, AppServerResponse, BoxFuture,
};
use crate::content_length_framing::ContentLengthFramingDecoder;
use crate::jsonrpc::{JsonRpcConnection, JsonRpcError, JsonRpcId};

#[derive(Debug, Clone, Default)]
pub struct CodexAppServerAuth {
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub account_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AuthParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    access_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    refresh_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    account_id: Option<String>,
}

impl From<&CodexAppServerAuth> for AuthParams {
    fn from(value: &CodexAppServerAuth) -> Self {
        Self {
            access_token: value.access_token.clone(),
            refresh_token: value.refresh_token.clone(),
            account_id: value.account_id.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ClientInfo {
    name: String,
    title: String,
    version: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct InitializeParams {
    protocol_version: u32,
    auth: AuthParams,
    client_info: ClientInfo,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct NewSessionParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UserMessageParams {
    message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resume: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CancelParams {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "conversationId"
    )]
    session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "turnId")]
    turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CommandExecutionApprovalParams {
    #[serde(rename = "commandId")]
    command_id: String,
    approved: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FileChangeApprovalParams {
    #[serde(rename = "proposalId")]
    proposal_id: String,
    approved: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpdateSessionParams {
    diff: Vec<SessionDiff>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
enum TurnCompletionStatus {
    Completed,
    Failed,
    #[serde(rename = "canceled", alias = "cancelled")]
    Canceled,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TurnFailure {
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
enum MessageRole {
    User,
    Assistant,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum SessionDiff {
    #[serde(rename = "newConversation")]
    NewSession {
        #[serde(rename = "conversationId")]
        session_id: String,
    },
    NewTurn {
        #[serde(rename = "turnId")]
        turn_id: String,
    },
    TurnCompleted {
        #[serde(rename = "turnId")]
        turn_id: String,
        status: TurnCompletionStatus,
        #[serde(default)]
        error: Option<TurnFailure>,
    },
    NewMessage {
        #[serde(rename = "messageId")]
        message_id: String,
        role: MessageRole,
        content: String,
        #[serde(default)]
        done: bool,
    },
    UpdateMessage {
        #[serde(rename = "messageId")]
        message_id: String,
        #[serde(default)]
        content: Option<String>,
        #[serde(default, rename = "contentDelta")]
        content_delta: Option<String>,
        #[serde(default)]
        done: Option<bool>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone)]
struct AccumulatedMessage {
    role: MessageRole,
    text: String,
    done: bool,
}

#[derive(Debug, Default)]
struct SessionAccumulator {
    messages: HashMap<String, AccumulatedMessage>,
    emitted_message_ids: HashSet<String>,
}

impl SessionAccumulator {
    fn reset(&mut self) {
        self.messages.clear();
        self.emitted_message_ids.clear();
    }

    fn apply_new_message(
        &mut self,
        message_id: String,
        role: MessageRole,
        content: String,
        done: bool,
    ) -> Option<AppServerEvent> {
        self.messages.insert(
            message_id.clone(),
            AccumulatedMessage {
                role,
                text: content,
                done,
            },
        );
        self.maybe_emit_message(&message_id)
    }

    fn apply_update_message(
        &mut self,
        message_id: String,
        content: Option<String>,
        content_delta: Option<String>,
        done: Option<bool>,
    ) -> Option<AppServerEvent> {
        let entry = self
            .messages
            .entry(message_id.clone())
            .or_insert(AccumulatedMessage {
                role: MessageRole::Unknown,
                text: String::new(),
                done: false,
            });

        if let Some(content) = content {
            entry.text = content;
        } else if let Some(delta) = content_delta {
            entry.text.push_str(&delta);
        }

        if let Some(done) = done {
            entry.done = done;
        }

        self.maybe_emit_message(&message_id)
    }

    fn maybe_emit_message(&mut self, message_id: &str) -> Option<AppServerEvent> {
        if self.emitted_message_ids.contains(message_id) {
            return None;
        }
        let entry = self.messages.get(message_id)?;
        if !entry.done || entry.text.trim().is_empty() {
            return None;
        }

        let text = entry.text.clone();
        let ev = match entry.role {
            MessageRole::User => AppServerEvent::UserMessage { text },
            MessageRole::Assistant => AppServerEvent::AssistantMessage { text },
            MessageRole::Unknown => return None,
        };

        self.emitted_message_ids.insert(message_id.to_owned());
        Some(ev)
    }
}

#[derive(Debug, Default)]
struct CodexAppServerStateInner {
    session_id: Option<String>,
    active_turn_id: Option<String>,
}

#[derive(Debug)]
struct CodexAppServerState {
    inner: Mutex<CodexAppServerStateInner>,
    accumulator: Mutex<SessionAccumulator>,
}

impl CodexAppServerState {
    fn new() -> Self {
        Self {
            inner: Mutex::new(CodexAppServerStateInner::default()),
            accumulator: Mutex::new(SessionAccumulator::default()),
        }
    }

    async fn reset_for_new_session(&self) {
        {
            let mut inner = self.inner.lock().await;
            inner.session_id = None;
            inner.active_turn_id = None;
        }
        self.accumulator.lock().await.reset();
    }

    async fn set_session_id(&self, session_id: String) {
        self.inner.lock().await.session_id = Some(session_id);
    }

    async fn session_id(&self) -> Option<String> {
        self.inner.lock().await.session_id.clone()
    }

    async fn active_turn_id(&self) -> Option<String> {
        self.inner.lock().await.active_turn_id.clone()
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
    protocol_version: u32,
    auth: CodexAppServerAuth,
}

impl CodexAppServerClient {
    fn new(
        conn: Arc<JsonRpcConnection>,
        state: Arc<CodexAppServerState>,
        cwd: PathBuf,
        client_name: String,
        client_title: String,
        client_version: String,
        protocol_version: u32,
        auth: CodexAppServerAuth,
    ) -> Self {
        Self {
            conn,
            state,
            cwd,
            client_name,
            client_title,
            client_version,
            protocol_version,
            auth,
        }
    }

    async fn initialize_handshake(&self) -> Result<(), AppServerProcessError> {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.initialize");
        let _guard = span.enter();

        let params = InitializeParams {
            protocol_version: self.protocol_version,
            auth: AuthParams::from(&self.auth),
            client_info: ClientInfo {
                name: self.client_name.clone(),
                title: self.client_title.clone(),
                version: self.client_version.clone(),
            },
        };

        let params =
            serde_json::to_value(params).map_err(|err| AppServerProcessError::ConnectFailed {
                reason: format!("initialize params serialization failed: {err}"),
            })?;

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

    async fn start_new_session(&self) -> Result<(), AppServerRequestError> {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.new_session");
        let _guard = span.enter();

        let params = NewSessionParams {
            cwd: Some(self.cwd.to_string_lossy().to_string()),
        };
        let params = serde_json::to_value(params).map_err(|err| AppServerRequestError::Failed {
            reason: format!("newConversation params serialization failed: {err}"),
        })?;

        let result = self
            .conn
            .request("newConversation", Some(params))
            .await
            .map_err(|err| AppServerRequestError::Failed {
                reason: format!("newConversation failed: {err}"),
            })?;
        if let Some(session_id) = parse_session_id_from_new_conversation_result(&result) {
            self.state.set_session_id(session_id).await;
        }
        Ok(())
    }

    async fn send_user_message(
        &self,
        prompt: &str,
        resume_session_id: Option<String>,
    ) -> Result<(), AppServerRequestError> {
        let span = redesmyn_logging::redesmyn_info_span!(
            "codex_app_server.user_message",
            resume = resume_session_id.as_deref().unwrap_or("<new>")
        );
        let _guard = span.enter();

        let params = UserMessageParams {
            message: prompt.to_owned(),
            resume: resume_session_id,
        };
        let params = serde_json::to_value(params).map_err(|err| AppServerRequestError::Failed {
            reason: format!("userMessage params serialization failed: {err}"),
        })?;

        let _ = self
            .conn
            .request("userMessage", Some(params))
            .await
            .map_err(|err| AppServerRequestError::Failed {
                reason: format!("userMessage failed: {err}"),
            })?;
        Ok(())
    }

    async fn cancel_active_turn(&self) -> Result<(), AppServerRequestError> {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.cancel");
        let _guard = span.enter();

        let session_id = self.state.session_id().await;
        let turn_id = self.state.active_turn_id().await;

        let params = CancelParams {
            session_id,
            turn_id,
        };
        let params = serde_json::to_value(params).map_err(|err| AppServerRequestError::Failed {
            reason: format!("cancel params serialization failed: {err}"),
        })?;

        let _ = self
            .conn
            .request("cancel", Some(params))
            .await
            .map_err(|err| AppServerRequestError::Failed {
                reason: format!("cancel failed: {err}"),
            })?;
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
                    match &intent {
                        AppServerTurnIntent::StartNew { .. } => {
                            self.state.reset_for_new_session().await;
                            self.start_new_session().await?;
                        }
                        AppServerTurnIntent::Resume { external, .. } => match external {
                            DomainExternalSessionRef::CodexSession { session_id, .. } => {
                                self.state.set_session_id(session_id.clone()).await;
                            }
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
                        },
                    }

                    let (prompt, resume) = match intent {
                        AppServerTurnIntent::StartNew { prompt } => (prompt, None),
                        AppServerTurnIntent::Resume { prompt, external } => match external {
                            DomainExternalSessionRef::CodexSession { session_id, .. } => {
                                (prompt, Some(session_id))
                            }
                            _ => (prompt, None),
                        },
                    };

                    self.send_user_message(&prompt, resume).await?;
                    Ok(AppServerResponse::MessageAccepted)
                }
                AppServerRequest::Interrupt => {
                    self.cancel_active_turn().await?;
                    Ok(AppServerResponse::Interrupted)
                }
            }
        })
    }
}

fn parse_session_id_from_new_conversation_result(result: &serde_json::Value) -> Option<String> {
    result
        .get("conversationId")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
        .or_else(|| {
            result
                .get("conversation_id")
                .and_then(|v| v.as_str())
                .map(ToOwned::to_owned)
        })
}

#[derive(Debug)]
pub struct CodexAppServerProcessConfig {
    pub argv: Vec<String>,
    pub cwd: PathBuf,
    pub env: Vec<(String, String)>,
    pub client_name: String,
    pub client_title: String,
    pub client_version: String,
    pub protocol_version: u32,
    pub auth: CodexAppServerAuth,
}

impl CodexAppServerProcessConfig {
    #[must_use]
    pub fn codex_default(cwd: PathBuf) -> Self {
        Self {
            argv: vec!["codex".to_owned(), "app-server".to_owned()],
            cwd,
            env: Vec::new(),
            client_name: "redesmyn".to_owned(),
            client_title: "Redesmyn".to_owned(),
            client_version: env!("CARGO_PKG_VERSION").to_owned(),
            protocol_version: 2,
            auth: CodexAppServerAuth::default(),
        }
    }
}

pub struct CodexAppServerProcess {
    config: CodexAppServerProcessConfig,
    child: Mutex<Option<tokio::process::Child>>,
    conn: Mutex<Option<Arc<JsonRpcConnection>>>,
}

impl CodexAppServerProcess {
    #[must_use]
    pub fn new(config: CodexAppServerProcessConfig) -> Self {
        Self {
            config,
            child: Mutex::new(None),
            conn: Mutex::new(None),
        }
    }

    pub async fn connect_stream(
        &self,
        reader: Box<dyn AsyncRead + Unpin + Send>,
        writer: Box<dyn AsyncWrite + Unpin + Send>,
    ) -> Result<AppServerConnection, AppServerProcessError> {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.connect");
        let _guard = span.enter();

        let state = Arc::new(CodexAppServerState::new());
        let conn = Arc::new(JsonRpcConnection::new(writer));
        *self.conn.lock().await = Some(Arc::clone(&conn));

        let (events_tx, events_rx) = mpsc::channel::<AppServerEvent>(256);
        spawn_reader_loop(reader, Arc::clone(&conn), Arc::clone(&state), events_tx);

        let client = Arc::new(CodexAppServerClient::new(
            Arc::clone(&conn),
            state,
            self.config.cwd.clone(),
            self.config.client_name.clone(),
            self.config.client_title.clone(),
            self.config.client_version.clone(),
            self.config.protocol_version,
            self.config.auth.clone(),
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
            if let Some(conn) = self.conn.lock().await.take() {
                let _ = conn.notify("exit", None).await;
            }

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
    mut reader: Box<dyn AsyncRead + Unpin + Send>,
    conn: Arc<JsonRpcConnection>,
    state: Arc<CodexAppServerState>,
    events_tx: mpsc::Sender<AppServerEvent>,
) {
    tokio::spawn(async move {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.reader");
        let _guard = span.enter();

        let mut decoder = ContentLengthFramingDecoder::new();
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
        let params = value
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        if let Some(id_value) = value.get("id") {
            if let Some(id) = JsonRpcId::try_from_json(id_value) {
                handle_server_request(conn, state, events_tx, id, method, &params).await;
            }
            return Ok(());
        }

        handle_notification(conn, state, events_tx, method, &params).await;
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

async fn handle_server_request(
    conn: &Arc<JsonRpcConnection>,
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    id: JsonRpcId,
    method: &str,
    params: &serde_json::Value,
) {
    match method {
        "conversationId" => {
            let session_id = state.session_id().await;
            match session_id {
                Some(session_id) => {
                    conn.respond_ok(id, serde_json::Value::String(session_id))
                        .await;
                }
                None => {
                    conn.respond_error(id, -32000, "missing active session id")
                        .await;
                }
            }
        }
        "loginWithChatGPT" => {
            let url = params
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned();

            let msg = if url.is_empty() {
                "Login required: run `codex login` and retry.".to_owned()
            } else {
                format!("Login required: open {url} in your browser, then retry.")
            };

            let _ = events_tx
                .send(AppServerEvent::AssistantMessage { text: msg })
                .await;

            conn.respond_ok(id, serde_json::json!({ "opened": false }))
                .await;
        }
        _ => {
            tracing::warn!(?id, method, "unsupported server request");
            conn.respond_error(id, -32601, "method not supported").await;
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunCommandParams {
    #[serde(rename = "commandId", alias = "id")]
    command_id: String,
    command: String,
    #[serde(default)]
    cwd: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProposalToEditFileParams {
    #[serde(rename = "proposalId", alias = "id")]
    proposal_id: String,
    path: String,
    diff: String,
}

async fn handle_notification(
    conn: &Arc<JsonRpcConnection>,
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    method: &str,
    params: &serde_json::Value,
) {
    match method {
        "updateConversation" => {
            let params: UpdateSessionParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid updateConversation params");
                    return;
                }
            };
            handle_update_session(state, events_tx, params).await;
        }
        "runCommand" => {
            let params: RunCommandParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid runCommand params");
                    return;
                }
            };

            let _ = events_tx
                .send(AppServerEvent::ToolInvocation {
                    tool_name: "run_command".to_owned(),
                    tool_call_id: Some(params.command_id.clone()),
                    input: serde_json::json!({
                        "command": params.command,
                        "cwd": params.cwd,
                    })
                    .to_string(),
                })
                .await;

            // Safe-by-default: do not auto-run; deny until a control-plane approval path exists.
            let conn = Arc::clone(conn);
            tokio::spawn(async move {
                let params = CommandExecutionApprovalParams {
                    command_id: params.command_id,
                    approved: false,
                };
                if let Ok(params) = serde_json::to_value(params) {
                    let _ = conn.request("commandExecutionApproval", Some(params)).await;
                }
            });
        }
        "proposalToEditFile" => {
            let params: ProposalToEditFileParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid proposalToEditFile params");
                    return;
                }
            };

            let _ = events_tx
                .send(AppServerEvent::ToolInvocation {
                    tool_name: "proposal_to_edit_file".to_owned(),
                    tool_call_id: Some(params.proposal_id.clone()),
                    input: serde_json::json!({
                        "path": params.path,
                        "diff": params.diff,
                    })
                    .to_string(),
                })
                .await;

            // Safe-by-default: do not auto-edit; deny until a control-plane approval path exists.
            let conn = Arc::clone(conn);
            tokio::spawn(async move {
                let params = FileChangeApprovalParams {
                    proposal_id: params.proposal_id,
                    approved: false,
                };
                if let Ok(params) = serde_json::to_value(params) {
                    let _ = conn.request("fileChangeApproval", Some(params)).await;
                }
            });
        }
        "shutdown" => {
            tracing::info!("codex app-server requested shutdown");
        }
        "appendToLog" | "rerender" => {
            // Non-durable UI-only notifications. Ignore for now.
        }
        _ => {}
    }
}

async fn handle_update_session(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: UpdateSessionParams,
) {
    let mut events = Vec::new();

    {
        let mut inner = state.inner.lock().await;
        let mut acc = state.accumulator.lock().await;

        for diff in params.diff {
            match diff {
                SessionDiff::NewSession { session_id } => {
                    if inner.session_id.as_deref() != Some(session_id.as_str()) {
                        inner.session_id = Some(session_id);
                    }
                }
                SessionDiff::NewTurn { turn_id } => {
                    inner.active_turn_id = Some(turn_id.clone());

                    let external_session_ref = inner.session_id.as_ref().map(|session_id| {
                        ExternalSessionRef::CodexSession {
                            session_id: session_id.clone(),
                            turn_id: Some(turn_id.clone()),
                        }
                    });

                    tracing::info!(
                        session_id = inner.session_id.as_deref().unwrap_or("<unknown>"),
                        turn_id = turn_id,
                        "codex app-server turn started"
                    );

                    events.push(AppServerEvent::TurnStarted {
                        turn_id: Some(turn_id),
                        external_session_ref,
                    });
                }
                SessionDiff::NewMessage {
                    message_id,
                    role,
                    content,
                    done,
                } => {
                    if let Some(ev) = acc.apply_new_message(message_id, role, content, done) {
                        events.push(ev);
                    }
                }
                SessionDiff::UpdateMessage {
                    message_id,
                    content,
                    content_delta,
                    done,
                } => {
                    if let Some(ev) =
                        acc.apply_update_message(message_id, content, content_delta, done)
                    {
                        events.push(ev);
                    }
                }
                SessionDiff::TurnCompleted {
                    turn_id,
                    status,
                    error,
                } => {
                    inner.active_turn_id = None;

                    let external_session_ref = inner.session_id.as_ref().map(|session_id| {
                        ExternalSessionRef::CodexSession {
                            session_id: session_id.clone(),
                            turn_id: Some(turn_id.clone()),
                        }
                    });

                    let error = match status {
                        TurnCompletionStatus::Failed => error
                            .and_then(|e| e.message)
                            .map(|message| ErrorEnvelope::new(ErrorCategory::Unavailable, message)),
                        _ => None,
                    };

                    tracing::info!(
                        session_id = inner.session_id.as_deref().unwrap_or("<unknown>"),
                        turn_id = turn_id,
                        status = ?status,
                        "codex app-server turn completed"
                    );

                    events.push(AppServerEvent::TurnCompleted {
                        turn_id: Some(turn_id),
                        external_session_ref,
                        error,
                    });
                }
                SessionDiff::Unknown => {}
            }
        }
    }

    for event in events {
        let _ = events_tx.send(event).await;
    }
}
