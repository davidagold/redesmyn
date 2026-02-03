use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use redesmyn_domain::agent::{AppServerTurnIntent, ExternalSessionRef as DomainExternalSessionRef};
use redesmyn_logging::tracing;
use redesmyn_protocol::session::{
    CommandExecutionPermissionRequest, ExternalSessionRef, FileChangePermissionRequest,
    PermissionDecision, PermissionDecisionBy, PermissionRequest, PermissionsMode,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt as _, AsyncRead, AsyncWrite, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::time::Duration;

use crate::app_server::{
    AppServerClient, AppServerConnection, AppServerEvent, AppServerProcess, AppServerProcessError,
    AppServerRequest, AppServerRequestError, AppServerResponse, BoxFuture,
};
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
struct ThreadStartParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ThreadResumeParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnStartParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    input: Vec<UserInput>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum UserInput {
    Text { text: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnInterruptParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
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

#[derive(Debug)]
struct CodexAppServerStateInner {
    session_id: Option<String>,
    active_turn_id: Option<String>,
    emitted_turn_starts: HashSet<String>,
    emitted_turn_completions: HashSet<String>,
    emitted_item_starts: HashSet<String>,
    emitted_item_ids: HashSet<String>,
    permissions_mode: PermissionsMode,
    pending_permission_requests: HashMap<String, oneshot::Sender<PermissionDecision>>,
}

impl Default for CodexAppServerStateInner {
    fn default() -> Self {
        Self {
            session_id: None,
            active_turn_id: None,
            emitted_turn_starts: HashSet::new(),
            emitted_turn_completions: HashSet::new(),
            emitted_item_starts: HashSet::new(),
            emitted_item_ids: HashSet::new(),
            permissions_mode: PermissionsMode::Ask,
            pending_permission_requests: HashMap::new(),
        }
    }
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
            inner.emitted_turn_starts.clear();
            inner.emitted_turn_completions.clear();
            inner.emitted_item_starts.clear();
            inner.emitted_item_ids.clear();
            inner.pending_permission_requests.clear();
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

    async fn permissions_mode(&self) -> PermissionsMode {
        self.inner.lock().await.permissions_mode
    }

    async fn set_permissions_mode(&self, mode: PermissionsMode) {
        self.inner.lock().await.permissions_mode = mode;
    }

    async fn register_permission_request(
        &self,
        request_id: String,
    ) -> Option<oneshot::Receiver<PermissionDecision>> {
        let (tx, rx) = oneshot::channel();
        let mut inner = self.inner.lock().await;
        if inner.pending_permission_requests.contains_key(&request_id) {
            return None;
        }
        inner.pending_permission_requests.insert(request_id, tx);
        Some(rx)
    }

    async fn clear_permission_request(&self, request_id: &str) {
        self.inner.lock().await.pending_permission_requests.remove(request_id);
    }

    async fn respond_permission_request(
        &self,
        request_id: &str,
        decision: PermissionDecision,
    ) -> Result<(), AppServerRequestError> {
        let tx = self
            .inner
            .lock()
            .await
            .pending_permission_requests
            .remove(request_id);
        let Some(tx) = tx else {
            return Err(AppServerRequestError::Failed {
                reason: format!("unknown permission request: {request_id}"),
            });
        };

        let _ = tx.send(decision);
        Ok(())
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
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.thread_start");
        let _guard = span.enter();

        let params = ThreadStartParams {
            cwd: Some(self.cwd.to_string_lossy().to_string()),
        };
        let params = serde_json::to_value(params).map_err(|err| AppServerRequestError::Failed {
            reason: format!("thread/start params serialization failed: {err}"),
        })?;

        let result = self
            .conn
            .request("thread/start", Some(params))
            .await
            .map_err(|err| AppServerRequestError::Failed {
                reason: format!("thread/start failed: {err}"),
            })?;

        let thread_id =
            parse_thread_id_from_result(&result).ok_or_else(|| AppServerRequestError::Failed {
                reason: format!(
                    "thread/start response missing thread id: {}",
                    result.to_string()
                ),
            })?;
        self.state.set_session_id(thread_id).await;
        Ok(())
    }

    async fn resume_session(&self, conversation_id: &str) -> Result<(), AppServerRequestError> {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.thread_resume");
        let _guard = span.enter();

        let params = ThreadResumeParams {
            thread_id: conversation_id.to_owned(),
            cwd: Some(self.cwd.to_string_lossy().to_string()),
        };
        let params = serde_json::to_value(params).map_err(|err| AppServerRequestError::Failed {
            reason: format!("thread/resume params serialization failed: {err}"),
        })?;

        let result = self
            .conn
            .request("thread/resume", Some(params))
            .await
            .map_err(|err| AppServerRequestError::Failed {
                reason: format!("thread/resume failed: {err}"),
            })?;

        if let Some(thread_id) = parse_thread_id_from_result(&result) {
            self.state.set_session_id(thread_id).await;
        } else {
            self.state.set_session_id(conversation_id.to_owned()).await;
        }

        Ok(())
    }

    async fn send_user_message(&self, prompt: &str) -> Result<(), AppServerRequestError> {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.turn_start");
        let _guard = span.enter();

        let thread_id =
            self.state
                .session_id()
                .await
                .ok_or_else(|| AppServerRequestError::Failed {
                    reason: "missing active codex thread id".to_owned(),
                })?;

        let params = TurnStartParams {
            thread_id,
            input: vec![UserInput::Text {
                text: prompt.to_owned(),
            }],
        };
        let params = serde_json::to_value(params).map_err(|err| AppServerRequestError::Failed {
            reason: format!("turn/start params serialization failed: {err}"),
        })?;

        let result = self
            .conn
            .request("turn/start", Some(params))
            .await
            .map_err(|err| AppServerRequestError::Failed {
                reason: format!("turn/start failed: {err}"),
            })?;

        if let Some(turn_id) = parse_turn_id_from_turn_start_result(&result) {
            let mut inner = self.state.inner.lock().await;
            inner.active_turn_id = Some(turn_id);
        }
        Ok(())
    }

    async fn cancel_active_turn(&self) -> Result<(), AppServerRequestError> {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.cancel");
        let _guard = span.enter();

        let session_id = self.state.session_id().await;
        let turn_id = self.state.active_turn_id().await;

        let (Some(session_id), Some(turn_id)) = (session_id, turn_id) else {
            return Ok(());
        };

        let params = TurnInterruptParams {
            thread_id: session_id,
            turn_id,
        };
        let params = serde_json::to_value(params).map_err(|err| AppServerRequestError::Failed {
            reason: format!("turn/interrupt params serialization failed: {err}"),
        })?;

        let _ = self
            .conn
            .request("turn/interrupt", Some(params))
            .await
            .map_err(|err| AppServerRequestError::Failed {
                reason: format!("turn/interrupt failed: {err}"),
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
                    let prompt = match &intent {
                        AppServerTurnIntent::StartNew { prompt } => prompt.as_str(),
                        AppServerTurnIntent::Resume { prompt, .. } => prompt.as_str(),
                    };

                    match &intent {
                        AppServerTurnIntent::StartNew { .. } => {
                            self.state.reset_for_new_session().await;
                            self.start_new_session().await?;
                        }
                        AppServerTurnIntent::Resume { external, .. } => {
                            let resume_id = match external {
                                DomainExternalSessionRef::CodexThread { thread_id, .. } => {
                                    thread_id.clone()
                                }
                                DomainExternalSessionRef::CodexSession { session_id, .. } => {
                                    session_id.clone()
                                }
                                DomainExternalSessionRef::None => {
                                    return Err(AppServerRequestError::Failed {
                                        reason: "cannot resume: ExternalSessionRef::None"
                                            .to_owned(),
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

                            let current = self.state.session_id().await;
                            if current.as_deref() != Some(resume_id.as_str()) {
                                self.state.reset_for_new_session().await;
                                self.resume_session(&resume_id).await?;
                            }
                        }
                    };

                    self.send_user_message(prompt).await?;
                    Ok(AppServerResponse::MessageAccepted)
                }
                AppServerRequest::Interrupt => {
                    self.cancel_active_turn().await?;
                    Ok(AppServerResponse::Interrupted)
                }
                AppServerRequest::SetPermissionsMode { mode } => {
                    self.state.set_permissions_mode(mode).await;
                    Ok(AppServerResponse::PermissionsModeSet)
                }
                AppServerRequest::RespondPermissionRequest {
                    request_id,
                    decision,
                } => {
                    self.state
                        .respond_permission_request(&request_id, decision)
                        .await?;
                    Ok(AppServerResponse::PermissionRequestResponded)
                }
            }
        })
    }
}

fn parse_thread_id_from_result(result: &serde_json::Value) -> Option<String> {
    fn take_string(value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.to_owned()),
            _ => None,
        }
    }

    fn lookup(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
        obj.get(key).and_then(take_string)
    }

    match result {
        serde_json::Value::String(_) => take_string(result),
        serde_json::Value::Object(obj) => lookup(obj, "conversationId")
            .or_else(|| lookup(obj, "conversation_id"))
            .or_else(|| lookup(obj, "threadId"))
            .or_else(|| lookup(obj, "thread_id"))
            .or_else(|| lookup(obj, "id"))
            .or_else(|| {
                obj.get("conversation")
                    .and_then(|v| v.as_object())
                    .and_then(|nested| lookup(nested, "id"))
            })
            .or_else(|| {
                obj.get("thread")
                    .and_then(|v| v.as_object())
                    .and_then(|nested| lookup(nested, "id"))
            }),
        _ => None,
    }
}

fn parse_turn_id_from_turn_start_result(result: &serde_json::Value) -> Option<String> {
    fn take_string(value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.to_owned()),
            _ => None,
        }
    }

    match result {
        serde_json::Value::String(_) => take_string(result),
        serde_json::Value::Object(obj) => obj
            .get("turn")
            .and_then(|v| v.as_object())
            .and_then(|turn| turn.get("id"))
            .and_then(take_string)
            .or_else(|| obj.get("turnId").and_then(take_string))
            .or_else(|| obj.get("turn_id").and_then(take_string)),
        _ => None,
    }
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
    reader: Box<dyn AsyncRead + Unpin + Send>,
    conn: Arc<JsonRpcConnection>,
    state: Arc<CodexAppServerState>,
    events_tx: mpsc::Sender<AppServerEvent>,
) {
    tokio::spawn(async move {
        let span = redesmyn_logging::redesmyn_info_span!("codex_app_server.reader");
        let _guard = span.enter();

        let mut reader = BufReader::new(reader);
        let mut line = String::new();

        loop {
            line.clear();
            let read = match reader.read_line(&mut line).await {
                Ok(0) => return,
                Ok(n) => n,
                Err(err) => {
                    tracing::warn!(error = %err, "codex app-server read failed");
                    return;
                }
            };

            if read == 0 {
                continue;
            }

            let frame = line.trim_end_matches(|c| c == '\n' || c == '\r');
            if frame.is_empty() {
                continue;
            }

            if let Err(err) = handle_incoming_frame(&conn, &state, &events_tx, frame).await {
                tracing::warn!(error = %err, "codex app-server frame handling error");
            }
        }
    });
}

async fn handle_incoming_frame(
    conn: &Arc<JsonRpcConnection>,
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    frame: &str,
) -> Result<(), JsonRpcError> {
    let value: serde_json::Value = serde_json::from_str(frame)?;

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
        "item/commandExecution/requestApproval" => {
            let params: CommandExecutionRequestApprovalParams = match serde_json::from_value(
                params.clone(),
            ) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid item/commandExecution/requestApproval params");
                    conn.respond_error(id, -32602, "invalid params").await;
                    return;
                }
            };

            let request_id = params.item_id.clone();
            let request = PermissionRequest::CommandExecution(CommandExecutionPermissionRequest {
                command: params.command.clone(),
                cwd: params.cwd.clone(),
                reason: params.reason.clone(),
            });

            let summary = params
                .command
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .map(|cmd| format!("Run command: {cmd}"))
                .unwrap_or_else(|| "Run a command".to_owned());

            let _ = events_tx
                .send(AppServerEvent::PermissionRequested {
                    request_id: request_id.clone(),
                    summary: summary.clone(),
                    request,
                })
                .await;

            let mode = state.permissions_mode().await;
            match mode {
                PermissionsMode::AutoApprove => {
                    conn.respond_ok(id, serde_json::json!({ "decision": "approve" }))
                        .await;
                    let _ = events_tx
                        .send(AppServerEvent::PermissionDecided {
                            request_id,
                            decision: PermissionDecision::Approve,
                            decided_by: PermissionDecisionBy::ModeAutoApprove,
                        })
                        .await;
                }
                PermissionsMode::Deny => {
                    conn.respond_ok(id, serde_json::json!({ "decision": "decline" }))
                        .await;
                    let _ = events_tx
                        .send(AppServerEvent::PermissionDecided {
                            request_id,
                            decision: PermissionDecision::Deny,
                            decided_by: PermissionDecisionBy::ModeAutoDeny,
                        })
                        .await;
                }
                _ => {
                    let Some(rx) = state.register_permission_request(request_id.clone()).await
                    else {
                        conn.respond_ok(id, serde_json::json!({ "decision": "decline" }))
                            .await;
                        let _ = events_tx
                            .send(AppServerEvent::PermissionDecided {
                                request_id,
                                decision: PermissionDecision::Deny,
                                decided_by: PermissionDecisionBy::Timeout,
                            })
                            .await;
                        return;
                    };

                    let conn = Arc::clone(conn);
                    let state = Arc::clone(state);
                    let events_tx = events_tx.clone();

                    tokio::spawn(async move {
                        const APPROVAL_TIMEOUT: Duration = Duration::from_secs(120);

                        let (decision, decided_by) =
                            match tokio::time::timeout(APPROVAL_TIMEOUT, rx).await {
                                Ok(Ok(decision)) => (decision, PermissionDecisionBy::User),
                                Ok(Err(_)) | Err(_) => (PermissionDecision::Deny, PermissionDecisionBy::Timeout),
                            };

                        if decided_by == PermissionDecisionBy::Timeout {
                            state.clear_permission_request(&request_id).await;
                        }

                        let decision_for_wire = match decision {
                            PermissionDecision::Approve => "approve",
                            PermissionDecision::Deny | PermissionDecision::Unknown => "decline",
                        };
                        conn.respond_ok(id, serde_json::json!({ "decision": decision_for_wire }))
                            .await;

                        let decision_for_event = match decision {
                            PermissionDecision::Approve => PermissionDecision::Approve,
                            PermissionDecision::Deny | PermissionDecision::Unknown => PermissionDecision::Deny,
                        };
                        let _ = events_tx
                            .send(AppServerEvent::PermissionDecided {
                                request_id,
                                decision: decision_for_event,
                                decided_by,
                            })
                            .await;
                    });
                }
            }
        }
        "item/fileChange/requestApproval" => {
            let params: FileChangeRequestApprovalParams = match serde_json::from_value(
                params.clone(),
            ) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid item/fileChange/requestApproval params");
                    conn.respond_error(id, -32602, "invalid params").await;
                    return;
                }
            };

            let request_id = params.item_id.clone();
            let request = PermissionRequest::FileChange(FileChangePermissionRequest {
                grant_root: params.grant_root.clone(),
                reason: params.reason.clone(),
            });

            let summary = params
                .grant_root
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .map(|root| format!("Allow file changes under: {root}"))
                .unwrap_or_else(|| "Allow a file change".to_owned());

            let _ = events_tx
                .send(AppServerEvent::PermissionRequested {
                    request_id: request_id.clone(),
                    summary: summary.clone(),
                    request,
                })
                .await;

            let mode = state.permissions_mode().await;
            match mode {
                PermissionsMode::AutoApprove => {
                    conn.respond_ok(id, serde_json::json!({ "decision": "approve" }))
                        .await;
                    let _ = events_tx
                        .send(AppServerEvent::PermissionDecided {
                            request_id,
                            decision: PermissionDecision::Approve,
                            decided_by: PermissionDecisionBy::ModeAutoApprove,
                        })
                        .await;
                }
                PermissionsMode::Deny => {
                    conn.respond_ok(id, serde_json::json!({ "decision": "decline" }))
                        .await;
                    let _ = events_tx
                        .send(AppServerEvent::PermissionDecided {
                            request_id,
                            decision: PermissionDecision::Deny,
                            decided_by: PermissionDecisionBy::ModeAutoDeny,
                        })
                        .await;
                }
                _ => {
                    let Some(rx) = state.register_permission_request(request_id.clone()).await
                    else {
                        conn.respond_ok(id, serde_json::json!({ "decision": "decline" }))
                            .await;
                        let _ = events_tx
                            .send(AppServerEvent::PermissionDecided {
                                request_id,
                                decision: PermissionDecision::Deny,
                                decided_by: PermissionDecisionBy::Timeout,
                            })
                            .await;
                        return;
                    };

                    let conn = Arc::clone(conn);
                    let state = Arc::clone(state);
                    let events_tx = events_tx.clone();

                    tokio::spawn(async move {
                        const APPROVAL_TIMEOUT: Duration = Duration::from_secs(120);

                        let (decision, decided_by) =
                            match tokio::time::timeout(APPROVAL_TIMEOUT, rx).await {
                                Ok(Ok(decision)) => (decision, PermissionDecisionBy::User),
                                Ok(Err(_)) | Err(_) => (PermissionDecision::Deny, PermissionDecisionBy::Timeout),
                            };

                        if decided_by == PermissionDecisionBy::Timeout {
                            state.clear_permission_request(&request_id).await;
                        }

                        let decision_for_wire = match decision {
                            PermissionDecision::Approve => "approve",
                            PermissionDecision::Deny | PermissionDecision::Unknown => "decline",
                        };
                        conn.respond_ok(id, serde_json::json!({ "decision": decision_for_wire }))
                            .await;

                        let decision_for_event = match decision {
                            PermissionDecision::Approve => PermissionDecision::Approve,
                            PermissionDecision::Deny | PermissionDecision::Unknown => PermissionDecision::Deny,
                        };
                        let _ = events_tx
                            .send(AppServerEvent::PermissionDecided {
                                request_id,
                                decision: decision_for_event,
                                decided_by,
                            })
                            .await;
                    });
                }
            }
        }
        "item/tool/requestUserInput" => {
            // Safe-by-default: without a control-plane surface for this, fail fast so the
            // agent can continue the turn instead of hanging indefinitely.
            conn.respond_error(id, -32601, "user input requests not supported")
                .await;
        }
        "applyPatchApproval" | "execCommandApproval" => {
            // Legacy approval APIs (for sendUserMessage/sendUserTurn). We don't support these yet;
            // deny so the agent can continue instead of hanging.
            conn.respond_ok(id, serde_json::json!({ "decision": "denied" }))
                .await;
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
struct CommandExecutionRequestApprovalParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
    #[serde(rename = "itemId")]
    item_id: String,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileChangeRequestApprovalParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
    #[serde(rename = "itemId")]
    item_id: String,
    #[serde(default)]
    grant_root: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ItemEventParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
    item: ThreadItem,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ThreadItem {
    AgentMessage {
        id: String,
        #[serde(default)]
        text: String,
    },
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<String>,
        #[serde(default)]
        content: Vec<String>,
    },
    CommandExecution {
        id: String,
        command: String,
        cwd: String,
        status: ItemStatus,
        #[serde(default, rename = "aggregatedOutput")]
        aggregated_output: Option<String>,
        #[serde(default, rename = "exitCode")]
        exit_code: Option<i32>,
    },
    FileChange {
        id: String,
        status: ItemStatus,
        #[serde(default)]
        changes: Vec<serde_json::Value>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ItemStatus {
    InProgress,
    Completed,
    Failed,
    Declined,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TurnCompletedParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    turn: TurnSummary,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TurnStartedParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    turn: TurnSummary,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AgentMessageDeltaParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
    #[serde(rename = "itemId")]
    item_id: String,
    delta: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReasoningSummaryPartAddedParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
    #[serde(rename = "itemId")]
    item_id: String,
    #[serde(rename = "summaryIndex")]
    summary_index: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReasoningSummaryTextDeltaParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
    #[serde(rename = "itemId")]
    item_id: String,
    delta: String,
    #[serde(rename = "summaryIndex")]
    summary_index: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReasoningTextDeltaParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
    #[serde(rename = "itemId")]
    item_id: String,
    delta: String,
    #[serde(rename = "contentIndex")]
    content_index: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ItemOutputDeltaParams {
    #[serde(rename = "threadId")]
    thread_id: String,
    #[serde(rename = "turnId")]
    turn_id: String,
    #[serde(rename = "itemId")]
    item_id: String,
    delta: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TurnSummary {
    id: String,
    status: TurnStatus,
    #[serde(default)]
    error: Option<TurnError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
enum TurnStatus {
    Completed,
    Failed,
    Interrupted,
    InProgress,
    #[serde(rename = "cancelled", alias = "canceled")]
    Cancelled,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TurnError {
    #[serde(default)]
    message: Option<String>,
}

async fn handle_notification(
    _conn: &Arc<JsonRpcConnection>,
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
        "turn/started" => {
            let params: TurnStartedParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid turn/started params");
                    return;
                }
            };
            handle_turn_started(state, events_tx, params).await;
        }
        "item/started" => {
            let params: ItemEventParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid item/started params");
                    return;
                }
            };
            handle_item_started(state, events_tx, params).await;
        }
        "item/agentMessage/delta" => {
            let params: AgentMessageDeltaParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid item/agentMessage/delta params");
                    return;
                }
            };
            handle_agent_message_delta(state, events_tx, params).await;
        }
        "item/reasoning/summaryPartAdded" => {
            let params: ReasoningSummaryPartAddedParams =
                match serde_json::from_value(params.clone()) {
                    Ok(v) => v,
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "invalid item/reasoning/summaryPartAdded params"
                        );
                        return;
                    }
                };
            handle_reasoning_summary_part_added(state, events_tx, params).await;
        }
        "item/reasoning/summaryTextDelta" => {
            let params: ReasoningSummaryTextDeltaParams =
                match serde_json::from_value(params.clone()) {
                    Ok(v) => v,
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            "invalid item/reasoning/summaryTextDelta params"
                        );
                        return;
                    }
                };
            handle_reasoning_summary_text_delta(state, events_tx, params).await;
        }
        "item/reasoning/textDelta" => {
            let params: ReasoningTextDeltaParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid item/reasoning/textDelta params");
                    return;
                }
            };
            handle_reasoning_text_delta(state, events_tx, params).await;
        }
        "item/commandExecution/outputDelta" => {
            let params: ItemOutputDeltaParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid item/commandExecution/outputDelta params");
                    return;
                }
            };
            handle_tool_output_delta(state, events_tx, "exec_command", params).await;
        }
        "item/fileChange/outputDelta" => {
            let params: ItemOutputDeltaParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid item/fileChange/outputDelta params");
                    return;
                }
            };
            handle_tool_output_delta(state, events_tx, "file_change", params).await;
        }
        "item/completed" => {
            let params: ItemEventParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid item/completed params");
                    return;
                }
            };
            handle_item_completed(state, events_tx, params).await;
        }
        "turn/completed" => {
            let params: TurnCompletedParams = match serde_json::from_value(params.clone()) {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(error = %err, "invalid turn/completed params");
                    return;
                }
            };
            handle_turn_completed(state, events_tx, params).await;
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

async fn handle_turn_started(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: TurnStartedParams,
) {
    let event = {
        let mut inner = state.inner.lock().await;
        match inner.session_id.as_deref() {
            Some(session_id) if session_id != params.thread_id => return,
            None => {
                inner.session_id = Some(params.thread_id.clone());
            }
            Some(_) => {}
        }

        inner.active_turn_id = Some(params.turn.id.clone());

        if inner.emitted_turn_starts.insert(params.turn.id.clone()) {
            Some(AppServerEvent::TurnStarted {
                turn_id: Some(params.turn.id.clone()),
                external_session_ref: Some(ExternalSessionRef::CodexThread {
                    thread_id: params.thread_id.clone(),
                    turn_id: Some(params.turn.id.clone()),
                }),
            })
        } else {
            None
        }
    };

    if let Some(event) = event {
        let _ = events_tx.send(event).await;
    }
}

async fn handle_agent_message_delta(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: AgentMessageDeltaParams,
) {
    if params.delta.is_empty() {
        return;
    }

    {
        let mut inner = state.inner.lock().await;
        match inner.session_id.as_deref() {
            Some(session_id) if session_id != params.thread_id => return,
            None => {
                inner.session_id = Some(params.thread_id.clone());
            }
            Some(_) => {}
        }

        inner.active_turn_id = Some(params.turn_id.clone());
    }

    let _ = events_tx
        .send(AppServerEvent::AssistantMessageDelta {
            turn_id: Some(params.turn_id),
            item_id: Some(params.item_id),
            delta: params.delta,
        })
        .await;
}

async fn handle_reasoning_summary_part_added(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: ReasoningSummaryPartAddedParams,
) {
    {
        let mut inner = state.inner.lock().await;
        match inner.session_id.as_deref() {
            Some(session_id) if session_id != params.thread_id => return,
            None => {
                inner.session_id = Some(params.thread_id.clone());
            }
            Some(_) => {}
        }

        inner.active_turn_id = Some(params.turn_id.clone());
    }

    let _ = events_tx
        .send(AppServerEvent::AssistantReasoningSummaryPartAdded {
            turn_id: Some(params.turn_id),
            item_id: Some(params.item_id),
            summary_index: params.summary_index,
        })
        .await;
}

async fn handle_reasoning_summary_text_delta(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: ReasoningSummaryTextDeltaParams,
) {
    if params.delta.is_empty() {
        return;
    }

    {
        let mut inner = state.inner.lock().await;
        match inner.session_id.as_deref() {
            Some(session_id) if session_id != params.thread_id => return,
            None => {
                inner.session_id = Some(params.thread_id.clone());
            }
            Some(_) => {}
        }

        inner.active_turn_id = Some(params.turn_id.clone());
    }

    let _ = events_tx
        .send(AppServerEvent::AssistantReasoningSummaryDelta {
            turn_id: Some(params.turn_id),
            item_id: Some(params.item_id),
            summary_index: params.summary_index,
            delta: params.delta,
        })
        .await;
}

async fn handle_reasoning_text_delta(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: ReasoningTextDeltaParams,
) {
    if params.delta.is_empty() {
        return;
    }

    {
        let mut inner = state.inner.lock().await;
        match inner.session_id.as_deref() {
            Some(session_id) if session_id != params.thread_id => return,
            None => {
                inner.session_id = Some(params.thread_id.clone());
            }
            Some(_) => {}
        }

        inner.active_turn_id = Some(params.turn_id.clone());
    }

    let _ = events_tx
        .send(AppServerEvent::AssistantReasoningRawDelta {
            turn_id: Some(params.turn_id),
            item_id: Some(params.item_id),
            content_index: params.content_index,
            delta: params.delta,
        })
        .await;
}

async fn handle_tool_output_delta(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    tool_name: &'static str,
    params: ItemOutputDeltaParams,
) {
    if params.delta.is_empty() {
        return;
    }

    {
        let mut inner = state.inner.lock().await;
        match inner.session_id.as_deref() {
            Some(session_id) if session_id != params.thread_id => return,
            None => {
                inner.session_id = Some(params.thread_id.clone());
            }
            Some(_) => {}
        }

        inner.active_turn_id = Some(params.turn_id.clone());
    }

    let _ = events_tx
        .send(AppServerEvent::ToolOutputDelta {
            turn_id: Some(params.turn_id),
            tool_name: tool_name.to_owned(),
            tool_call_id: Some(params.item_id),
            delta: params.delta,
        })
        .await;
}

async fn handle_item_started(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: ItemEventParams,
) {
    let (turn_started, tool_invocation) = {
        let mut inner = state.inner.lock().await;
        match inner.session_id.as_deref() {
            Some(session_id) if session_id != params.thread_id => return,
            None => {
                inner.session_id = Some(params.thread_id.clone());
            }
            Some(_) => {}
        }

        inner.active_turn_id = Some(params.turn_id.clone());

        let turn_started = inner
            .emitted_turn_starts
            .insert(params.turn_id.clone())
            .then(|| AppServerEvent::TurnStarted {
                turn_id: Some(params.turn_id.clone()),
                external_session_ref: Some(ExternalSessionRef::CodexThread {
                    thread_id: params.thread_id.clone(),
                    turn_id: Some(params.turn_id.clone()),
                }),
            });

        let tool_invocation = match &params.item {
            ThreadItem::CommandExecution {
                id, command, cwd, ..
            } => inner.emitted_item_starts.insert(id.clone()).then(|| {
                AppServerEvent::ToolInvocation {
                    tool_name: "exec_command".to_owned(),
                    tool_call_id: Some(id.clone()),
                    input: serde_json::json!({
                        "command": command,
                        "cwd": cwd,
                    })
                    .to_string(),
                }
            }),
            ThreadItem::FileChange { id, changes, .. } => inner
                .emitted_item_starts
                .insert(id.clone())
                .then(|| AppServerEvent::ToolInvocation {
                    tool_name: "file_change".to_owned(),
                    tool_call_id: Some(id.clone()),
                    input: serde_json::json!({
                        "changes": changes.len(),
                    })
                    .to_string(),
                }),
            _ => None,
        };

        (turn_started, tool_invocation)
    };

    if let Some(event) = turn_started {
        let _ = events_tx.send(event).await;
    }
    if let Some(event) = tool_invocation {
        let _ = events_tx.send(event).await;
    }
}

async fn handle_item_completed(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: ItemEventParams,
) {
    match params.item {
        ThreadItem::AgentMessage { id, text } => {
            if text.trim().is_empty() {
                return;
            }

            {
                let mut inner = state.inner.lock().await;
                match inner.session_id.as_deref() {
                    Some(session_id) if session_id != params.thread_id => return,
                    None => {
                        inner.session_id = Some(params.thread_id.clone());
                    }
                    Some(_) => {}
                }

                if !inner.emitted_item_ids.insert(id) {
                    return;
                }
            }

            let _ = events_tx
                .send(AppServerEvent::AssistantMessage { text })
                .await;
        }
        ThreadItem::Reasoning {
            id,
            summary,
            content,
        } => {
            let summary_text = summary
                .into_iter()
                .filter(|line| !line.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            let raw_text = content
                .into_iter()
                .filter(|line| !line.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n");

            if summary_text.trim().is_empty() && raw_text.trim().is_empty() {
                return;
            }

            {
                let mut inner = state.inner.lock().await;
                match inner.session_id.as_deref() {
                    Some(session_id) if session_id != params.thread_id => return,
                    None => {
                        inner.session_id = Some(params.thread_id.clone());
                    }
                    Some(_) => {}
                }

                if !inner.emitted_item_ids.insert(id.clone()) {
                    return;
                }
            }

            let _ = events_tx
                .send(AppServerEvent::AssistantReasoning {
                    turn_id: Some(params.turn_id),
                    item_id: Some(id),
                    summary: summary_text,
                    raw: (!raw_text.trim().is_empty()).then_some(raw_text),
                    signature: None,
                })
                .await;
        }
        ThreadItem::CommandExecution {
            id,
            status,
            aggregated_output,
            exit_code,
            ..
        } => {
            {
                let mut inner = state.inner.lock().await;
                match inner.session_id.as_deref() {
                    Some(session_id) if session_id != params.thread_id => return,
                    None => {
                        inner.session_id = Some(params.thread_id.clone());
                    }
                    Some(_) => {}
                }

                if !inner.emitted_item_ids.insert(id.clone()) {
                    return;
                }
            }

            let mut output = String::new();
            if status != ItemStatus::Completed {
                output.push_str(&format!("status: {status:?}\n"));
            }
            if let Some(exit_code) = exit_code {
                output.push_str(&format!("exit_code: {exit_code}\n"));
            }
            if let Some(aggregated_output) = aggregated_output {
                output.push_str(&aggregated_output);
            }

            let _ = events_tx
                .send(AppServerEvent::ToolResult {
                    tool_name: "exec_command".to_owned(),
                    tool_call_id: Some(id),
                    output,
                    error: None,
                })
                .await;
        }
        ThreadItem::FileChange {
            id,
            status,
            changes,
        } => {
            {
                let mut inner = state.inner.lock().await;
                match inner.session_id.as_deref() {
                    Some(session_id) if session_id != params.thread_id => return,
                    None => {
                        inner.session_id = Some(params.thread_id.clone());
                    }
                    Some(_) => {}
                }

                if !inner.emitted_item_ids.insert(id.clone()) {
                    return;
                }
            }

            let output = serde_json::json!({
                "status": format!("{status:?}"),
                "changes": changes.len(),
            })
            .to_string();

            let _ = events_tx
                .send(AppServerEvent::ToolResult {
                    tool_name: "file_change".to_owned(),
                    tool_call_id: Some(id),
                    output,
                    error: None,
                })
                .await;
        }
        _ => {}
    }
}

async fn handle_turn_completed(
    state: &Arc<CodexAppServerState>,
    events_tx: &mpsc::Sender<AppServerEvent>,
    params: TurnCompletedParams,
) {
    let event = {
        let mut inner = state.inner.lock().await;
        match inner.session_id.as_deref() {
            Some(session_id) if session_id != params.thread_id => return,
            None => {
                inner.session_id = Some(params.thread_id.clone());
            }
            Some(_) => {}
        }

        if !inner
            .emitted_turn_completions
            .insert(params.turn.id.clone())
        {
            return;
        }

        if inner.active_turn_id.as_deref() == Some(params.turn.id.as_str()) {
            inner.active_turn_id = None;
        }

        let error = match params.turn.status {
            TurnStatus::Failed => params
                .turn
                .error
                .and_then(|e| e.message)
                .map(|message| ErrorEnvelope::new(ErrorCategory::Unavailable, message)),
            _ => None,
        };

        Some(AppServerEvent::TurnCompleted {
            turn_id: Some(params.turn.id.clone()),
            external_session_ref: Some(ExternalSessionRef::CodexThread {
                thread_id: params.thread_id.clone(),
                turn_id: Some(params.turn.id.clone()),
            }),
            error,
        })
    };

    if let Some(event) = event {
        let _ = events_tx.send(event).await;
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

                    // TODO(T-68): Harden against out-of-order diffs (e.g. `newTurn` arriving
                    // before `newConversation`) so we always capture/persist the external
                    // `session_id` on TurnStarted events.
                    let external_session_ref = inner.session_id.as_ref().map(|session_id| {
                        ExternalSessionRef::CodexThread {
                            thread_id: session_id.clone(),
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
                    let delta_text =
                        (role == MessageRole::Assistant && !done && !content.is_empty())
                            .then(|| content.clone());
                    if let Some(ev) = acc.apply_new_message(message_id.clone(), role, content, done)
                    {
                        events.push(ev);
                    }
                    if let Some(delta) = delta_text {
                        events.push(AppServerEvent::AssistantMessageDelta {
                            turn_id: inner.active_turn_id.clone(),
                            item_id: Some(message_id),
                            delta,
                        });
                    }
                }
                SessionDiff::UpdateMessage {
                    message_id,
                    content,
                    content_delta,
                    done,
                } => {
                    let role = acc
                        .messages
                        .get(&message_id)
                        .map(|message| message.role)
                        .unwrap_or(MessageRole::Unknown);
                    let delta_text = match &content_delta {
                        Some(delta)
                            if role == MessageRole::Assistant
                                && !delta.is_empty()
                                && !acc.emitted_message_ids.contains(&message_id) =>
                        {
                            Some(delta.clone())
                        }
                        _ => None,
                    };
                    if let Some(ev) =
                        acc.apply_update_message(message_id.clone(), content, content_delta, done)
                    {
                        events.push(ev);
                    }
                    if let Some(delta) = delta_text {
                        events.push(AppServerEvent::AssistantMessageDelta {
                            turn_id: inner.active_turn_id.clone(),
                            item_id: Some(message_id),
                            delta,
                        });
                    }
                }
                SessionDiff::TurnCompleted {
                    turn_id,
                    status,
                    error,
                } => {
                    inner.active_turn_id = None;

                    let external_session_ref = inner.session_id.as_ref().map(|session_id| {
                        ExternalSessionRef::CodexThread {
                            thread_id: session_id.clone(),
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
