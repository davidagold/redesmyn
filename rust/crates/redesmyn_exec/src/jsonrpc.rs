use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use redesmyn_logging::tracing;
use tokio::io::{AsyncWrite, AsyncWriteExt as _};
use tokio::sync::{Mutex, oneshot};

const JSONRPC_VERSION: &str = "2.0";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum JsonRpcId {
    Number(i64),
    String(String),
}

impl JsonRpcId {
    pub(crate) fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Number(n) => serde_json::Value::Number((*n).into()),
            Self::String(s) => serde_json::Value::String(s.clone()),
        }
    }

    pub(crate) fn try_from_json(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::Number(n) => n.as_i64().map(Self::Number),
            serde_json::Value::String(s) => Some(Self::String(s.clone())),
            _ => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum JsonRpcError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid utf-8")]
    Utf8,
    #[error("invalid json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("jsonrpc error response: {message}")]
    RemoteError { message: String },
}

pub(crate) struct JsonRpcConnection {
    writer: Mutex<Box<dyn AsyncWrite + Unpin + Send>>,
    next_id: AtomicU64,
    pending: Mutex<HashMap<JsonRpcId, oneshot::Sender<Result<serde_json::Value, JsonRpcError>>>>,
}

impl std::fmt::Debug for JsonRpcConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonRpcConnection")
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .field("writer", &"<writer>")
            .field("pending", &"<pending>")
            .finish()
    }
}

impl JsonRpcConnection {
    pub(crate) fn new(writer: Box<dyn AsyncWrite + Unpin + Send>) -> Self {
        Self {
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
        let header = format!("Content-Length: {}\r\n\r\n", json.len());

        let mut writer = self.writer.lock().await;
        writer.write_all(header.as_bytes()).await?;
        writer.write_all(&json).await?;
        writer.flush().await?;
        Ok(())
    }

    pub(crate) async fn request(
        &self,
        method: &'static str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, JsonRpcError> {
        let request_id = self.next_request_id();
        let (tx, rx) = oneshot::channel();

        {
            let mut pending = self.pending.lock().await;
            pending.insert(request_id.clone(), tx);
        }

        let mut payload = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "id": request_id.to_json(),
        });
        if let Some(params) = params {
            payload["params"] = params;
        }

        if let Err(err) = self.send_value(&payload).await {
            let mut pending = self.pending.lock().await;
            pending.remove(&request_id);
            return Err(err);
        }

        // TODO(T-68): Add per-request timeout/cancellation so a wedged or stalled app-server
        // doesn't hang structured turns indefinitely. The supervisor should be able to surface a
        // clear error and/or trigger reconnect on timeout.
        match rx.await {
            Ok(payload) => payload,
            Err(_) => Err(JsonRpcError::RemoteError {
                message: "request response channel closed".to_owned(),
            }),
        }
    }

    pub(crate) async fn notify(
        &self,
        method: &'static str,
        params: Option<serde_json::Value>,
    ) -> Result<(), JsonRpcError> {
        let mut payload = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
        });
        if let Some(params) = params {
            payload["params"] = params;
        }
        self.send_value(&payload).await
    }

    pub(crate) async fn deliver_response(
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

    pub(crate) async fn respond_ok(&self, id: JsonRpcId, result: serde_json::Value) {
        let response = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "id": id.to_json(),
            "result": result,
        });
        if let Err(err) = self.send_value(&response).await {
            tracing::warn!(error = %err, "failed to send jsonrpc response");
        }
    }

    pub(crate) async fn respond_error(&self, id: JsonRpcId, code: i64, message: &str) {
        let response = serde_json::json!({
            "jsonrpc": JSONRPC_VERSION,
            "id": id.to_json(),
            "error": { "code": code, "message": message }
        });
        if let Err(err) = self.send_value(&response).await {
            tracing::warn!(error = %err, "failed to send jsonrpc error response");
        }
    }
}
