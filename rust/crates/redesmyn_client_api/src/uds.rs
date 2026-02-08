//! Unix domain socket dialer helpers for the Client API protocol.
//!
//! This is intentionally small: it only handles connecting and wrapping a stream in the
//! `redesmyn_transport` framed endpoint. Higher-level retry/reconnect behavior should be layered
//! above `Client`/`ClientTask`.

use std::path::PathBuf;
use std::time::Duration;

use redesmyn_protocol::client::ClientFrame;
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope};
use redesmyn_transport::BoxFuture;
use redesmyn_transport::client::codec::{JsonCodec, ProtobufCodec};
use redesmyn_transport::client::framed::{DEFAULT_MAX_FRAME_LEN, FramedEndpoint};
use redesmyn_transport::client::{ClientConnection, ClientTransportError};

use tokio::net::UnixStream;

use crate::{Client, ClientTask};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientApiCodec {
    Json,
    Protobuf,
}

#[derive(Debug, Clone)]
pub struct UdsConnectOptions {
    pub socket_path: PathBuf,
    pub codec: ClientApiCodec,
    pub connect_timeout: Duration,
    pub max_frame_len: usize,
}

impl UdsConnectOptions {
    #[must_use]
    pub fn new(socket_path: PathBuf) -> Self {
        Self {
            socket_path,
            codec: ClientApiCodec::Protobuf,
            connect_timeout: Duration::from_secs(2),
            max_frame_len: DEFAULT_MAX_FRAME_LEN,
        }
    }
}

#[derive(Debug)]
pub enum UdsConnection {
    Json(FramedEndpoint<JsonCodec, UnixStream>),
    Protobuf(FramedEndpoint<ProtobufCodec, UnixStream>),
}

impl ClientConnection for UdsConnection {
    fn send(&mut self, frame: ClientFrame) -> BoxFuture<'_, Result<(), ClientTransportError>> {
        match self {
            Self::Json(conn) => conn.send(frame),
            Self::Protobuf(conn) => conn.send(frame),
        }
    }

    fn recv(&mut self) -> BoxFuture<'_, Result<ClientFrame, ClientTransportError>> {
        match self {
            Self::Json(conn) => conn.recv(),
            Self::Protobuf(conn) => conn.recv(),
        }
    }
}

pub async fn connect_uds_connection(
    options: &UdsConnectOptions,
) -> Result<UdsConnection, ErrorEnvelope> {
    let stream = connect_stream(options).await?;

    let max_frame_len = options.max_frame_len;
    let conn = match options.codec {
        ClientApiCodec::Json => UdsConnection::Json(
            FramedEndpoint::new(stream, JsonCodec::new()).with_max_frame_len(max_frame_len),
        ),
        ClientApiCodec::Protobuf => UdsConnection::Protobuf(
            FramedEndpoint::new(stream, ProtobufCodec::new()).with_max_frame_len(max_frame_len),
        ),
    };

    Ok(conn)
}

pub async fn connect_uds(
    options: UdsConnectOptions,
    buffer: usize,
) -> Result<(Client, ClientTask<UdsConnection>), ErrorEnvelope> {
    let conn = connect_uds_connection(&options).await?;
    Ok(Client::connect(conn, buffer))
}

async fn connect_stream(options: &UdsConnectOptions) -> Result<UnixStream, ErrorEnvelope> {
    let deadline = tokio::time::Instant::now() + options.connect_timeout;
    let mut backoff = Duration::from_millis(10);

    loop {
        match UnixStream::connect(&options.socket_path).await {
            Ok(stream) => return Ok(stream),
            Err(err) => {
                if tokio::time::Instant::now() >= deadline {
                    return Err(connect_error(options, err));
                }
            }
        }

        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_millis(250));
    }
}

fn connect_error(options: &UdsConnectOptions, err: std::io::Error) -> ErrorEnvelope {
    let mut detail = ErrorDetail::new();
    detail.insert(
        "socket_path".to_string(),
        options.socket_path.display().to_string(),
    );
    detail.insert(
        "connect_timeout_ms".to_string(),
        options.connect_timeout.as_millis().to_string(),
    );
    detail.insert("codec".to_string(), format!("{:?}", options.codec));
    detail.insert("io_error".to_string(), err.to_string());

    ErrorEnvelope::new(
        ErrorCategory::Unavailable,
        "Failed to connect to control plane socket.",
    )
    .with_detail(detail)
}
