use std::io::{Read as _, Write as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::thread;

use prost::Message as _;
use tokio::sync::{mpsc, oneshot};

use redesmyn_logging::tracing;
use redesmyn_protocol::ProtocolEnvelope;
use redesmyn_protocol::pb::redesmyn::protocol::v1 as pbv1;
use redesmyn_protocol::ui_driver::{
    UiDriverFrame, UiDriverMessage, UiDriverRequest, UiDriverResponse,
};

pub const DEFAULT_MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub struct UiDriverCommand {
    pub envelope: ProtocolEnvelope,
    pub request: UiDriverRequest,
    pub respond_to: oneshot::Sender<UiDriverResponse>,
}

#[derive(Debug, Clone)]
pub struct UiDriverServerConfig {
    pub socket_path: PathBuf,
    pub max_frame_len: usize,
}

#[derive(Debug)]
pub struct UiDriverServerHandle {
    _thread: thread::JoinHandle<()>,
}

#[derive(Debug, thiserror::Error)]
pub enum UiDriverStartError {
    #[error("failed to bind ui driver socket at {socket_path}: {source}")]
    Bind {
        socket_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub fn start_ui_driver_server(
    config: UiDriverServerConfig,
    tx: mpsc::UnboundedSender<UiDriverCommand>,
) -> Result<UiDriverServerHandle, UiDriverStartError> {
    if config.socket_path.exists() {
        let _ = std::fs::remove_file(&config.socket_path);
    }

    let listener =
        UnixListener::bind(&config.socket_path).map_err(|source| UiDriverStartError::Bind {
            socket_path: config.socket_path.clone(),
            source,
        })?;

    tracing::info!(
        socket_path = %config.socket_path.display(),
        "ui driver server listening"
    );

    let thread = thread::spawn(move || {
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(stream) => stream,
                Err(err) => {
                    tracing::warn!(error = %err, "ui driver accept failed");
                    continue;
                }
            };

            let task_tx = tx.clone();
            let max_frame_len = config.max_frame_len;
            thread::spawn(move || {
                if let Err(err) = handle_connection(stream, max_frame_len, task_tx) {
                    tracing::warn!(error = %err, "ui driver connection closed with error");
                }
            });
        }
    });

    Ok(UiDriverServerHandle { _thread: thread })
}

#[derive(Debug, thiserror::Error)]
enum UiDriverConnectionError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Decode(#[from] prost::DecodeError),
    #[error("invalid protocol frame: {0}")]
    Frame(String),
    #[error("ui driver command channel closed")]
    ChannelClosed,
}

fn handle_connection(
    mut stream: UnixStream,
    max_frame_len: usize,
    tx: mpsc::UnboundedSender<UiDriverCommand>,
) -> Result<(), UiDriverConnectionError> {
    loop {
        let Some(bytes) = read_framed_bytes(&mut stream, max_frame_len)? else {
            return Ok(());
        };

        let proto = pbv1::UiDriverFrame::decode(bytes.as_slice())?;
        let frame = UiDriverFrame::try_from_protobuf(proto).map_err(|err| {
            UiDriverConnectionError::Frame(format!("{:?}: {}", err.category, err.message))
        })?;

        let UiDriverMessage::Request(request) = frame.message else {
            continue;
        };

        let (respond_to, rx) = oneshot::channel();
        tx.send(UiDriverCommand {
            envelope: frame.envelope,
            request,
            respond_to,
        })
        .map_err(|_| UiDriverConnectionError::ChannelClosed)?;

        let response = rx
            .blocking_recv()
            .map_err(|_| UiDriverConnectionError::ChannelClosed)?;

        let response_frame =
            UiDriverFrame::new(ProtocolEnvelope::new(), UiDriverMessage::Response(response));
        let out = response_frame.to_protobuf().encode_to_vec();
        write_framed_bytes(&mut stream, &out, max_frame_len)?;
    }
}

fn read_framed_bytes(
    stream: &mut UnixStream,
    max_frame_len: usize,
) -> Result<Option<Vec<u8>>, std::io::Error> {
    let mut len_buf = [0_u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > max_frame_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {len} > {max_frame_len} bytes"),
        ));
    }

    let mut bytes = vec![0_u8; len];
    stream.read_exact(&mut bytes)?;
    Ok(Some(bytes))
}

fn write_framed_bytes(
    stream: &mut UnixStream,
    bytes: &[u8],
    max_frame_len: usize,
) -> Result<(), std::io::Error> {
    let len = bytes.len();
    if len > max_frame_len || len > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {len} > {max_frame_len} bytes"),
        ));
    }

    stream.write_all(&(len as u32).to_be_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()?;
    Ok(())
}
