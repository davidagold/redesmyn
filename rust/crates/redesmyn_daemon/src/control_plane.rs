use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use redesmyn_logging::tracing;
use redesmyn_protocol::DaemonHello;
use redesmyn_protocol::daemon::{ControlPlaneHelloAck, DaemonFrame, DaemonMessage};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, ProtocolEnvelope, ProtocolVersion};
use redesmyn_transport::{DaemonConnection, TransportError};
use tokio::sync::watch;

use crate::backoff::Backoff;
use crate::capabilities::DaemonCapabilities;
use crate::host_identity::HostIdentity;

pub type ControlPlaneConnectFuture =
    Pin<Box<dyn Future<Output = Result<Box<dyn DaemonConnection>, TransportError>> + Send>>;

pub trait ControlPlaneConnector: Send + Sync + 'static {
    fn connect(&self) -> ControlPlaneConnectFuture;
}

impl<F, Fut> ControlPlaneConnector for F
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Box<dyn DaemonConnection>, TransportError>> + Send + 'static,
{
    fn connect(&self) -> ControlPlaneConnectFuture {
        Box::pin((self)())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Handshaking,
    Connected { accepted_protocol: ProtocolVersion },
    Fatal { error: ErrorEnvelope },
}

#[derive(Debug, thiserror::Error)]
enum HandshakeError {
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error("control plane rejected handshake: {0:?}")]
    Rejected(ErrorEnvelope),
    #[error("unexpected message during handshake: {0:?}")]
    UnexpectedMessage(DaemonMessage),
    #[error("invalid accepted protocol: accepted={accepted} supported={supported}")]
    InvalidAcceptedProtocol {
        accepted: ProtocolVersion,
        supported: ProtocolVersion,
    },
}

impl HandshakeError {
    fn into_envelope(&self) -> ErrorEnvelope {
        match self {
            Self::Transport(_) => {
                ErrorEnvelope::new(ErrorCategory::Unavailable, "Transport error during handshake.")
            }
            Self::Rejected(err) => err.clone(),
            Self::UnexpectedMessage(_) => ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Unexpected message during handshake.",
            ),
            Self::InvalidAcceptedProtocol { .. } => ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Control plane accepted an invalid protocol version.",
            ),
        }
    }

    fn is_fatal(&self) -> bool {
        match self {
            Self::InvalidAcceptedProtocol { .. } => true,
            Self::Rejected(err) => err.category == ErrorCategory::InvalidRequest,
            _ => false,
        }
    }
}

pub async fn run_control_plane_connection_manager(
    connector: Arc<dyn ControlPlaneConnector>,
    identity: HostIdentity,
    capabilities: DaemonCapabilities,
    supported_protocol: ProtocolVersion,
    backoff: Backoff,
    shutdown_tx: watch::Sender<bool>,
    mut shutdown_rx: watch::Receiver<bool>,
    state_tx: watch::Sender<ConnectionState>,
) {
    let span = redesmyn_logging::redesmyn_info_span!("daemon.control_plane.connection_manager");
    redesmyn_logging::span::record_host_id(&span, identity.host_id);
    redesmyn_logging::span::record_host_instance_id(&span, identity.host_instance_id);
    let _enter = span.enter();

    let mut backoff = backoff;
    let mut minor_mismatch_logged = false;

    loop {
        if *shutdown_rx.borrow() {
            let _ = state_tx.send(ConnectionState::Disconnected);
            return;
        }

        let _ = state_tx.send(ConnectionState::Connecting);
        let mut conn = tokio::select! {
            result = connector.connect() => {
                match result {
                    Ok(conn) => conn,
                    Err(err) => {
                        tracing::warn!(error = %err, "failed to connect to control plane");
                        wait_backoff_delay(&mut backoff, &mut shutdown_rx).await;
                        continue;
                    }
                }
            }
            _ = shutdown_rx.changed() => {
                let _ = state_tx.send(ConnectionState::Disconnected);
                return;
            }
        };

        let _ = state_tx.send(ConnectionState::Handshaking);
        match perform_handshake(&mut *conn, identity, capabilities, supported_protocol, &mut shutdown_rx).await {
            Ok(accepted_protocol) => {
                backoff.reset();
                if !minor_mismatch_logged && accepted_protocol.minor != supported_protocol.minor {
                    tracing::info!(
                        supported = %supported_protocol,
                        accepted = %accepted_protocol,
                        "minor protocol mismatch accepted"
                    );
                    minor_mismatch_logged = true;
                }
                let _ = state_tx.send(ConnectionState::Connected { accepted_protocol });

                match run_connected_session(&mut *conn, &mut shutdown_rx).await {
                    Ok(()) => {
                        let _ = state_tx.send(ConnectionState::Disconnected);
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "control plane session ended with error");
                        let _ = state_tx.send(ConnectionState::Disconnected);
                    }
                }

                wait_backoff_delay(&mut backoff, &mut shutdown_rx).await;
            }
            Err(err) => {
                let error = err.into_envelope();
                if err.is_fatal() {
                    tracing::error!(
                        category = %error.category,
                        message = %error.message,
                        "fatal handshake failure; shutting down daemon"
                    );
                    let _ = state_tx.send(ConnectionState::Fatal { error });
                    let _ = shutdown_tx.send(true);
                    return;
                }

                tracing::warn!(
                    category = %error.category,
                    message = %error.message,
                    "handshake failed; retrying"
                );
                let _ = state_tx.send(ConnectionState::Disconnected);
                wait_backoff_delay(&mut backoff, &mut shutdown_rx).await;
            }
        }
    }
}

async fn wait_backoff_delay(backoff: &mut Backoff, shutdown_rx: &mut watch::Receiver<bool>) {
    if *shutdown_rx.borrow() {
        return;
    }
    let delay = backoff.next_delay();
    tokio::select! {
        _ = tokio::time::sleep(delay) => {}
        _ = shutdown_rx.changed() => {}
    }
}

async fn perform_handshake(
    conn: &mut dyn DaemonConnection,
    identity: HostIdentity,
    capabilities: DaemonCapabilities,
    supported_protocol: ProtocolVersion,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> Result<ProtocolVersion, HandshakeError> {
    let hello = DaemonHello {
        host_id: identity.host_id,
        host_instance_id: identity.host_instance_id,
        capabilities: capabilities.to_wire_strings(),
        supported_protocol,
    };

    let mut envelope = ProtocolEnvelope::new();
    envelope.protocol_major = supported_protocol.major;
    envelope.protocol_minor = supported_protocol.minor;

    let correlation_id = envelope.msg_id;
    tokio::select! {
        result = conn.send(DaemonFrame::new(envelope, DaemonMessage::DaemonHello(hello))) => {
            result?;
        }
        _ = shutdown_rx.changed() => return Err(HandshakeError::Transport(TransportError::ChannelClosed)),
    }

    let frame = tokio::select! {
        result = conn.recv() => result?,
        _ = shutdown_rx.changed() => return Err(HandshakeError::Transport(TransportError::ChannelClosed)),
    };
    match frame.message {
        DaemonMessage::ControlPlaneHelloAck(ControlPlaneHelloAck {
            accepted_protocol,
            ..
        }) => {
            if accepted_protocol.major != supported_protocol.major
                || accepted_protocol.minor > supported_protocol.minor
            {
                return Err(HandshakeError::InvalidAcceptedProtocol {
                    accepted: accepted_protocol,
                    supported: supported_protocol,
                });
            }

            if frame.envelope.correlation_id.is_some_and(|id| id != correlation_id) {
                tracing::debug!(
                    expected = %correlation_id,
                    actual = ?frame.envelope.correlation_id,
                    "handshake ack correlation_id mismatch; continuing"
                );
            }

            Ok(accepted_protocol)
        }
        DaemonMessage::Error(err) => Err(HandshakeError::Rejected(err)),
        other => Err(HandshakeError::UnexpectedMessage(other)),
    }
}

async fn run_connected_session(
    conn: &mut dyn DaemonConnection,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> Result<(), TransportError> {
    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }

        tokio::select! {
            _ = shutdown_rx.changed() => {}
            frame = conn.recv() => {
                match frame {
                    Ok(_frame) => {
                        // Domain 3 skeleton: no command routing yet.
                        // Keep the connection open and drain inbound messages.
                    }
                    Err(TransportError::ChannelClosed) => return Ok(()),
                    Err(err) => return Err(err),
                }
            }
        }
    }
}
