//! Typed transports and codecs for client ↔ control plane API communication (T-12).

use redesmyn_logging::tracing::{self, field};
use redesmyn_protocol::client::{ClientFrame, ClientMessage};

use crate::BoxFuture;

pub mod codec;
pub mod framed;
pub mod in_proc;
pub mod wiretap;

#[derive(Debug, thiserror::Error)]
pub enum ClientTransportError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Codec(#[from] crate::client::codec::CodecError),
    #[error("transport channel closed")]
    ChannelClosed,
    #[error("frame too large: {len} > {max} bytes")]
    FrameTooLarge { len: usize, max: usize },
}

/// A bidirectional client API protocol connection.
///
/// Implementations:
/// - [`in_proc::InProcEndpoint`]: in-memory channels, no serialization.
/// - [`framed::FramedEndpoint`]: length-delimited frames over an IO stream, codec-pluggable.
pub trait ClientConnection: Send {
    fn send(&mut self, frame: ClientFrame) -> BoxFuture<'_, Result<(), ClientTransportError>>;
    fn recv(&mut self) -> BoxFuture<'_, Result<ClientFrame, ClientTransportError>>;
}

fn map_read_exact_error(err: std::io::Error) -> ClientTransportError {
    match err.kind() {
        std::io::ErrorKind::UnexpectedEof => ClientTransportError::ChannelClosed,
        _ => ClientTransportError::Io(err),
    }
}

pub(crate) fn span_for_frame(name: &'static str, frame: &ClientFrame) -> tracing::Span {
    let envelope = &frame.envelope;
    let span = tracing::debug_span!(
        "client.transport",
        op = name,
        protocol_version = %envelope.protocol_version(),
        msg_id = %envelope.msg_id,
        correlation_id = field::Empty,
        trace_id = field::Empty,
        request_id = field::Empty,
        subscription_id = field::Empty,
    );

    if let Some(id) = envelope.correlation_id {
        span.record("correlation_id", field::display(id));
    }
    if let Some(id) = envelope.trace_id {
        span.record("trace_id", field::display(id));
    }

    match &frame.message {
        ClientMessage::Request(req) => {
            span.record("request_id", field::display(req.request_id));
        }
        ClientMessage::Response(resp) => {
            span.record("request_id", field::display(resp.request_id));
        }
        ClientMessage::Subscribe(sub) => {
            span.record("subscription_id", field::display(sub.subscription_id));
        }
        ClientMessage::Event(ev) => {
            span.record("subscription_id", field::display(ev.subscription_id));
        }
        ClientMessage::Unsubscribe(unsub) => {
            span.record("subscription_id", field::display(unsub.subscription_id));
        }
    }

    span
}
