//! Typed transports and codecs for control plane ↔ daemon communication.
//!
//! Domain 0 goals:
//! - keep daemon embedding and remote connections on the same typed boundary,
//! - support a fast default codec (Protobuf) and a debuggable codec (JSON),
//! - provide a small in-proc transport with backpressure for the desktop app.

#![forbid(unsafe_code)]

use std::{future::Future, pin::Pin};

use redesmyn_protocol::daemon::{DaemonFrame, MessageEnvelope};
use tracing::field;

pub mod codec;
pub mod framed;
pub mod in_proc;
pub mod wiretap;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Codec(#[from] crate::codec::CodecError),
    #[error("transport channel closed")]
    ChannelClosed,
    #[error("frame too large: {len} > {max} bytes")]
    FrameTooLarge { len: usize, max: usize },
}

/// A bidirectional daemon protocol connection.
///
/// Implementations:
/// - [`in_proc::InProcEndpoint`]: in-memory channels, no serialization.
/// - [`framed::FramedEndpoint`]: length-delimited frames over an IO stream, codec-pluggable.
pub trait DaemonConnection: Send {
    fn send(&mut self, frame: DaemonFrame) -> BoxFuture<'_, Result<(), TransportError>>;
    fn recv(&mut self) -> BoxFuture<'_, Result<DaemonFrame, TransportError>>;
}

pub(crate) fn span_for_envelope(name: &'static str, envelope: &MessageEnvelope) -> tracing::Span {
    let span = tracing::debug_span!(
        "daemon.transport",
        op = name,
        protocol_version = envelope.protocol_version,
        msg_id = %envelope.msg_id,
        in_reply_to = field::Empty,
        command_id = field::Empty,
        run_id = field::Empty,
    );

    if let Some(id) = envelope.in_reply_to {
        span.record("in_reply_to", field::display(id));
    }
    if let Some(id) = envelope.command_id {
        span.record("command_id", field::display(id));
    }
    if let Some(id) = envelope.run_id {
        span.record("run_id", field::display(id));
    }

    span
}
