use std::fmt;

use prost::Message;
use redesmyn_protocol::ErrorEnvelope;
use redesmyn_protocol::daemon::DaemonFrame;
use redesmyn_protocol::pb::redesmyn::protocol::v1 as pbv1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecKind {
    Json,
    Protobuf,
}

pub trait Codec: Send + Sync {
    fn kind(&self) -> CodecKind;
    fn encode_frame(&self, frame: &DaemonFrame) -> Result<Vec<u8>, CodecError>;
    fn decode_frame(&self, bytes: &[u8]) -> Result<DaemonFrame, CodecError>;
}

#[derive(Debug, Clone)]
pub struct FrameDecodeError {
    pub error: ErrorEnvelope,
}

impl fmt::Display for FrameDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid frame ({}): {}",
            self.error.category, self.error.message
        )
    }
}

impl std::error::Error for FrameDecodeError {}

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Protobuf(#[from] prost::DecodeError),
    #[error(transparent)]
    Frame(#[from] FrameDecodeError),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct JsonCodec;

impl JsonCodec {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Codec for JsonCodec {
    fn kind(&self) -> CodecKind {
        CodecKind::Json
    }

    fn encode_frame(&self, frame: &DaemonFrame) -> Result<Vec<u8>, CodecError> {
        Ok(serde_json::to_vec(frame)?)
    }

    fn decode_frame(&self, bytes: &[u8]) -> Result<DaemonFrame, CodecError> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ProtobufCodec;

impl ProtobufCodec {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Codec for ProtobufCodec {
    fn kind(&self) -> CodecKind {
        CodecKind::Protobuf
    }

    fn encode_frame(&self, frame: &DaemonFrame) -> Result<Vec<u8>, CodecError> {
        Ok(frame.to_protobuf().encode_to_vec())
    }

    fn decode_frame(&self, bytes: &[u8]) -> Result<DaemonFrame, CodecError> {
        let proto = pbv1::DaemonFrame::decode(bytes)?;
        DaemonFrame::try_from_protobuf(proto).map_err(|error| FrameDecodeError { error }.into())
    }
}
