use crate::codec::{Codec, CodecError};

#[derive(Debug, thiserror::Error)]
pub enum WiretapError {
    #[error(transparent)]
    Codec(#[from] CodecError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub fn decode_frame_to_json_value(
    codec: &dyn Codec,
    bytes: &[u8],
) -> Result<serde_json::Value, WiretapError> {
    let frame = codec.decode_frame(bytes)?;
    Ok(serde_json::to_value(frame)?)
}

pub fn decode_frame_to_json_string(
    codec: &dyn Codec,
    bytes: &[u8],
) -> Result<String, WiretapError> {
    let value = decode_frame_to_json_value(codec, bytes)?;
    Ok(serde_json::to_string_pretty(&value)?)
}
