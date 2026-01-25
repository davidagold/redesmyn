#[derive(Debug, thiserror::Error)]
pub(crate) enum FramingError {
    #[error("missing Content-Length header")]
    MissingContentLength,
    #[error("invalid Content-Length value")]
    InvalidContentLength,
    #[error("header bytes are not valid ascii")]
    NonAsciiHeader,
}

#[derive(Debug)]
pub(crate) struct ContentLengthFramingDecoder {
    buffer: Vec<u8>,
    expected_len: Option<usize>,
}

impl ContentLengthFramingDecoder {
    pub(crate) fn new() -> Self {
        Self {
            buffer: Vec::new(),
            expected_len: None,
        }
    }

    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<Vec<u8>>, FramingError> {
        self.buffer.extend_from_slice(chunk);
        self.drain_frames()
    }

    fn drain_frames(&mut self) -> Result<Vec<Vec<u8>>, FramingError> {
        let mut out = Vec::new();
        loop {
            if self.expected_len.is_none() {
                let Some((header_end, consumed)) = find_header_terminator(&self.buffer) else {
                    break;
                };

                let header_bytes = &self.buffer[..header_end];
                let header_str =
                    std::str::from_utf8(header_bytes).map_err(|_| FramingError::NonAsciiHeader)?;
                let len = parse_content_length(header_str)?;
                self.expected_len = Some(len);
                self.buffer.drain(..consumed);
            }

            let Some(expected) = self.expected_len else {
                break;
            };
            if self.buffer.len() < expected {
                break;
            }

            let payload = self.buffer.drain(..expected).collect::<Vec<u8>>();
            self.expected_len = None;
            out.push(payload);
        }

        Ok(out)
    }
}

fn find_header_terminator(buf: &[u8]) -> Option<(usize, usize)> {
    // Prefer strict LSP `\r\n\r\n`.
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some((pos, pos + 4));
    }
    // Accept `\n\n` as a fallback.
    if let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
        return Some((pos, pos + 2));
    }
    None
}

fn parse_content_length(headers: &str) -> Result<usize, FramingError> {
    for line in headers.lines() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        if k.trim().eq_ignore_ascii_case("Content-Length") {
            let value = v.trim();
            let len = value
                .parse::<usize>()
                .map_err(|_| FramingError::InvalidContentLength)?;
            return Ok(len);
        }
    }
    Err(FramingError::MissingContentLength)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_length_decoder_handles_chunked_reads() {
        let payload = br#"{"jsonrpc":"2.0","method":"ping","id":1}"#;
        let framed = format!("Content-Length: {}\r\n\r\n", payload.len());
        let mut bytes = framed.into_bytes();
        bytes.extend_from_slice(payload);

        let mut decoder = ContentLengthFramingDecoder::new();
        let mut out = Vec::new();

        for chunk in bytes.chunks(3) {
            let frames = decoder.push(chunk).expect("push");
            out.extend(frames);
        }

        assert_eq!(out.len(), 1);
        assert_eq!(out[0], payload);
    }
}
