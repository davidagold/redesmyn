use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use redesmyn_protocol::daemon::DaemonFrame;

use crate::codec::Codec;
use crate::{BoxFuture, DaemonConnection, TransportError};

pub const DEFAULT_MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

#[derive(Debug)]
pub struct FramedEndpoint<C, S> {
    codec: C,
    read: tokio::io::ReadHalf<S>,
    write: tokio::io::WriteHalf<S>,
    max_frame_len: usize,
}

impl<C, S> FramedEndpoint<C, S>
where
    C: Codec,
    S: AsyncRead + AsyncWrite,
{
    pub fn new(stream: S, codec: C) -> Self {
        let (read, write) = tokio::io::split(stream);
        Self {
            codec,
            read,
            write,
            max_frame_len: DEFAULT_MAX_FRAME_LEN,
        }
    }

    #[must_use]
    pub fn with_max_frame_len(mut self, max_frame_len: usize) -> Self {
        self.max_frame_len = max_frame_len;
        self
    }

    pub async fn send_frame(&mut self, frame: DaemonFrame) -> Result<(), TransportError> {
        let span = crate::span_for_envelope("daemon.framed.send", &frame.envelope);
        let _enter = span.enter();

        let bytes = self.codec.encode_frame(&frame)?;
        let len = bytes.len();
        if len > self.max_frame_len || len > u32::MAX as usize {
            return Err(TransportError::FrameTooLarge {
                len,
                max: self.max_frame_len,
            });
        }

        self.write.write_all(&(len as u32).to_be_bytes()).await?;
        self.write.write_all(&bytes).await?;
        self.write.flush().await?;
        Ok(())
    }

    pub async fn recv_frame(&mut self) -> Result<DaemonFrame, TransportError> {
        let mut len_buf = [0_u8; 4];
        self.read.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;
        if len > self.max_frame_len {
            return Err(TransportError::FrameTooLarge {
                len,
                max: self.max_frame_len,
            });
        }

        let mut bytes = vec![0_u8; len];
        self.read.read_exact(&mut bytes).await?;
        let frame = self.codec.decode_frame(&bytes)?;

        let span = crate::span_for_envelope("daemon.framed.recv", &frame.envelope);
        let _enter = span.enter();
        Ok(frame)
    }
}

impl<C, S> DaemonConnection for FramedEndpoint<C, S>
where
    C: Codec + 'static,
    S: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    fn send(&mut self, frame: DaemonFrame) -> BoxFuture<'_, Result<(), TransportError>> {
        Box::pin(async move { self.send_frame(frame).await })
    }

    fn recv(&mut self) -> BoxFuture<'_, Result<DaemonFrame, TransportError>> {
        Box::pin(async move { self.recv_frame().await })
    }
}
