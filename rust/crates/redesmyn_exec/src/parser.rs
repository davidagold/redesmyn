use redesmyn_protocol::session::SessionEventKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputStream {
    Stdout,
    Stderr,
}

/// Incremental output parser for exec-session output (T-35).
///
/// Parsers are expected to be:
/// - deterministic (pure state machine),
/// - incremental (consume chunks),
/// - and bounded (avoid unbounded buffering).
pub trait SessionOutputParser: Send {
    fn push_chunk(&mut self, stream: OutputStream, chunk: &str) -> Vec<SessionEventKind>;

    fn flush(&mut self) -> Vec<SessionEventKind> {
        Vec::new()
    }
}
