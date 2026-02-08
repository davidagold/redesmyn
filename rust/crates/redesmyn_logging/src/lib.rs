//! Logging + tracing foundations (see `epics/gpui/tasks/T-4/README.md`).
//!
//! ## Quick start
//!
//! - Call [`init`] once early in each binary.
//! - Use `redesmyn_info_span!` / `redesmyn_command_span!` to ensure stable span keys.
//!
//! ## UI progress hooks
//!
//! Conventions:
//!
//! - Every user-triggered mutation gets a `command_id` (idempotency key).
//! - Run the mutation within a span that records `command_id` so logs, traces, and UI state can be
//!   correlated.
//!
//! See `redesmyn_command_span!`.
//!
//! ## Payload logging policy
//!
//! By default, payload contents are omitted. When you *must* log payloads (e.g. protocol frames),
//! wrap them in [`Payload`] so output follows the configured policy.
//!
//! ## Environment variables
//!
//! - `REDESMYN_LOG_FORMAT=pretty|json` (default: `pretty`)
//! - `REDESMYN_LOG` (filter directives; falls back to `RUST_LOG`)
//! - `REDESMYN_LOG_PAYLOADS=off|redact|sample[:N]|full` (default: `off`)

mod config;
mod payload;
pub mod span;
mod subscriber;

pub use crate::config::{LogFormat, LoggingConfig, PayloadPolicy};
pub use crate::payload::{Payload, payload};

/// Re-export `tracing` so callers can use macros like `redesmyn_logging::tracing::info!(...)`
/// without adding a direct dependency.
pub use tracing;

use std::io::IsTerminal as _;
use std::sync::OnceLock;

static LOGGING_CONFIG: OnceLock<LoggingConfig> = OnceLock::new();

/// Initialize tracing/logging for the current process.
///
/// This is best-effort: calling `init()` multiple times will be a no-op.
pub fn init() {
    let _ = try_init();
}

pub fn try_init() -> Result<(), tracing::dispatcher::SetGlobalDefaultError> {
    try_init_with_config(LoggingConfig::from_env())
}

pub fn try_init_with_config(
    config: LoggingConfig,
) -> Result<(), tracing::dispatcher::SetGlobalDefaultError> {
    let _ = LOGGING_CONFIG.set(config.clone());

    let use_ansi = matches!(config.format, LogFormat::Pretty) && std::io::stderr().is_terminal();
    let dispatch = crate::subscriber::build_dispatch(config, std::io::stderr, use_ansi);
    tracing::dispatcher::set_global_default(dispatch)
}

fn payload_policy() -> PayloadPolicy {
    LOGGING_CONFIG
        .get()
        .map(|config| config.payload_policy)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Clone)]
    struct SharedMakeWriter {
        buf: Arc<Mutex<Vec<u8>>>,
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedMakeWriter {
        type Writer = SharedWriter;

        fn make_writer(&'a self) -> Self::Writer {
            SharedWriter {
                buf: Arc::clone(&self.buf),
            }
        }
    }

    struct SharedWriter {
        buf: Arc<Mutex<Vec<u8>>>,
    }

    impl io::Write for SharedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.buf
                .lock()
                .expect("lock poisoned")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn stable_fields_show_up_in_pretty_logs() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let writer = SharedMakeWriter { buf: buf.clone() };
        let dispatch = crate::subscriber::build_dispatch(
            LoggingConfig {
                format: LogFormat::Pretty,
                filter_directives: Some("info".to_string()),
                payload_policy: PayloadPolicy::MetadataOnly,
            },
            writer,
            false,
        );

        tracing::dispatcher::with_default(&dispatch, || {
            let span = crate::redesmyn_command_span!("demo", "cmd_01");
            crate::span::record_workspace_id(&span, "ws_01");

            let _guard = span.enter();
            tracing::info!("hello");
        });

        let output = String::from_utf8(buf.lock().expect("lock poisoned").clone())
            .expect("log output should be utf-8");

        assert!(output.contains("demo"));
        assert!(output.contains("workspace_id=ws_01"));
        assert!(output.contains("command_id=cmd_01"));
        assert!(output.contains("hello"));
    }

    #[test]
    fn stable_fields_show_up_in_json_logs() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let writer = SharedMakeWriter { buf: buf.clone() };
        let dispatch = crate::subscriber::build_dispatch(
            LoggingConfig {
                format: LogFormat::Json,
                filter_directives: Some("info".to_string()),
                payload_policy: PayloadPolicy::MetadataOnly,
            },
            writer,
            false,
        );

        tracing::dispatcher::with_default(&dispatch, || {
            let span = crate::redesmyn_command_span!("demo", "cmd_01");
            crate::span::record_workspace_id(&span, "ws_01");

            let _guard = span.enter();
            tracing::info!("hello");
        });

        let output = String::from_utf8(buf.lock().expect("lock poisoned").clone())
            .expect("log output should be utf-8");

        assert!(output.contains("\"workspace_id\":\"ws_01\""));
        assert!(output.contains("\"command_id\":\"cmd_01\""));
        assert!(output.contains("\"message\":\"hello\""));
    }
}
