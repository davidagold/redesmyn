//! Versioned, strongly-typed message schemas for Redesmyn boundaries.
//!
//! This crate contains wire types shared across the Rust workspace, including the
//! cross-boundary error envelope used by:
//! - daemon ↔ control plane (protocol errors),
//! - client ↔ control plane APIs,
//! - CLI formatting (exit codes + user-facing messages).

use std::collections::BTreeMap;

pub use redesmyn_errors::ErrorCategory;

pub mod daemon;

/// Optional structured detail for debugging/UX (no stack traces).
///
/// This is intentionally simple for now (stringly-typed map) and can evolve as
/// richer typed detail payloads become needed.
pub type ErrorDetail = BTreeMap<String, String>;

/// Minimal structured error type that can cross boundaries (protocol/API).
///
/// Conventions:
/// - `message` must be user-actionable and non-noisy.
/// - `detail` should avoid stack traces and large payloads.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ErrorEnvelope {
    pub category: ErrorCategory,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ErrorDetail>,
}

impl ErrorEnvelope {
    pub fn new(category: ErrorCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
            detail: None,
        }
    }

    pub fn with_detail(mut self, detail: ErrorDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    pub fn exit_code(&self) -> i32 {
        self.category.exit_code()
    }

    pub fn http_status(&self) -> u16 {
        self.category.http_status()
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorCategory, ErrorEnvelope};

    #[test]
    fn error_envelope_serializes_with_expected_shape() {
        let envelope = ErrorEnvelope::new(ErrorCategory::NotFound, "Task not found");
        let json = serde_json::to_string(&envelope).unwrap();
        assert_eq!(
            json,
            r#"{"category":"not_found","message":"Task not found"}"#
        );
    }
}
