//! Shared, cross-boundary error taxonomy and conventions.
//!
//! This crate defines:
//! - a small, stable error category enum (`ErrorCategory`),
//! - stable mappings to HTTP status codes and CLI exit codes.
//!
//! The wire-level error envelope lives in `redesmyn_protocol`.

use std::fmt;

/// Stable error categories used across boundaries (protocol/API/CLI).
///
/// Serialize/deserialize as `snake_case` strings to keep the wire format stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    InvalidRequest,
    NotFound,
    Conflict,
    Unauthorized,
    Unavailable,
    Internal,
}

impl ErrorCategory {
    /// Stable, user-facing exit codes for CLI/binaries.
    ///
    /// Conventions:
    /// - `0`: success (not represented here)
    /// - `1`: internal error (unexpected)
    /// - `2`: invalid request / usage error
    /// - `3`: not found
    /// - `4`: conflict
    /// - `5`: unauthorized
    /// - `6`: unavailable / temporarily down
    pub const fn exit_code(self) -> i32 {
        match self {
            Self::Internal => 1,
            Self::InvalidRequest => 2,
            Self::NotFound => 3,
            Self::Conflict => 4,
            Self::Unauthorized => 5,
            Self::Unavailable => 6,
        }
    }

    /// Stable mapping to HTTP status codes for API responses.
    pub const fn http_status(self) -> u16 {
        match self {
            Self::InvalidRequest => 400,
            Self::Unauthorized => 401,
            Self::NotFound => 404,
            Self::Conflict => 409,
            Self::Unavailable => 503,
            Self::Internal => 500,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Unauthorized => "unauthorized",
            Self::Unavailable => "unavailable",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::ErrorCategory;

    #[test]
    fn exit_code_mapping_is_stable() {
        assert_eq!(ErrorCategory::Internal.exit_code(), 1);
        assert_eq!(ErrorCategory::InvalidRequest.exit_code(), 2);
        assert_eq!(ErrorCategory::NotFound.exit_code(), 3);
        assert_eq!(ErrorCategory::Conflict.exit_code(), 4);
        assert_eq!(ErrorCategory::Unauthorized.exit_code(), 5);
        assert_eq!(ErrorCategory::Unavailable.exit_code(), 6);
    }

    #[test]
    fn http_status_mapping_is_stable() {
        assert_eq!(ErrorCategory::InvalidRequest.http_status(), 400);
        assert_eq!(ErrorCategory::Unauthorized.http_status(), 401);
        assert_eq!(ErrorCategory::NotFound.http_status(), 404);
        assert_eq!(ErrorCategory::Conflict.http_status(), 409);
        assert_eq!(ErrorCategory::Unavailable.http_status(), 503);
        assert_eq!(ErrorCategory::Internal.http_status(), 500);
    }

    #[test]
    fn serde_serializes_as_snake_case() {
        let json = serde_json::to_string(&ErrorCategory::InvalidRequest).unwrap();
        assert_eq!(json, "\"invalid_request\"");
    }
}
