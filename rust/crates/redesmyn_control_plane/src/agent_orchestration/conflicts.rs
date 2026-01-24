use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope};

pub(crate) const CONFLICT_CODE_KEY: &str = "conflict_code";
pub(crate) const CONFLICT_CODE_TURN_IN_PROGRESS: &str = "structured_turn_in_progress";
pub(crate) const CONFLICT_CODE_SESSION_CONFLICT: &str = "structured_session_conflict";

pub(crate) fn conflict_envelope(code: &str, message: &str) -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCategory::Conflict, message).with_detail(ErrorDetail::from([(
        CONFLICT_CODE_KEY.to_string(),
        code.to_string(),
    )]))
}

pub(crate) fn invalid_request(message: &str) -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCategory::InvalidRequest, message)
}
