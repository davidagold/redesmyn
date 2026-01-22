use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ControlPlaneError {
    #[error("Invalid task id.")]
    InvalidTaskId { task_id: String },
    #[error("Task not found.")]
    TaskNotFound { task_id: String },
    #[error("Invalid epic slug.")]
    InvalidEpicSlug { epic_slug: String },
    #[error("Epic not found.")]
    EpicNotFound { epic_slug: String },
    #[error("Invalid timestamp.")]
    InvalidTimestamp { unix_ms: i64 },
    #[error("Storage error.")]
    Storage(#[from] redesmyn_storage::StorageError),
}

impl From<ControlPlaneError> for ErrorEnvelope {
    fn from(err: ControlPlaneError) -> Self {
        match err {
            ControlPlaneError::InvalidTaskId { task_id } => {
                let detail = ErrorDetail::from([("task_id".to_string(), task_id)]);
                ErrorEnvelope::new(ErrorCategory::InvalidRequest, "Invalid task id.")
                    .with_detail(detail)
            }
            ControlPlaneError::TaskNotFound { task_id } => {
                let detail = ErrorDetail::from([("task_id".to_string(), task_id)]);
                ErrorEnvelope::new(ErrorCategory::NotFound, "Task not found.").with_detail(detail)
            }
            ControlPlaneError::InvalidEpicSlug { epic_slug } => {
                let detail = ErrorDetail::from([("epic_slug".to_string(), epic_slug)]);
                ErrorEnvelope::new(ErrorCategory::InvalidRequest, "Invalid epic slug.")
                    .with_detail(detail)
            }
            ControlPlaneError::EpicNotFound { epic_slug } => {
                let detail = ErrorDetail::from([("epic_slug".to_string(), epic_slug)]);
                ErrorEnvelope::new(ErrorCategory::NotFound, "Epic not found.").with_detail(detail)
            }
            ControlPlaneError::InvalidTimestamp { unix_ms } => {
                let detail = ErrorDetail::from([("unix_ms".to_string(), unix_ms.to_string())]);
                ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "Invalid timestamp in persisted data.",
                )
                .with_detail(detail)
            }
            ControlPlaneError::Storage(err) => {
                ErrorEnvelope::new(ErrorCategory::Internal, "Storage error.")
                    .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
            }
        }
    }
}
