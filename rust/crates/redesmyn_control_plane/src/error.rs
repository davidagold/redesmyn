use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ControlPlaneError {
    #[error("Invalid task id.")]
    InvalidTaskId { task_id: String },
    #[error("Task not found.")]
    TaskNotFound { task_id: String },
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
        }
    }
}
