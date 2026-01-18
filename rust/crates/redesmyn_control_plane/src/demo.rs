use crate::error::ControlPlaneError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSummary {
    pub task_id: String,
}

pub fn get_task(task_id: &str) -> Result<TaskSummary, ControlPlaneError> {
    if task_id.trim().is_empty() {
        return Err(ControlPlaneError::InvalidTaskId {
            task_id: task_id.to_string(),
        });
    }

    // Stubbed behavior used to demonstrate cross-boundary error mapping.
    if task_id == "T-404" {
        return Err(ControlPlaneError::TaskNotFound {
            task_id: task_id.to_string(),
        });
    }

    Ok(TaskSummary {
        task_id: task_id.to_string(),
    })
}
