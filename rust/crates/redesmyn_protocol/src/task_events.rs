//! Event log types for task lifecycle updates.

/// A task's durable `state` field changed (e.g. `todo` -> `in_progress`).
pub const TASK_STATE_CHANGED_EVENT: &str = "task.state.changed";
