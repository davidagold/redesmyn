//! Stable span keys and helpers for consistent tagging across the workspace.

use std::fmt;

use tracing::Span;

/// Stable span key: `workspace_id`
pub const WORKSPACE_ID: &str = "workspace_id";
/// Stable span key: `repo_id`
pub const REPO_ID: &str = "repo_id";
/// Stable span key: `epic_id`
pub const EPIC_ID: &str = "epic_id";
/// Stable span key: `task_id`
pub const TASK_ID: &str = "task_id";
/// Stable span key: `host_id`
pub const HOST_ID: &str = "host_id";
/// Stable span key: `run_id`
pub const RUN_ID: &str = "run_id";
/// Stable span key: `command_id`
pub const COMMAND_ID: &str = "command_id";

/// Create an `info` span with Redesmyn's stable span keys declared as fields.
///
/// The span name must be a string literal (a `tracing` constraint).
#[macro_export]
macro_rules! redesmyn_info_span {
    ($name:literal) => {
        $crate::tracing::info_span!(
            $name,
            workspace_id = $crate::tracing::field::Empty,
            repo_id = $crate::tracing::field::Empty,
            epic_id = $crate::tracing::field::Empty,
            task_id = $crate::tracing::field::Empty,
            host_id = $crate::tracing::field::Empty,
            run_id = $crate::tracing::field::Empty,
            command_id = $crate::tracing::field::Empty,
        )
    };
    ($name:literal, $($field:tt)*) => {
        $crate::tracing::info_span!(
            $name,
            workspace_id = $crate::tracing::field::Empty,
            repo_id = $crate::tracing::field::Empty,
            epic_id = $crate::tracing::field::Empty,
            task_id = $crate::tracing::field::Empty,
            host_id = $crate::tracing::field::Empty,
            run_id = $crate::tracing::field::Empty,
            command_id = $crate::tracing::field::Empty,
            $($field)*
        )
    };
}

/// Create an `info` span for a user-triggered mutation.
///
/// `command_id` is treated as an idempotency key across:
/// - logs/traces,
/// - protocol envelopes, and
/// - UI progress state.
#[macro_export]
macro_rules! redesmyn_command_span {
    ($name:literal, $command_id:expr) => {
        $crate::tracing::info_span!(
            $name,
            workspace_id = $crate::tracing::field::Empty,
            repo_id = $crate::tracing::field::Empty,
            epic_id = $crate::tracing::field::Empty,
            task_id = $crate::tracing::field::Empty,
            host_id = $crate::tracing::field::Empty,
            run_id = $crate::tracing::field::Empty,
            command_id = %$command_id,
        )
    };
    ($name:literal, $command_id:expr, $($field:tt)*) => {
        $crate::tracing::info_span!(
            $name,
            workspace_id = $crate::tracing::field::Empty,
            repo_id = $crate::tracing::field::Empty,
            epic_id = $crate::tracing::field::Empty,
            task_id = $crate::tracing::field::Empty,
            host_id = $crate::tracing::field::Empty,
            run_id = $crate::tracing::field::Empty,
            command_id = %$command_id,
            $($field)*
        )
    };
}

pub fn record_workspace_id(span: &Span, workspace_id: impl fmt::Display) {
    span.record(WORKSPACE_ID, tracing::field::display(workspace_id));
}

pub fn record_repo_id(span: &Span, repo_id: impl fmt::Display) {
    span.record(REPO_ID, tracing::field::display(repo_id));
}

pub fn record_epic_id(span: &Span, epic_id: impl fmt::Display) {
    span.record(EPIC_ID, tracing::field::display(epic_id));
}

pub fn record_task_id(span: &Span, task_id: impl fmt::Display) {
    span.record(TASK_ID, tracing::field::display(task_id));
}

pub fn record_host_id(span: &Span, host_id: impl fmt::Display) {
    span.record(HOST_ID, tracing::field::display(host_id));
}

pub fn record_run_id(span: &Span, run_id: impl fmt::Display) {
    span.record(RUN_ID, tracing::field::display(run_id));
}

pub fn record_command_id(span: &Span, command_id: impl fmt::Display) {
    span.record(COMMAND_ID, tracing::field::display(command_id));
}
