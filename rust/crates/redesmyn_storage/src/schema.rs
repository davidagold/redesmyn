//! Typed values for schema-constrained columns.
//!
//! SQLite stores these as `TEXT` with `CHECK (...)` constraints. We mirror the
//! allowed values as Rust enums to avoid stringly-typed typos in queries.

#![allow(clippy::module_name_repetitions)]

use std::fmt;

/// Scope kind for tables that use `scope_kind` + `scope_workspace_id` + `scope_repo_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoScopeKind {
    None,
    Repo,
}

impl RepoScopeKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Repo => "repo",
        }
    }
}

impl fmt::Display for RepoScopeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandState {
    Queued,
    Accepted,
    Running,
    Blocked,
    Resumable,
    Succeeded,
    Failed,
    Canceled,
}

impl CommandState {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Accepted => "accepted",
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::Resumable => "resumable",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Canceled => "canceled",
        }
    }
}

impl fmt::Display for CommandState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeReadiness {
    Unknown,
    Ready,
    Blocked,
}

impl MergeReadiness {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Ready => "ready",
            Self::Blocked => "blocked",
        }
    }
}

impl fmt::Display for MergeReadiness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Todo,
    InProgress,
    Blocked,
    Done,
}

impl TaskState {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Todo => "todo",
            Self::InProgress => "in_progress",
            Self::Blocked => "blocked",
            Self::Done => "done",
        }
    }
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskRelationKind {
    After,
}

impl TaskRelationKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::After => "after",
        }
    }
}

impl fmt::Display for TaskRelationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Scope kind for `session_events`, which can be repo/epic/task scoped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionScopeKind {
    None,
    Repo,
    Epic,
    Task,
}

impl SessionScopeKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Repo => "repo",
            Self::Epic => "epic",
            Self::Task => "task",
        }
    }
}

impl fmt::Display for SessionScopeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Scope kind for `agent_sessions` (session identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentSessionScopeKind {
    Task,
    Chat,
}

impl AgentSessionScopeKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Task => "task",
            Self::Chat => "chat",
        }
    }
}

impl fmt::Display for AgentSessionScopeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Status for rows in `agent_sessions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentSessionStatus {
    Running,
    Blocked,
    Stopped,
    Error,
}

impl AgentSessionStatus {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::Stopped => "stopped",
            Self::Error => "error",
        }
    }
}

impl fmt::Display for AgentSessionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Agent provider persisted on `agent_sessions.agent_kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    Codex,
    ClaudeCode,
    Shell,
}

impl AgentKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::ClaudeCode => "claude_code",
            Self::Shell => "shell",
        }
    }
}

impl fmt::Display for AgentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
