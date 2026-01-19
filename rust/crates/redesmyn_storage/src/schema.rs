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

