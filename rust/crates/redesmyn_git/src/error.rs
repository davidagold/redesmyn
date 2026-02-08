use std::path::PathBuf;
use std::time::Duration;

use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope};

#[derive(Debug, thiserror::Error)]
pub enum GitError {
    #[error("git executable not found")]
    GitNotFound,

    #[error("failed to spawn git subprocess")]
    SpawnGit {
        #[source]
        source: std::io::Error,
    },

    #[error("git IO error")]
    Io {
        #[source]
        source: std::io::Error,
    },

    #[error("git command timed out: {op}")]
    Timeout { op: &'static str, timeout: Duration },

    #[error("git command cancelled: {op}")]
    Cancelled { op: &'static str },

    #[error("not a git repository: {repo_root}")]
    NotAGitRepository { repo_root: PathBuf },

    #[error("revision not found: {rev}")]
    RevisionNotFound { rev: String },

    #[error("git command failed: {op} (exit {exit_code})")]
    CommandFailed {
        op: &'static str,
        exit_code: i32,
        stderr: String,
    },

    #[error("failed to parse git output: {op}")]
    Parse { op: &'static str, reason: String },
}

impl GitError {
    #[must_use]
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::RevisionNotFound { .. } => ErrorCategory::NotFound,
            Self::NotAGitRepository { .. } => ErrorCategory::InvalidRequest,
            Self::GitNotFound
            | Self::SpawnGit { .. }
            | Self::Io { .. }
            | Self::Timeout { .. }
            | Self::Cancelled { .. } => ErrorCategory::Unavailable,
            Self::CommandFailed { exit_code, .. } => {
                if *exit_code == 128 {
                    // Git uses 128 for many "fatal" errors (including not-a-repo).
                    ErrorCategory::InvalidRequest
                } else {
                    ErrorCategory::Internal
                }
            }
            Self::Parse { .. } => ErrorCategory::Internal,
        }
    }
}

fn truncate_detail(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in value.chars().take(max_chars) {
        out.push(ch);
    }
    if value.chars().count() > max_chars {
        out.push_str("…");
    }
    out
}

impl From<GitError> for ErrorEnvelope {
    fn from(err: GitError) -> Self {
        let category = err.category();

        match err {
            GitError::GitNotFound => ErrorEnvelope::new(
                category,
                "git executable not found (is git installed and on PATH?).",
            ),
            GitError::SpawnGit { source } => {
                ErrorEnvelope::new(category, "Failed to spawn git.").with_detail(ErrorDetail::from(
                    [("error".to_string(), source.to_string())],
                ))
            }
            GitError::Io { source } => ErrorEnvelope::new(category, "Git IO error.").with_detail(
                ErrorDetail::from([("error".to_string(), source.to_string())]),
            ),
            GitError::Timeout { op, timeout } => {
                ErrorEnvelope::new(category, "Git command timed out.").with_detail(
                    ErrorDetail::from([
                        ("op".to_string(), op.to_string()),
                        ("timeout_ms".to_string(), timeout.as_millis().to_string()),
                    ]),
                )
            }
            GitError::Cancelled { op } => ErrorEnvelope::new(category, "Git command cancelled.")
                .with_detail(ErrorDetail::from([("op".to_string(), op.to_string())])),
            GitError::NotAGitRepository { repo_root } => ErrorEnvelope::new(
                category,
                "Not a git repository (does the repo root point at a valid checkout?).",
            )
            .with_detail(ErrorDetail::from([(
                "repo_root".to_string(),
                repo_root.display().to_string(),
            )])),
            GitError::RevisionNotFound { rev } => {
                ErrorEnvelope::new(category, "Revision not found.")
                    .with_detail(ErrorDetail::from([("rev".to_string(), rev)]))
            }
            GitError::CommandFailed {
                op,
                exit_code,
                stderr,
            } => ErrorEnvelope::new(category, "Git command failed.").with_detail(
                ErrorDetail::from([
                    ("op".to_string(), op.to_string()),
                    ("exit_code".to_string(), exit_code.to_string()),
                    ("stderr".to_string(), truncate_detail(stderr.trim(), 400)),
                ]),
            ),
            GitError::Parse { op, reason } => {
                ErrorEnvelope::new(category, "Failed to parse git output.").with_detail(
                    ErrorDetail::from([
                        ("op".to_string(), op.to_string()),
                        ("reason".to_string(), truncate_detail(&reason, 400)),
                    ]),
                )
            }
        }
    }
}
