use std::io;
use std::path::PathBuf;

use redesmyn_ids::{HostId, HostInstanceId, RepoId, WorkspaceId};
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope, RepoScope, Timestamp};

mod attachment_manager;
mod identity;
mod instance_lock;
mod registry;

#[cfg(test)]
mod tests;

pub use attachment_manager::{AttachedRepoRoots, RepoAttachmentManager};
pub use identity::{GitRepoIdentityResolver, RepoIdentityResolver};
pub use registry::{FileRepoRegistry, RepoRegistry, RepoRegistryError};

pub const CONFLICT_CODE_KEY: &str = "conflict_code";
pub const CONFLICT_CODE_REPO_INSTANCE_BUSY: &str = "repo_instance_busy";
pub const CONFLICT_CODE_REPO_IDENTITY_MISMATCH: &str = "repo_identity_mismatch";

#[derive(Debug, Clone)]
pub struct RepoRegistrationRequest {
    pub scope: RepoScope,
    pub repo_root: PathBuf,
    pub display_name: Option<String>,
    pub trusted: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RepoInstanceOwner {
    pub host_id: HostId,
    pub host_instance_id: HostInstanceId,
    pub pid: u32,
    pub acquired_at: Timestamp,
}

#[derive(Debug, thiserror::Error)]
pub enum RepoAttachError {
    #[error("daemon runtime is unavailable")]
    DaemonUnavailable,
    #[error(transparent)]
    Registry(#[from] RepoRegistryError),
    #[error("registered repo path is invalid: {repo_root}")]
    InvalidRepoPath { repo_root: PathBuf },
    #[error("registered repo is not a git repository: {repo_root}")]
    NotGitRepository { repo_root: PathBuf },
    #[error(
        "registered repo identity mismatch for workspace_id={workspace_id} repo_id={repo_id}: expected={expected_identity}, actual={actual_identity}"
    )]
    IdentityMismatch {
        workspace_id: WorkspaceId,
        repo_id: RepoId,
        expected_identity: String,
        actual_identity: String,
    },
    #[error("repo instance is busy: {lock_path}")]
    RepoInstanceBusy {
        lock_path: PathBuf,
        owner: Option<RepoInstanceOwner>,
    },
    #[error("failed to attach repo: {context}: {source}")]
    Io {
        context: &'static str,
        #[source]
        source: io::Error,
    },
}

impl RepoAttachError {
    #[must_use]
    pub fn as_error_envelope(&self) -> ErrorEnvelope {
        match self {
            Self::RepoInstanceBusy { lock_path, owner } => {
                let mut detail = ErrorDetail::from([
                    (
                        CONFLICT_CODE_KEY.to_string(),
                        CONFLICT_CODE_REPO_INSTANCE_BUSY.to_string(),
                    ),
                    ("lock_path".to_string(), lock_path.display().to_string()),
                ]);
                if let Some(owner) = owner {
                    detail.insert("owner_host_id".to_string(), owner.host_id.to_string());
                    detail.insert(
                        "owner_host_instance_id".to_string(),
                        owner.host_instance_id.to_string(),
                    );
                    detail.insert("owner_pid".to_string(), owner.pid.to_string());
                    detail.insert(
                        "owner_acquired_at".to_string(),
                        owner.acquired_at.into_offset_date_time().to_string(),
                    );
                }
                ErrorEnvelope::new(
                    ErrorCategory::Conflict,
                    "Repo instance is already attached by another daemon.",
                )
                .with_detail(detail)
            }
            Self::IdentityMismatch {
                workspace_id,
                repo_id,
                expected_identity,
                actual_identity,
                ..
            } => ErrorEnvelope::new(
                ErrorCategory::Conflict,
                "Registered repo identity does not match the local repository.",
            )
            .with_detail(ErrorDetail::from([
                (
                    CONFLICT_CODE_KEY.to_string(),
                    CONFLICT_CODE_REPO_IDENTITY_MISMATCH.to_string(),
                ),
                (
                    "expected_repo_identity".to_string(),
                    expected_identity.to_string(),
                ),
                (
                    "actual_repo_identity".to_string(),
                    actual_identity.to_string(),
                ),
                ("workspace_id".to_string(), workspace_id.to_string()),
                ("repo_id".to_string(), repo_id.to_string()),
            ])),
            Self::Registry(RepoRegistryError::ScopeNotRegistered {
                workspace_id,
                repo_id,
            }) => ErrorEnvelope::new(
                ErrorCategory::NotFound,
                "Repo scope is not registered on this daemon.",
            )
            .with_detail(ErrorDetail::from([
                ("workspace_id".to_string(), workspace_id.to_string()),
                ("repo_id".to_string(), repo_id.to_string()),
            ])),
            Self::Registry(err) => ErrorEnvelope::new(ErrorCategory::Unavailable, err.to_string()),
            Self::InvalidRepoPath { repo_root } => ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Registered repo path is invalid.",
            )
            .with_detail(ErrorDetail::from([(
                "repo_root".to_string(),
                repo_root.display().to_string(),
            )])),
            Self::NotGitRepository { repo_root } => ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Registered repo path is not a git repository.",
            )
            .with_detail(ErrorDetail::from([(
                "repo_root".to_string(),
                repo_root.display().to_string(),
            )])),
            Self::DaemonUnavailable => {
                ErrorEnvelope::new(ErrorCategory::Unavailable, "Daemon runtime is unavailable.")
            }
            Self::Io { .. } => ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Repo attach failed due to local I/O error.",
            ),
        }
    }
}

impl From<RepoAttachError> for ErrorEnvelope {
    fn from(value: RepoAttachError) -> Self {
        value.as_error_envelope()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RepoRegisterError {
    #[error("daemon runtime is unavailable")]
    DaemonUnavailable,
    #[error(transparent)]
    Registry(#[from] RepoRegistryError),
    #[error("repo path is invalid: {repo_root}")]
    InvalidRepoPath { repo_root: PathBuf },
    #[error("repo path is not a git repository: {repo_root}")]
    NotGitRepository { repo_root: PathBuf },
    #[error("failed to register repo: {context}: {source}")]
    Io {
        context: &'static str,
        #[source]
        source: io::Error,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum RepoDetachError {
    #[error("daemon runtime is unavailable")]
    DaemonUnavailable,
}
