use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::sync::watch;

use crate::{BoxFuture, GitError, GitOid, GitRevision, GitWorktree, GitWorktreeTarget};

/// Options that apply to all backend operations: timeouts and cancellation.
#[derive(Debug, Clone, Default)]
pub struct GitRunOptions {
    pub timeout: Option<Duration>,
    pub cancel: Option<watch::Receiver<bool>>,
}

impl GitRunOptions {
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    #[must_use]
    pub fn with_cancel(mut self, cancel: watch::Receiver<bool>) -> Self {
        self.cancel = Some(cancel);
        self
    }
}

#[derive(Debug, Clone)]
pub struct GitWorktreeAddOptions {
    pub force: bool,
    pub detach: bool,
    pub checkout: bool,
    /// Create a new branch with `-b` (or reset with `-B` when `reset_branch=true`).
    pub new_branch: Option<String>,
    pub reset_branch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitRemoteUrl {
    pub name: String,
    pub url: String,
}

impl Default for GitWorktreeAddOptions {
    fn default() -> Self {
        Self {
            force: false,
            detach: false,
            checkout: true,
            new_branch: None,
            reset_branch: false,
        }
    }
}

/// Git operations required by daemon-side subsystems.
///
/// This trait is intentionally small and semantic (not "run arbitrary git
/// commands") so we can later swap implementations (e.g. a Rust-native backend)
/// without rewriting higher-level logic.
pub trait GitBackend: Send + Sync + 'static {
    /// Resolve the repository's absolute `.git` directory path.
    fn absolute_git_dir<'a>(
        &'a self,
        repo_root: &'a Path,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<PathBuf, GitError>>;

    /// Read a git config value by key, returning `None` when the key is unset.
    fn config_get<'a>(
        &'a self,
        repo_root: &'a Path,
        key: &'a str,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<Option<String>, GitError>>;

    /// List configured remotes in command output order.
    fn remote_urls<'a>(
        &'a self,
        repo_root: &'a Path,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<Vec<GitRemoteUrl>, GitError>>;

    /// Resolve each rev to a commit id, returning `None` for revs that do not resolve.
    ///
    /// Implementations should prefer batching (one process invocation, or one
    /// repo scan) when feasible.
    fn resolve_commits<'a>(
        &'a self,
        repo_root: &'a Path,
        revs: &'a [GitRevision],
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<Vec<Option<GitOid>>, GitError>>;

    /// Resolve a single rev to a commit id. Returns a typed `not_found` error on failure.
    fn resolve_commit<'a>(
        &'a self,
        repo_root: &'a Path,
        rev: &'a GitRevision,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<GitOid, GitError>> {
        Box::pin(async move {
            let out = self
                .resolve_commits(repo_root, &[rev.clone()], options)
                .await?;
            match out.into_iter().next().unwrap_or(None) {
                Some(oid) => Ok(oid),
                None => Err(GitError::RevisionNotFound {
                    rev: rev.to_string(),
                }),
            }
        })
    }

    /// Compute the merge base between two commits.
    ///
    /// Returns `None` when no merge base exists.
    fn merge_base<'a>(
        &'a self,
        repo_root: &'a Path,
        a: &'a GitOid,
        b: &'a GitOid,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<Option<GitOid>, GitError>>;

    /// Returns `true` if `ancestor` is an ancestor of `descendant`.
    fn is_ancestor<'a>(
        &'a self,
        repo_root: &'a Path,
        ancestor: &'a GitOid,
        descendant: &'a GitOid,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<bool, GitError>>;

    /// List repo worktrees via a stable machine-readable output form.
    fn worktree_list<'a>(
        &'a self,
        repo_root: &'a Path,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<Vec<GitWorktree>, GitError>>;

    fn worktree_add<'a>(
        &'a self,
        repo_root: &'a Path,
        path: &'a Path,
        target: &'a GitWorktreeTarget,
        add_options: GitWorktreeAddOptions,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<(), GitError>>;

    fn worktree_remove<'a>(
        &'a self,
        repo_root: &'a Path,
        path: &'a Path,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<(), GitError>>;

    fn worktree_prune<'a>(
        &'a self,
        repo_root: &'a Path,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<(), GitError>>;
}
