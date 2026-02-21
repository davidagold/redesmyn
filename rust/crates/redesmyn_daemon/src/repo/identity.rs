use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use redesmyn_git::{GitBackend, GitError, GitRunOptions};

pub(crate) type RepoIdentityFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug)]
pub(crate) enum RepoLocalValidationError {
    NotGitRepository,
    Io {
        context: &'static str,
        source: io::Error,
    },
}

pub trait RepoIdentityResolver: Send + Sync + 'static {
    fn resolve_git_dir<'a>(
        &'a self,
        repo_root: &'a Path,
    ) -> RepoIdentityFuture<'a, Result<PathBuf, RepoLocalValidationError>>;

    fn compute_repo_identity<'a>(
        &'a self,
        repo_root: &'a Path,
    ) -> RepoIdentityFuture<'a, Result<String, RepoLocalValidationError>>;
}

#[derive(Clone)]
pub struct GitRepoIdentityResolver {
    git_backend: Arc<dyn GitBackend>,
}

impl GitRepoIdentityResolver {
    #[must_use]
    pub fn new(git_backend: Arc<dyn GitBackend>) -> Self {
        Self { git_backend }
    }

    async fn first_remote_url(
        &self,
        repo_root: &Path,
    ) -> Result<Option<String>, RepoLocalValidationError> {
        for key in ["remote.origin.url", "remote.upstream.url"] {
            let value = self
                .git_backend
                .config_get(repo_root, key, GitRunOptions::default())
                .await
                .map_err(|source| map_git_error("read git config", repo_root, source))?;
            if value.is_some() {
                return Ok(value);
            }
        }

        let remotes = self
            .git_backend
            .remote_urls(repo_root, GitRunOptions::default())
            .await
            .map_err(|source| map_git_error("list remotes", repo_root, source))?;
        Ok(remotes.into_iter().next().map(|entry| entry.url))
    }
}

impl RepoIdentityResolver for GitRepoIdentityResolver {
    fn resolve_git_dir<'a>(
        &'a self,
        repo_root: &'a Path,
    ) -> RepoIdentityFuture<'a, Result<PathBuf, RepoLocalValidationError>> {
        Box::pin(async move {
            let git_dir = self
                .git_backend
                .absolute_git_dir(repo_root, GitRunOptions::default())
                .await
                .map_err(|source| map_git_error("resolve git dir", repo_root, source))?;

            std::fs::canonicalize(&git_dir).map_err(|source| RepoLocalValidationError::Io {
                context: "canonicalize git dir",
                source,
            })
        })
    }

    fn compute_repo_identity<'a>(
        &'a self,
        repo_root: &'a Path,
    ) -> RepoIdentityFuture<'a, Result<String, RepoLocalValidationError>> {
        Box::pin(async move {
            let _git_dir = self.resolve_git_dir(repo_root).await?;

            let remote = self.first_remote_url(repo_root).await?;
            let source = remote
                .as_deref()
                .map(normalize_remote_url)
                .unwrap_or_else(|| repo_root.display().to_string());

            Ok(stable_fnv1a64_hex(source.as_bytes()))
        })
    }
}

fn map_git_error(
    context: &'static str,
    repo_root: &Path,
    source: GitError,
) -> RepoLocalValidationError {
    match source {
        GitError::NotAGitRepository { .. } => RepoLocalValidationError::NotGitRepository,
        other => RepoLocalValidationError::Io {
            context,
            source: io::Error::other(format!("repo_root={} error={other}", repo_root.display())),
        },
    }
}

pub(crate) fn normalize_remote_url(remote: &str) -> String {
    let value = remote.trim();

    if let Some((user_host, path)) = value.split_once(':')
        && let Some((_user, host)) = user_host.split_once('@')
        && !path.starts_with('/')
    {
        let normalized_path = trim_git_suffix(path.trim().trim_start_matches('/'));
        return format!("{}/{}", host.to_ascii_lowercase(), normalized_path);
    }

    for prefix in ["https://", "http://", "ssh://", "git://"] {
        if let Some(remainder) = value.strip_prefix(prefix) {
            let cleaned = remainder
                .split('#')
                .next()
                .unwrap_or(remainder)
                .split('?')
                .next()
                .unwrap_or(remainder)
                .trim_end_matches('/');

            if let Some((host, rest)) = cleaned.split_once('/') {
                let normalized = trim_git_suffix(rest.trim().trim_start_matches('/'));
                return format!("{}/{}", host.trim().to_ascii_lowercase(), normalized);
            }

            return cleaned.to_ascii_lowercase();
        }
    }

    trim_git_suffix(value.trim_end_matches('/')).to_string()
}

fn trim_git_suffix(value: &str) -> &str {
    value.strip_suffix(".git").unwrap_or(value)
}

pub(crate) fn stable_fnv1a64_hex(bytes: &[u8]) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0001_0000_01b3;

    let hash = bytes.iter().fold(OFFSET, |acc, byte| {
        (acc ^ u64::from(*byte)).wrapping_mul(PRIME)
    });

    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::{normalize_remote_url, stable_fnv1a64_hex};

    #[test]
    fn normalizes_remote_variants_to_same_identity_source() {
        let a = normalize_remote_url("https://github.com/example/repo.git");
        let b = normalize_remote_url("git@github.com:example/repo.git");
        let c = normalize_remote_url("ssh://github.com/example/repo.git");

        assert_eq!(a, "github.com/example/repo");
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn stable_hash_is_deterministic() {
        assert_eq!(
            stable_fnv1a64_hex(b"github.com/example/repo"),
            stable_fnv1a64_hex(b"github.com/example/repo")
        );
        assert_ne!(
            stable_fnv1a64_hex(b"github.com/example/repo-a"),
            stable_fnv1a64_hex(b"github.com/example/repo-b")
        );
    }
}
