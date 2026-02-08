use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use redesmyn_git::GitBackend;
use redesmyn_logging::tracing;
use redesmyn_protocol::RepoScope;
use tokio::sync::watch;
use tokio::task::JoinHandle;

#[derive(Debug, thiserror::Error)]
pub enum RepoRegistryError {
    #[error("repo registry is not configured (T-24)")]
    Unconfigured,
}

pub trait RepoRegistry: Send + Sync + 'static {
    fn resolve_repo_root(&self, scope: RepoScope) -> Result<PathBuf, RepoRegistryError>;
}

#[derive(Debug)]
pub struct UnconfiguredRepoRegistry;

impl RepoRegistry for UnconfiguredRepoRegistry {
    fn resolve_repo_root(&self, _scope: RepoScope) -> Result<PathBuf, RepoRegistryError> {
        Err(RepoRegistryError::Unconfigured)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RepoAttachError {
    #[error("daemon runtime is unavailable")]
    DaemonUnavailable,
    #[error(transparent)]
    Registry(#[from] RepoRegistryError),
}

#[derive(Debug, thiserror::Error)]
pub enum RepoDetachError {
    #[error("daemon runtime is unavailable")]
    DaemonUnavailable,
    #[error("repo not attached: {scope:?}")]
    NotAttached { scope: RepoScope },
}

#[derive(Debug)]
struct AttachedRepo {
    #[allow(dead_code)]
    repo_root: PathBuf,
    tasks: Vec<JoinHandle<()>>,
}

pub struct RepoAttachmentManager {
    registry: Arc<dyn RepoRegistry>,
    #[allow(dead_code)]
    git_backend: Arc<dyn GitBackend>,
    attached: HashMap<RepoScope, AttachedRepo>,
}

impl RepoAttachmentManager {
    #[must_use]
    pub fn new(registry: Arc<dyn RepoRegistry>, git_backend: Arc<dyn GitBackend>) -> Self {
        Self {
            registry,
            git_backend,
            attached: HashMap::new(),
        }
    }

    pub fn attach(
        &mut self,
        scope: RepoScope,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<(), RepoAttachError> {
        if self.attached.contains_key(&scope) {
            return Ok(());
        }

        let repo_root = self.registry.resolve_repo_root(scope)?;
        let tasks = spawn_repo_placeholder_tasks(scope, repo_root.clone(), shutdown_rx);
        self.attached
            .insert(scope, AttachedRepo { repo_root, tasks });
        Ok(())
    }

    pub fn detach(&mut self, scope: RepoScope) -> Result<(), RepoDetachError> {
        let Some(attached) = self.attached.remove(&scope) else {
            return Err(RepoDetachError::NotAttached { scope });
        };

        for task in attached.tasks {
            task.abort();
        }
        Ok(())
    }
}

fn spawn_repo_placeholder_tasks(
    scope: RepoScope,
    repo_root: PathBuf,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Vec<JoinHandle<()>> {
    let handle = tokio::spawn(async move {
        let span = redesmyn_logging::redesmyn_info_span!("daemon.repo.worker");
        redesmyn_logging::span::record_workspace_id(&span, scope.workspace_id);
        redesmyn_logging::span::record_repo_id(&span, scope.repo_id);

        let _enter = span.enter();
        tracing::info!(repo_root = %repo_root.display(), "repo placeholder task started");

        loop {
            if *shutdown_rx.borrow() {
                break;
            }
            if shutdown_rx.changed().await.is_err() {
                break;
            }
        }

        tracing::info!("repo placeholder task exiting");
    });

    vec![handle]
}
