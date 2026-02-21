use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use redesmyn_logging::tracing;
use redesmyn_protocol::{RepoScope, Timestamp};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::host_identity::HostIdentity;
use crate::repo::identity::RepoLocalValidationError;
use crate::repo::instance_lock::RepoInstanceLock;
use crate::repo::registry::RepoRegistration;
use crate::repo::{
    RepoAttachError, RepoDetachError, RepoIdentityResolver, RepoInstanceOwner, RepoRegisterError,
    RepoRegistrationRequest, RepoRegistry,
};

#[derive(Debug, Clone, Default)]
pub struct AttachedRepoRoots {
    roots: Arc<RwLock<HashMap<RepoScope, PathBuf>>>,
}

impl AttachedRepoRoots {
    pub fn set(&self, scope: RepoScope, repo_root: PathBuf) {
        self.roots
            .write()
            .expect("attached repo roots lock poisoned")
            .insert(scope, repo_root);
    }

    pub fn remove(&self, scope: RepoScope) {
        self.roots
            .write()
            .expect("attached repo roots lock poisoned")
            .remove(&scope);
    }

    pub fn resolve(&self, scope: RepoScope) -> Option<PathBuf> {
        self.roots
            .read()
            .expect("attached repo roots lock poisoned")
            .get(&scope)
            .cloned()
    }

    pub fn scopes(&self) -> Vec<RepoScope> {
        self.roots
            .read()
            .expect("attached repo roots lock poisoned")
            .keys()
            .copied()
            .collect()
    }
}

#[derive(Debug)]
struct AttachedRepo {
    repo_root: PathBuf,
    #[allow(dead_code)]
    instance_lock: RepoInstanceLock,
    tasks: Vec<JoinHandle<()>>,
}

pub struct RepoAttachmentManager {
    identity: HostIdentity,
    registry: Arc<dyn RepoRegistry>,
    identity_resolver: Arc<dyn RepoIdentityResolver>,
    attached_roots: AttachedRepoRoots,
    attached: HashMap<RepoScope, AttachedRepo>,
}

impl RepoAttachmentManager {
    #[must_use]
    pub fn new(
        identity: HostIdentity,
        registry: Arc<dyn RepoRegistry>,
        identity_resolver: Arc<dyn RepoIdentityResolver>,
        attached_roots: AttachedRepoRoots,
    ) -> Self {
        Self {
            identity,
            registry,
            identity_resolver,
            attached_roots,
            attached: HashMap::new(),
        }
    }

    pub async fn register_repo(
        &self,
        request: RepoRegistrationRequest,
    ) -> Result<(), RepoRegisterError> {
        let repo_root =
            canonicalize_repo_root(&request.repo_root).map_err(|source| RepoRegisterError::Io {
                context: "canonicalize repo path",
                source,
            })?;

        if !repo_root.is_dir() {
            return Err(RepoRegisterError::InvalidRepoPath { repo_root });
        }

        let repo_identity = self
            .identity_resolver
            .compute_repo_identity(&repo_root)
            .await
            .map_err(|err| map_validation_to_register_error(err, &repo_root))?;

        // v0 registration flow: a same-host daemon call writes local registry state.
        // Paths never cross the control-plane boundary.
        self.registry.register_repo(RepoRegistration {
            scope: request.scope,
            repo_root,
            display_name: request.display_name,
            trusted: request.trusted,
            last_seen_at: Some(Timestamp::now_utc()),
            repo_identity: Some(repo_identity),
        })?;

        Ok(())
    }

    pub async fn attach(
        &mut self,
        scope: RepoScope,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<(), RepoAttachError> {
        if let Some(attached) = self.attached.get(&scope) {
            self.attached_roots.set(scope, attached.repo_root.clone());
            self.registry.mark_repo_seen(scope)?;
            return Ok(());
        }

        let registration = self.registry.resolve_repo_registration(scope)?;
        let repo_root = canonicalize_repo_root(&registration.repo_root).map_err(|source| {
            RepoAttachError::Io {
                context: "canonicalize registered repo path",
                source,
            }
        })?;

        if !repo_root.is_dir() {
            return Err(RepoAttachError::InvalidRepoPath { repo_root });
        }

        let actual_identity = self
            .identity_resolver
            .compute_repo_identity(&repo_root)
            .await
            .map_err(|err| map_validation_to_attach_error(err, &repo_root))?;

        if let Some(expected_identity) = registration.repo_identity {
            if expected_identity != actual_identity {
                return Err(RepoAttachError::IdentityMismatch {
                    workspace_id: scope.workspace_id,
                    repo_id: scope.repo_id,
                    expected_identity,
                    actual_identity,
                });
            }
        }

        let git_dir = self
            .identity_resolver
            .resolve_git_dir(&repo_root)
            .await
            .map_err(|err| map_validation_to_attach_error(err, &repo_root))?;

        let owner = RepoInstanceOwner {
            host_id: self.identity.host_id,
            host_instance_id: self.identity.host_instance_id,
            pid: std::process::id(),
            acquired_at: Timestamp::now_utc(),
        };

        let instance_lock = RepoInstanceLock::acquire(&git_dir, owner)?;

        tracing::info!(
            repo_root = %repo_root.display(),
            lock_path = %instance_lock.lock_path.display(),
            "repo attached with instance lock",
        );

        let tasks = spawn_repo_placeholder_tasks(scope, repo_root.clone(), shutdown_rx);
        self.attached_roots.set(scope, repo_root.clone());
        self.registry.mark_repo_seen(scope)?;
        self.attached.insert(
            scope,
            AttachedRepo {
                repo_root,
                instance_lock,
                tasks,
            },
        );
        Ok(())
    }

    pub fn detach(&mut self, scope: RepoScope) -> Result<bool, RepoDetachError> {
        let Some(attached) = self.attached.remove(&scope) else {
            self.attached_roots.remove(scope);
            return Ok(false);
        };

        for task in attached.tasks {
            task.abort();
        }

        self.attached_roots.remove(scope);
        Ok(true)
    }
}

fn canonicalize_repo_root(repo_root: &Path) -> Result<PathBuf, std::io::Error> {
    fs::canonicalize(repo_root)
}

fn map_validation_to_register_error(
    err: RepoLocalValidationError,
    repo_root: &Path,
) -> RepoRegisterError {
    match err {
        RepoLocalValidationError::NotGitRepository => RepoRegisterError::NotGitRepository {
            repo_root: repo_root.to_path_buf(),
        },
        RepoLocalValidationError::Io { context, source } => {
            RepoRegisterError::Io { context, source }
        }
    }
}

fn map_validation_to_attach_error(
    err: RepoLocalValidationError,
    repo_root: &Path,
) -> RepoAttachError {
    match err {
        RepoLocalValidationError::NotGitRepository => RepoAttachError::NotGitRepository {
            repo_root: repo_root.to_path_buf(),
        },
        RepoLocalValidationError::Io { context, source } => RepoAttachError::Io { context, source },
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
