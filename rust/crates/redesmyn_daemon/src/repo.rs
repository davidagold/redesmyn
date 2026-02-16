use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, RwLock};

use fs4::fs_std::FileExt;
use redesmyn_ids::{HostId, HostInstanceId, RepoId, WorkspaceId};
use redesmyn_logging::tracing;
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope, RepoScope, Timestamp};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::host_identity::HostIdentity;

const REPO_REGISTRY_FILE_NAME: &str = "registry.v1.json";
const REPO_REGISTRY_SCHEMA_VERSION: u32 = 1;
const LOCK_FILE_NAME: &str = "redesmyn.repo_instance.lock";
const LOCK_OWNER_FILE_NAME: &str = "redesmyn.repo_instance.owner.json";

pub const CONFLICT_CODE_KEY: &str = "conflict_code";
pub const CONFLICT_CODE_REPO_INSTANCE_BUSY: &str = "repo_instance_busy";
pub const CONFLICT_CODE_REPO_IDENTITY_MISMATCH: &str = "repo_identity_mismatch";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RepoRegistration {
    pub scope: RepoScope,
    pub repo_root: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trusted: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen_at: Option<Timestamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo_identity: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RepoRegistrationRequest {
    pub scope: RepoScope,
    pub repo_root: PathBuf,
    pub display_name: Option<String>,
    pub trusted: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum RepoRegistryError {
    #[error("repo registry is not configured")]
    Unconfigured,
    #[error("repo scope is not registered: workspace_id={workspace_id} repo_id={repo_id}")]
    ScopeNotRegistered {
        workspace_id: WorkspaceId,
        repo_id: RepoId,
    },
    #[error("repo registry operation is unsupported: {operation}")]
    UnsupportedOperation { operation: &'static str },
    #[error("repo registry I/O failed while {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("repo registry is invalid at {path}: {message}")]
    Corrupt { path: PathBuf, message: String },
}

pub trait RepoRegistry: Send + Sync + 'static {
    fn resolve_repo_root(&self, scope: RepoScope) -> Result<PathBuf, RepoRegistryError>;

    fn resolve_repo_registration(
        &self,
        scope: RepoScope,
    ) -> Result<RepoRegistration, RepoRegistryError> {
        let repo_root = self.resolve_repo_root(scope)?;
        Ok(RepoRegistration {
            scope,
            repo_root,
            display_name: None,
            trusted: None,
            last_seen_at: None,
            repo_identity: None,
        })
    }

    fn register_repo(&self, _registration: RepoRegistration) -> Result<(), RepoRegistryError> {
        Err(RepoRegistryError::UnsupportedOperation {
            operation: "register_repo",
        })
    }

    fn mark_repo_seen(&self, _scope: RepoScope) -> Result<(), RepoRegistryError> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct UnconfiguredRepoRegistry;

impl RepoRegistry for UnconfiguredRepoRegistry {
    fn resolve_repo_root(&self, _scope: RepoScope) -> Result<PathBuf, RepoRegistryError> {
        Err(RepoRegistryError::Unconfigured)
    }
}

#[derive(Debug)]
pub struct FileRepoRegistry {
    registry_dir: PathBuf,
    registry_path: PathBuf,
    io_lock: Mutex<()>,
}

impl FileRepoRegistry {
    #[must_use]
    pub fn new(registry_dir: PathBuf) -> Self {
        let registry_path = registry_dir.join(REPO_REGISTRY_FILE_NAME);
        Self {
            registry_dir,
            registry_path,
            io_lock: Mutex::new(()),
        }
    }

    fn with_locked_document<T>(
        &self,
        operation: &'static str,
        f: impl FnOnce(&mut RepoRegistryDocument) -> Result<T, RepoRegistryError>,
    ) -> Result<T, RepoRegistryError> {
        let _guard = self.io_lock.lock().expect("repo registry lock poisoned");

        let mut doc = self.load_document()?;
        let result = f(&mut doc)?;
        self.save_document(operation, &doc)?;
        Ok(result)
    }

    fn load_document(&self) -> Result<RepoRegistryDocument, RepoRegistryError> {
        if !self.registry_path.exists() {
            return Ok(RepoRegistryDocument::default());
        }

        let data = fs::read(&self.registry_path).map_err(|source| RepoRegistryError::Io {
            operation: "read registry",
            path: self.registry_path.clone(),
            source,
        })?;

        serde_json::from_slice::<RepoRegistryDocument>(&data).map_err(|err| {
            RepoRegistryError::Corrupt {
                path: self.registry_path.clone(),
                message: err.to_string(),
            }
        })
    }

    fn save_document(
        &self,
        operation: &'static str,
        doc: &RepoRegistryDocument,
    ) -> Result<(), RepoRegistryError> {
        fs::create_dir_all(&self.registry_dir).map_err(|source| RepoRegistryError::Io {
            operation: "create registry directory",
            path: self.registry_dir.clone(),
            source,
        })?;

        let mut sorted = doc.clone();
        sorted.registrations.sort_by_key(|registration| {
            (
                registration.scope.workspace_id.to_string(),
                registration.scope.repo_id.to_string(),
            )
        });

        let encoded =
            serde_json::to_vec_pretty(&sorted).map_err(|err| RepoRegistryError::Corrupt {
                path: self.registry_path.clone(),
                message: format!("failed to encode registry JSON: {err}"),
            })?;

        let temp_path = self.registry_path.with_extension("json.tmp");
        fs::write(&temp_path, encoded).map_err(|source| RepoRegistryError::Io {
            operation: "write registry temp file",
            path: temp_path.clone(),
            source,
        })?;

        fs::rename(&temp_path, &self.registry_path).map_err(|source| RepoRegistryError::Io {
            operation,
            path: self.registry_path.clone(),
            source,
        })
    }

    fn lookup<'a>(
        registrations: &'a [RepoRegistration],
        scope: RepoScope,
    ) -> Option<&'a RepoRegistration> {
        registrations
            .iter()
            .find(|registration| registration.scope == scope)
    }

    fn lookup_mut(
        registrations: &mut [RepoRegistration],
        scope: RepoScope,
    ) -> Option<&mut RepoRegistration> {
        registrations
            .iter_mut()
            .find(|registration| registration.scope == scope)
    }

    fn scope_not_registered(scope: RepoScope) -> RepoRegistryError {
        RepoRegistryError::ScopeNotRegistered {
            workspace_id: scope.workspace_id,
            repo_id: scope.repo_id,
        }
    }
}

impl RepoRegistry for FileRepoRegistry {
    fn resolve_repo_root(&self, scope: RepoScope) -> Result<PathBuf, RepoRegistryError> {
        let _guard = self.io_lock.lock().expect("repo registry lock poisoned");
        let doc = self.load_document()?;
        let Some(registration) = Self::lookup(&doc.registrations, scope) else {
            return Err(Self::scope_not_registered(scope));
        };
        Ok(registration.repo_root.clone())
    }

    fn resolve_repo_registration(
        &self,
        scope: RepoScope,
    ) -> Result<RepoRegistration, RepoRegistryError> {
        let _guard = self.io_lock.lock().expect("repo registry lock poisoned");
        let doc = self.load_document()?;
        Self::lookup(&doc.registrations, scope)
            .cloned()
            .ok_or_else(|| Self::scope_not_registered(scope))
    }

    fn register_repo(&self, registration: RepoRegistration) -> Result<(), RepoRegistryError> {
        self.with_locked_document("persist registry", |doc| {
            if doc.schema_version != REPO_REGISTRY_SCHEMA_VERSION {
                return Err(RepoRegistryError::Corrupt {
                    path: self.registry_path.clone(),
                    message: format!(
                        "unsupported schema_version={} (expected {})",
                        doc.schema_version, REPO_REGISTRY_SCHEMA_VERSION
                    ),
                });
            }

            if let Some(existing) = Self::lookup_mut(&mut doc.registrations, registration.scope) {
                *existing = registration;
            } else {
                doc.registrations.push(registration);
            }

            Ok(())
        })
    }

    fn mark_repo_seen(&self, scope: RepoScope) -> Result<(), RepoRegistryError> {
        self.with_locked_document("persist registry", |doc| {
            let Some(existing) = Self::lookup_mut(&mut doc.registrations, scope) else {
                return Err(Self::scope_not_registered(scope));
            };
            existing.last_seen_at = Some(Timestamp::now_utc());
            Ok(())
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RepoRegistryDocument {
    schema_version: u32,
    #[serde(default)]
    registrations: Vec<RepoRegistration>,
}

impl Default for RepoRegistryDocument {
    fn default() -> Self {
        Self {
            schema_version: REPO_REGISTRY_SCHEMA_VERSION,
            registrations: Vec::new(),
        }
    }
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
struct RepoInstanceLock {
    #[allow(dead_code)]
    file: std::fs::File,
    lock_path: PathBuf,
}

impl RepoInstanceLock {
    fn acquire(git_dir: &Path, owner: RepoInstanceOwner) -> Result<Self, RepoAttachError> {
        let lock_path = git_dir.join(LOCK_FILE_NAME);
        let owner_path = git_dir.join(LOCK_OWNER_FILE_NAME);
        let lock_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|source| RepoAttachError::Io {
                context: "open repo instance lock",
                source,
            })?;

        match lock_file.try_lock_exclusive() {
            Ok(()) => {
                write_owner_metadata(&owner_path, &owner)?;
                Ok(Self {
                    file: lock_file,
                    lock_path,
                })
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                let owner = read_owner_metadata(&owner_path).ok();
                Err(RepoAttachError::RepoInstanceBusy { lock_path, owner })
            }
            Err(source) => Err(RepoAttachError::Io {
                context: "acquire repo instance lock",
                source,
            }),
        }
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
    attached_roots: AttachedRepoRoots,
    attached: HashMap<RepoScope, AttachedRepo>,
}

impl RepoAttachmentManager {
    #[must_use]
    pub fn new(
        identity: HostIdentity,
        registry: Arc<dyn RepoRegistry>,
        attached_roots: AttachedRepoRoots,
    ) -> Self {
        Self {
            identity,
            registry,
            attached_roots,
            attached: HashMap::new(),
        }
    }

    pub fn register_repo(&self, request: RepoRegistrationRequest) -> Result<(), RepoRegisterError> {
        let repo_root =
            canonicalize_repo_root(&request.repo_root).map_err(|source| RepoRegisterError::Io {
                context: "canonicalize repo path",
                source,
            })?;

        if !repo_root.is_dir() {
            return Err(RepoRegisterError::InvalidRepoPath { repo_root });
        }

        ensure_git_repository(&repo_root).map_err(|err| match err {
            RepoLocalValidationError::NotGitRepository => RepoRegisterError::NotGitRepository {
                repo_root: repo_root.clone(),
            },
            RepoLocalValidationError::Io { context, source } => {
                RepoRegisterError::Io { context, source }
            }
        })?;

        let repo_identity = compute_repo_identity(&repo_root).map_err(|err| match err {
            RepoLocalValidationError::NotGitRepository => RepoRegisterError::NotGitRepository {
                repo_root: repo_root.clone(),
            },
            RepoLocalValidationError::Io { context, source } => {
                RepoRegisterError::Io { context, source }
            }
        })?;

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

    pub fn attach(
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

        ensure_git_repository(&repo_root).map_err(|err| match err {
            RepoLocalValidationError::NotGitRepository => RepoAttachError::NotGitRepository {
                repo_root: repo_root.clone(),
            },
            RepoLocalValidationError::Io { context, source } => {
                RepoAttachError::Io { context, source }
            }
        })?;

        let actual_identity = compute_repo_identity(&repo_root).map_err(|err| match err {
            RepoLocalValidationError::NotGitRepository => RepoAttachError::NotGitRepository {
                repo_root: repo_root.clone(),
            },
            RepoLocalValidationError::Io { context, source } => {
                RepoAttachError::Io { context, source }
            }
        })?;

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

        let git_dir = resolve_git_dir(&repo_root).map_err(|err| match err {
            RepoLocalValidationError::NotGitRepository => RepoAttachError::NotGitRepository {
                repo_root: repo_root.clone(),
            },
            RepoLocalValidationError::Io { context, source } => {
                RepoAttachError::Io { context, source }
            }
        })?;

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

    pub fn detach(&mut self, scope: RepoScope) -> Result<(), RepoDetachError> {
        if let Some(attached) = self.attached.remove(&scope) {
            for task in attached.tasks {
                task.abort();
            }
        }

        self.attached_roots.remove(scope);
        Ok(())
    }
}

#[derive(Debug)]
enum RepoLocalValidationError {
    NotGitRepository,
    Io {
        context: &'static str,
        source: io::Error,
    },
}

fn canonicalize_repo_root(repo_root: &Path) -> Result<PathBuf, io::Error> {
    fs::canonicalize(repo_root)
}

fn ensure_git_repository(repo_root: &Path) -> Result<(), RepoLocalValidationError> {
    resolve_git_dir(repo_root).map(|_| ())
}

fn resolve_git_dir(repo_root: &Path) -> Result<PathBuf, RepoLocalValidationError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("rev-parse")
        .arg("--absolute-git-dir")
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .map_err(|source| RepoLocalValidationError::Io {
            context: "run git rev-parse --absolute-git-dir",
            source,
        })?;

    if !output.status.success() {
        return Err(RepoLocalValidationError::NotGitRepository);
    }

    let raw = String::from_utf8(output.stdout).map_err(|err| RepoLocalValidationError::Io {
        context: "decode git rev-parse output",
        source: io::Error::new(io::ErrorKind::InvalidData, err),
    })?;

    let git_dir = raw.trim();
    if git_dir.is_empty() {
        return Err(RepoLocalValidationError::NotGitRepository);
    }

    fs::canonicalize(git_dir).map_err(|source| RepoLocalValidationError::Io {
        context: "canonicalize git dir",
        source,
    })
}

fn compute_repo_identity(repo_root: &Path) -> Result<String, RepoLocalValidationError> {
    let _git_dir = resolve_git_dir(repo_root)?;

    let remote = first_remote_url(repo_root)?;
    let source = remote
        .as_deref()
        .map(normalize_remote_url)
        .unwrap_or_else(|| repo_root.display().to_string());

    Ok(stable_fnv1a64_hex(source.as_bytes()))
}

fn first_remote_url(repo_root: &Path) -> Result<Option<String>, RepoLocalValidationError> {
    for args in [
        ["config", "--get", "remote.origin.url"].as_slice(),
        ["config", "--get", "remote.upstream.url"].as_slice(),
    ] {
        if let Some(value) = git_capture(repo_root, args)? {
            return Ok(Some(value));
        }
    }

    let remotes = git_capture(repo_root, &["remote", "-v"])?;
    let Some(remotes) = remotes else {
        return Ok(None);
    };

    for line in remotes.lines() {
        let mut parts = line.split_whitespace();
        let _name = parts.next();
        if let Some(url) = parts.next() {
            return Ok(Some(url.to_string()));
        }
    }

    Ok(None)
}

fn git_capture(
    repo_root: &Path,
    args: &[&str],
) -> Result<Option<String>, RepoLocalValidationError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .map_err(|source| RepoLocalValidationError::Io {
            context: "run git command",
            source,
        })?;

    if !output.status.success() {
        return Ok(None);
    }

    let value = String::from_utf8(output.stdout).map_err(|err| RepoLocalValidationError::Io {
        context: "decode git command output",
        source: io::Error::new(io::ErrorKind::InvalidData, err),
    })?;

    let trimmed = value.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}

fn normalize_remote_url(remote: &str) -> String {
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

fn stable_fnv1a64_hex(bytes: &[u8]) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0001_0000_01b3;

    let hash = bytes.iter().fold(OFFSET, |acc, byte| {
        (acc ^ u64::from(*byte)).wrapping_mul(PRIME)
    });

    format!("{hash:016x}")
}

fn write_owner_metadata(
    owner_path: &Path,
    owner: &RepoInstanceOwner,
) -> Result<(), RepoAttachError> {
    let bytes = serde_json::to_vec_pretty(owner).map_err(|err| RepoAttachError::Io {
        context: "encode repo instance owner metadata",
        source: io::Error::new(io::ErrorKind::InvalidData, err),
    })?;

    fs::write(owner_path, bytes).map_err(|source| RepoAttachError::Io {
        context: "write repo instance owner metadata",
        source,
    })
}

fn read_owner_metadata(owner_path: &Path) -> Result<RepoInstanceOwner, io::Error> {
    let bytes = fs::read(owner_path)?;
    serde_json::from_slice::<RepoInstanceOwner>(&bytes)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
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

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;
    use std::sync::Arc;

    use tempfile::TempDir;
    use tokio::sync::watch;

    use super::{
        AttachedRepoRoots, FileRepoRegistry, RepoAttachError, RepoAttachmentManager,
        RepoRegistrationRequest,
    };
    use crate::host_identity::HostIdentity;
    use redesmyn_ids::{HostId, RepoId, WorkspaceId};
    use redesmyn_protocol::RepoScope;

    fn run_git(repo_root: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .output()
            .expect("run git command");

        assert!(
            output.status.success(),
            "git command failed: args={args:?}, stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn init_repo(temp: &TempDir, remote_url: &str) -> std::path::PathBuf {
        let repo_root = temp.path().join("repo");
        std::fs::create_dir_all(&repo_root).expect("create repo root");

        run_git(&repo_root, &["init"]);
        run_git(&repo_root, &["config", "user.email", "test@example.com"]);
        run_git(&repo_root, &["config", "user.name", "Test User"]);
        run_git(&repo_root, &["remote", "add", "origin", remote_url]);

        std::fs::write(repo_root.join("README.md"), "hello\n").expect("write README");
        run_git(&repo_root, &["add", "README.md"]);
        run_git(&repo_root, &["commit", "-m", "initial"]);

        repo_root
    }

    fn test_scope() -> RepoScope {
        RepoScope::new(WorkspaceId::new(), RepoId::new())
    }

    #[tokio::test]
    async fn register_attach_and_detach_are_idempotent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo_root = init_repo(&temp, "https://github.com/example/repo-a.git");
        let expected_repo_root = std::fs::canonicalize(&repo_root).expect("canonicalize repo root");

        let scope = test_scope();
        let registry = Arc::new(FileRepoRegistry::new(temp.path().join("registry")));
        let attached_roots = AttachedRepoRoots::default();
        let identity = HostIdentity::new(HostId::new());
        let mut manager = RepoAttachmentManager::new(identity, registry, attached_roots.clone());

        manager
            .register_repo(RepoRegistrationRequest {
                scope,
                repo_root: repo_root.clone(),
                display_name: Some("repo-a".to_string()),
                trusted: Some(true),
            })
            .expect("register repo");

        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        manager
            .attach(scope, shutdown_rx.clone())
            .expect("first attach");
        manager
            .attach(scope, shutdown_rx)
            .expect("second attach is idempotent");

        assert_eq!(attached_roots.resolve(scope), Some(expected_repo_root));

        manager.detach(scope).expect("first detach");
        manager.detach(scope).expect("second detach is idempotent");

        assert!(attached_roots.resolve(scope).is_none());
    }

    #[tokio::test]
    async fn attach_detects_identity_mismatch_after_registration() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo_root = init_repo(&temp, "https://github.com/example/repo-a.git");

        let scope = test_scope();
        let registry = Arc::new(FileRepoRegistry::new(temp.path().join("registry")));
        let attached_roots = AttachedRepoRoots::default();

        let manager_identity = HostIdentity::new(HostId::new());
        let mut manager = RepoAttachmentManager::new(manager_identity, registry, attached_roots);

        manager
            .register_repo(RepoRegistrationRequest {
                scope,
                repo_root: repo_root.clone(),
                display_name: None,
                trusted: None,
            })
            .expect("register repo");

        run_git(
            &repo_root,
            &[
                "remote",
                "set-url",
                "origin",
                "https://github.com/example/repo-b.git",
            ],
        );

        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let err = manager
            .attach(scope, shutdown_rx)
            .expect_err("attach should fail on identity mismatch");

        match err {
            RepoAttachError::IdentityMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn second_attach_reports_repo_instance_busy_with_owner_metadata() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo_root = init_repo(&temp, "https://github.com/example/repo-a.git");

        let scope = test_scope();
        let registry = Arc::new(FileRepoRegistry::new(temp.path().join("registry")));

        let attached_roots_a = AttachedRepoRoots::default();
        let mut manager_a = RepoAttachmentManager::new(
            HostIdentity::new(HostId::new()),
            registry.clone(),
            attached_roots_a,
        );

        manager_a
            .register_repo(RepoRegistrationRequest {
                scope,
                repo_root,
                display_name: None,
                trusted: None,
            })
            .expect("register repo");

        let (_shutdown_tx_a, shutdown_rx_a) = watch::channel(false);
        manager_a.attach(scope, shutdown_rx_a).expect("attach A");

        let attached_roots_b = AttachedRepoRoots::default();
        let mut manager_b = RepoAttachmentManager::new(
            HostIdentity::new(HostId::new()),
            registry,
            attached_roots_b,
        );

        let (_shutdown_tx_b, shutdown_rx_b) = watch::channel(false);
        let err = manager_b
            .attach(scope, shutdown_rx_b)
            .expect_err("attach B should fail while A holds lock");

        match err {
            RepoAttachError::RepoInstanceBusy { lock_path, owner } => {
                assert!(lock_path.ends_with("redesmyn.repo_instance.lock"));
                assert!(owner.is_some(), "expected owner metadata for busy lock");
                let envelope =
                    RepoAttachError::RepoInstanceBusy { lock_path, owner }.as_error_envelope();
                let detail = envelope.detail.expect("busy detail");
                assert_eq!(
                    detail.get("conflict_code").map(String::as_str),
                    Some("repo_instance_busy")
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
