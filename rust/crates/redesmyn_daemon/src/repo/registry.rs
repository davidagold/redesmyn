use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Mutex;

use redesmyn_ids::{RepoId, WorkspaceId};
use redesmyn_protocol::{RepoScope, Timestamp};

const REPO_REGISTRY_FILE_NAME: &str = "registry.v1.json";
const REPO_REGISTRY_SCHEMA_VERSION: u32 = 1;

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

    fn lookup(registrations: &[RepoRegistration], scope: RepoScope) -> Option<&RepoRegistration> {
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
