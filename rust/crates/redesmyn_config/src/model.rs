use std::ffi::OsStr;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::ValidationError;
use crate::secret::SecretString;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigProfile {
    Dev,
    Release,
}

impl Default for ConfigProfile {
    fn default() -> Self {
        if cfg!(debug_assertions) {
            Self::Dev
        } else {
            Self::Release
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RustConfig {
    pub profile: ConfigProfile,
    pub control_plane: ControlPlaneConfig,
    pub daemon: DaemonConfig,
    pub desktop: DesktopConfig,
}

impl RustConfig {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.daemon.executor.max_concurrency == 0 {
            return Err(ValidationError::InvalidMaxConcurrency {
                value: self.daemon.executor.max_concurrency,
            });
        }
        if self.desktop.window.width == 0 || self.desktop.window.height == 0 {
            return Err(ValidationError::InvalidWindowSize {
                width: self.desktop.window.width,
                height: self.desktop.window.height,
            });
        }
        if self.control_plane.db.path.as_os_str().is_empty() {
            return Err(ValidationError::EmptyDbPath);
        }
        if self.control_plane.db.path.file_name() == Some(OsStr::new("redesmyn.sqlite3"))
            && self
                .control_plane
                .db
                .path
                .parent()
                .and_then(|p| p.file_name())
                == Some(OsStr::new(crate::paths::DEFAULT_STATE_DIR_NAME))
        {
            return Err(ValidationError::LegacyDbPathNotAllowed {
                path: self.control_plane.db.path.clone(),
            });
        }
        if self
            .control_plane
            .api
            .client_socket_path
            .as_os_str()
            .is_empty()
        {
            return Err(ValidationError::EmptyClientSocketPath);
        }
        if self.daemon.repo_registry_dir.as_os_str().is_empty() {
            return Err(ValidationError::EmptyRepoRegistryDir);
        }
        if self.daemon.worktree_root.as_os_str().is_empty() {
            return Err(ValidationError::EmptyWorktreeRoot);
        }

        if self.profile == ConfigProfile::Release {
            let token = self.control_plane.auth.daemon_token.expose();
            if token.is_empty() || token == "dev" {
                return Err(ValidationError::ReleaseDaemonTokenIsDev);
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlPlaneConfig {
    pub db: ControlPlaneDbConfig,
    pub api: ControlPlaneApiConfig,
    pub auth: ControlPlaneAuthConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlPlaneDbConfig {
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlPlaneApiConfig {
    pub bind: SocketAddr,
    pub client_socket_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlPlaneAuthConfig {
    pub daemon_token: SecretString,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonConfig {
    pub repo_registry_dir: PathBuf,
    pub worktree_root: PathBuf,
    pub executor: ExecutorConfig,
    pub sandbox: SandboxConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorConfig {
    pub max_concurrency: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxType {
    None,
    Worktree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxNetworkMode {
    Allow,
    Deny,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxConfig {
    pub kind: SandboxType,
    pub network: SandboxNetworkMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesktopConfig {
    pub embed_control_plane: bool,
    pub embed_daemon: bool,
    pub window: WindowConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
}

pub(crate) fn default_api_bind() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9234)
}

pub(crate) fn default_executor_max_concurrency() -> usize {
    4
}

pub(crate) fn default_window() -> WindowConfig {
    WindowConfig {
        width: 1280,
        height: 800,
    }
}
