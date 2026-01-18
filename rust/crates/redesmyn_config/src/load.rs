use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use config::{Environment, File, FileFormat};
use serde::Deserialize;

use crate::error::{LoadConfigError, ValidationError};
use crate::model::{
    ConfigProfile, ControlPlaneApiConfig, ControlPlaneAuthConfig, ControlPlaneConfig,
    ControlPlaneDbConfig, DaemonConfig, DesktopConfig, ExecutorConfig, RustConfig, SandboxConfig,
    SandboxNetworkMode, SandboxType, WindowConfig, default_api_bind,
    default_executor_max_concurrency, default_window,
};
use crate::paths::{default_db_path, default_repo_registry_dir, discover_repo_root_from_cwd};
use crate::secret::SecretString;

#[derive(Debug, Clone)]
pub enum ConfigFiles {
    /// Load from:
    /// - `$XDG_CONFIG_HOME/redesmyn/config.toml` (or `~/.config/redesmyn/config.toml`)
    /// - `<repo>/.redesmyn/config.toml` (if a repo root can be discovered or provided)
    Default,
    /// Load from exactly these file paths, in this order.
    Explicit(Vec<PathBuf>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotenvMode {
    /// Load `.env` in debug builds, ignore in release builds.
    Auto,
    /// Always load `.env` (if present).
    Always,
    /// Never load `.env`.
    Never,
}

#[derive(Debug, Clone)]
pub struct LoadConfigOptions {
    pub repo_root: Option<PathBuf>,
    pub config_files: ConfigFiles,
    pub dotenv: DotenvMode,
    pub include_env: bool,
    pub default_profile: ConfigProfile,
}

impl Default for LoadConfigOptions {
    fn default() -> Self {
        Self {
            repo_root: None,
            config_files: ConfigFiles::Default,
            dotenv: DotenvMode::Auto,
            include_env: true,
            default_profile: ConfigProfile::default(),
        }
    }
}

pub fn load_rust_config(options: LoadConfigOptions) -> Result<RustConfig, LoadConfigError> {
    let repo_root = options.repo_root.or_else(discover_repo_root_from_cwd);
    let base_dir = match repo_root.as_ref() {
        Some(repo_root) => repo_root.clone(),
        None => std::env::current_dir().map_err(|source| LoadConfigError::NoBaseDir { source })?,
    };

    load_dotenv_if_configured(options.dotenv, repo_root.as_deref(), &base_dir)?;

    let mut builder = config::Config::builder();
    let (config_files, required) =
        config_files_to_load(&options.config_files, repo_root.as_deref());
    for path in config_files {
        let mut source = File::from(path).format(FileFormat::Toml);
        if !required {
            source = source.required(false);
        }
        builder = builder.add_source(source);
    }

    if options.include_env {
        builder = builder.add_source(
            Environment::with_prefix("REDESMYN")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true)
                .ignore_empty(true),
        );
    }

    let merged = builder.build()?;
    let input: RootConfigInput = merged.try_deserialize()?;

    let ctx = LoadContext {
        base_dir,
        default_profile: options.default_profile,
    };

    let config = build_rust_config(input.rust, &ctx)?;
    Ok(config)
}

struct LoadContext {
    base_dir: PathBuf,
    default_profile: ConfigProfile,
}

fn config_files_to_load(
    config_files: &ConfigFiles,
    repo_root: Option<&Path>,
) -> (Vec<PathBuf>, bool) {
    match config_files {
        ConfigFiles::Default => {
            let mut paths = Vec::new();
            if let Some(path) = crate::paths::global_config_path() {
                paths.push(path);
            }
            if let Some(repo_root) = repo_root {
                paths.push(crate::paths::repo_config_path(repo_root));
            }
            (paths, false)
        }
        ConfigFiles::Explicit(paths) => (paths.clone(), true),
    }
}

fn load_dotenv_if_configured(
    mode: DotenvMode,
    repo_root: Option<&Path>,
    base_dir: &Path,
) -> Result<(), LoadConfigError> {
    let should_load = match mode {
        DotenvMode::Auto => cfg!(debug_assertions),
        DotenvMode::Always => true,
        DotenvMode::Never => false,
    };
    if !should_load {
        return Ok(());
    }

    let dotenv_path = repo_root
        .map(|root| root.join(".env"))
        .unwrap_or_else(|| base_dir.join(".env"));

    if !dotenv_path.exists() {
        return Ok(());
    }

    dotenvy::from_path(&dotenv_path).map_err(|source| LoadConfigError::Dotenv {
        path: dotenv_path,
        source,
    })?;
    Ok(())
}

fn resolve_path(base_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn build_rust_config(
    input: RustConfigInput,
    ctx: &LoadContext,
) -> Result<RustConfig, ValidationError> {
    let profile = input.profile.unwrap_or(ctx.default_profile);

    let db_path = input
        .control_plane
        .db
        .path
        .map(|p| resolve_path(&ctx.base_dir, p))
        .unwrap_or_else(|| default_db_path(&ctx.base_dir));

    let repo_registry_dir = input
        .daemon
        .repo_registry_dir
        .map(|p| resolve_path(&ctx.base_dir, p))
        .unwrap_or_else(|| default_repo_registry_dir(&ctx.base_dir));

    let worktree_root = input
        .daemon
        .worktree_root
        .map(|p| resolve_path(&ctx.base_dir, p))
        .unwrap_or_else(|| ctx.base_dir.clone());

    let daemon_token = input
        .control_plane
        .auth
        .daemon_token
        .unwrap_or_else(|| SecretString::new("dev"));

    let api_bind = input
        .control_plane
        .api
        .bind
        .unwrap_or_else(default_api_bind);

    let max_concurrency = input
        .daemon
        .executor
        .max_concurrency
        .unwrap_or_else(default_executor_max_concurrency);

    let sandbox_kind = input.daemon.sandbox.kind.unwrap_or(SandboxType::None);
    let sandbox_network = input
        .daemon
        .sandbox
        .network
        .unwrap_or(SandboxNetworkMode::Allow);

    let embed_control_plane = input.desktop.embed_control_plane.unwrap_or(true);
    let embed_daemon = input.desktop.embed_daemon.unwrap_or(true);
    let default_window = default_window();
    let window = WindowConfig {
        width: input.desktop.window.width.unwrap_or(default_window.width),
        height: input.desktop.window.height.unwrap_or(default_window.height),
    };

    let config = RustConfig {
        profile,
        control_plane: ControlPlaneConfig {
            db: ControlPlaneDbConfig { path: db_path },
            api: ControlPlaneApiConfig { bind: api_bind },
            auth: ControlPlaneAuthConfig {
                daemon_token: daemon_token.into(),
            },
        },
        daemon: DaemonConfig {
            repo_registry_dir,
            worktree_root,
            executor: ExecutorConfig { max_concurrency },
            sandbox: SandboxConfig {
                kind: sandbox_kind,
                network: sandbox_network,
            },
        },
        desktop: DesktopConfig {
            embed_control_plane,
            embed_daemon,
            window,
        },
    };

    config.validate()?;
    Ok(config)
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct RootConfigInput {
    rust: RustConfigInput,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct RustConfigInput {
    profile: Option<ConfigProfile>,
    control_plane: ControlPlaneConfigInput,
    daemon: DaemonConfigInput,
    desktop: DesktopConfigInput,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct ControlPlaneConfigInput {
    db: ControlPlaneDbConfigInput,
    api: ControlPlaneApiConfigInput,
    auth: ControlPlaneAuthConfigInput,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct ControlPlaneDbConfigInput {
    path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct ControlPlaneApiConfigInput {
    bind: Option<SocketAddr>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct ControlPlaneAuthConfigInput {
    daemon_token: Option<SecretString>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct DaemonConfigInput {
    repo_registry_dir: Option<PathBuf>,
    worktree_root: Option<PathBuf>,
    executor: ExecutorConfigInput,
    sandbox: SandboxConfigInput,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct ExecutorConfigInput {
    max_concurrency: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct SandboxConfigInput {
    #[serde(rename = "type")]
    kind: Option<SandboxType>,
    network: Option<SandboxNetworkMode>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct DesktopConfigInput {
    embed_control_plane: Option<bool>,
    embed_daemon: Option<bool>,
    window: WindowConfigInput,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct WindowConfigInput {
    width: Option<u32>,
    height: Option<u32>,
}
