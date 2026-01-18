use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("daemon.executor.max_concurrency must be > 0 (got {value})")]
    InvalidMaxConcurrency { value: usize },

    #[error("desktop.window size must be > 0 (got {width}x{height})")]
    InvalidWindowSize { width: u32, height: u32 },

    #[error(
        "rust.profile=release requires a non-default daemon token; set `REDESMYN_RUST__CONTROL_PLANE__AUTH__DAEMON_TOKEN` or `[rust.control_plane.auth] daemon_token = \"...\"`"
    )]
    ReleaseDaemonTokenIsDev,

    #[error("control_plane.db.path must not be empty")]
    EmptyDbPath,

    #[error("daemon.repo_registry_dir must not be empty")]
    EmptyRepoRegistryDir,

    #[error("daemon.worktree_root must not be empty")]
    EmptyWorktreeRoot,
}

#[derive(Debug, Error)]
pub enum LoadConfigError {
    #[error(transparent)]
    Config(#[from] config::ConfigError),

    #[error("dotenv load failed for {path}")]
    Dotenv {
        path: PathBuf,
        #[source]
        source: dotenvy::Error,
    },

    #[error("failed to determine base directory for relative paths")]
    NoBaseDir {
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Validation(#[from] ValidationError),
}
