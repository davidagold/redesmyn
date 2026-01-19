use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_STATE_DIR_NAME: &str = ".redesmyn";

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// `$XDG_CONFIG_HOME/redesmyn/config.toml` or `~/.config/redesmyn/config.toml`.
pub fn global_config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| home_dir().map(|home| home.join(".config")))?;
    Some(base.join("redesmyn").join("config.toml"))
}

/// `<repo>/.redesmyn/config.toml`.
pub fn repo_config_path(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("config.toml")
}

pub(crate) fn state_dir(base_dir: &Path) -> PathBuf {
    base_dir.join(DEFAULT_STATE_DIR_NAME)
}

pub(crate) fn default_db_path(base_dir: &Path) -> PathBuf {
    state_dir(base_dir).join("redesmyn.sqlite3")
}

pub(crate) fn default_repo_registry_dir(base_dir: &Path) -> PathBuf {
    state_dir(base_dir).join("repos")
}

pub(crate) fn default_control_plane_client_socket_path(base_dir: &Path) -> PathBuf {
    state_dir(base_dir).join("control_plane.sock")
}

/// Best-effort repo root discovery by looking for `.git` in `start` or its ancestors.
pub fn discover_repo_root_from(start: &Path) -> Option<PathBuf> {
    let start_dir = if start.is_dir() {
        start
    } else {
        start.parent()?
    };
    for dir in start_dir.ancestors() {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
    }
    None
}

pub(crate) fn discover_repo_root_from_cwd() -> Option<PathBuf> {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| discover_repo_root_from(&cwd))
}
