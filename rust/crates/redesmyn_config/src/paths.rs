use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_STATE_DIR_NAME: &str = ".redesmyn";
const CONTROL_PLANE_SOCKET_FILE_NAME: &str = "control_plane.sock";

#[cfg(unix)]
const UNIX_SOCKET_PATH_MAX_BYTES: usize = 103;
#[cfg(unix)]
const FALLBACK_SOCKET_FILE_PREFIX: &str = "cp-";

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

/// `<repo>/.redesmyn/`.
///
/// This is the repo-scoped state directory shared by the legacy Python app and
/// the Rust port during the split-codebase period.
pub fn repo_state_dir(repo_root: &Path) -> PathBuf {
    state_dir(repo_root)
}

/// Legacy Python DB (Alembic): `<repo>/.redesmyn/redesmyn.sqlite3`.
pub fn legacy_db_path(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("redesmyn.sqlite3")
}

/// Rust control plane DB (sqlx): `<repo>/.redesmyn/redesmyn_rust.sqlite3`.
pub fn rust_db_path(repo_root: &Path) -> PathBuf {
    state_dir(repo_root).join("redesmyn_rust.sqlite3")
}

pub(crate) fn state_dir(base_dir: &Path) -> PathBuf {
    base_dir.join(DEFAULT_STATE_DIR_NAME)
}

pub(crate) fn default_db_path(base_dir: &Path) -> PathBuf {
    rust_db_path(base_dir)
}

pub(crate) fn default_repo_registry_dir(base_dir: &Path) -> PathBuf {
    state_dir(base_dir).join("repos")
}

pub(crate) fn default_control_plane_client_socket_path(base_dir: &Path) -> PathBuf {
    let repo_scoped = state_dir(base_dir).join(CONTROL_PLANE_SOCKET_FILE_NAME);

    #[cfg(unix)]
    {
        if unix_socket_path_fits(&repo_scoped) {
            return repo_scoped;
        }

        return fallback_control_plane_client_socket_path(base_dir);
    }

    #[cfg(not(unix))]
    {
        repo_scoped
    }
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

#[cfg(unix)]
fn unix_socket_path_fits(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt as _;

    path.as_os_str().as_bytes().len() <= UNIX_SOCKET_PATH_MAX_BYTES
}

#[cfg(unix)]
fn fallback_control_plane_client_socket_path(base_dir: &Path) -> PathBuf {
    let base_hash = stable_path_hash(base_dir);

    // Prefer the platform temp dir first; if that still exceeds the socket budget,
    // fall back to a very short `/tmp` path.
    let temp_candidate = std::env::temp_dir().join(format!(
        "{FALLBACK_SOCKET_FILE_PREFIX}{base_hash:016x}.sock"
    ));
    if unix_socket_path_fits(&temp_candidate) {
        return temp_candidate;
    }

    PathBuf::from(format!(
        "/tmp/{FALLBACK_SOCKET_FILE_PREFIX}{base_hash:016x}.sock"
    ))
}

#[cfg(unix)]
fn stable_path_hash(path: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt as _;

    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0001_0000_01b3;

    path.as_os_str()
        .as_bytes()
        .iter()
        .fold(FNV_OFFSET_BASIS, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_socket_path_is_repo_scoped_when_short() {
        let base_dir = Path::new("/tmp/redesmyn-config-test");
        let expected = base_dir
            .join(DEFAULT_STATE_DIR_NAME)
            .join(CONTROL_PLANE_SOCKET_FILE_NAME);

        assert_eq!(default_control_plane_client_socket_path(base_dir), expected);
    }

    #[cfg(unix)]
    #[test]
    fn default_socket_path_falls_back_for_long_repo_path() {
        let long_dir = format!("/tmp/{}", "a".repeat(240));
        let base_dir = PathBuf::from(long_dir);
        let path = default_control_plane_client_socket_path(&base_dir);
        let repo_scoped = state_dir(&base_dir).join(CONTROL_PLANE_SOCKET_FILE_NAME);

        assert_ne!(path, repo_scoped);
        assert!(
            unix_socket_path_fits(&path),
            "path too long: {}",
            path.display()
        );
    }

    #[cfg(unix)]
    #[test]
    fn default_socket_path_fallback_is_stable() {
        let long_dir = format!("/tmp/{}", "b".repeat(240));
        let base_dir = PathBuf::from(long_dir);

        let first = default_control_plane_client_socket_path(&base_dir);
        let second = default_control_plane_client_socket_path(&base_dir);
        assert_eq!(first, second);
    }
}
