use std::path::{Path, PathBuf};

use redesmyn_ids::{HostId, HostInstanceId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostIdentity {
    pub host_id: HostId,
    pub host_instance_id: HostInstanceId,
}

impl HostIdentity {
    #[must_use]
    pub fn new(host_id: HostId) -> Self {
        Self {
            host_id,
            host_instance_id: HostInstanceId::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HostIdentityError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("invalid host id in {path}: {value}")]
    InvalidHostId { path: PathBuf, value: String },
}

pub fn host_id_path(state_dir: &Path) -> PathBuf {
    state_dir.join("host_id")
}

pub fn load_or_create_host_id(state_dir: &Path) -> Result<HostId, HostIdentityError> {
    std::fs::create_dir_all(state_dir)?;

    let path = host_id_path(state_dir);
    if path.exists() {
        let raw = std::fs::read_to_string(&path)?;
        let value = raw.trim();
        return value
            .parse::<HostId>()
            .map_err(|_| HostIdentityError::InvalidHostId {
                path,
                value: value.to_string(),
            });
    }

    let host_id = HostId::new();
    std::fs::write(&path, format!("{host_id}\n"))?;
    Ok(host_id)
}

