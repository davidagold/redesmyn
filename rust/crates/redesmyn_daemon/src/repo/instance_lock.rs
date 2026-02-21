use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use fs4::fs_std::FileExt;

use crate::repo::{RepoAttachError, RepoInstanceOwner};

const LOCK_FILE_NAME: &str = "redesmyn.repo_instance.lock";
const LOCK_OWNER_FILE_NAME: &str = "redesmyn.repo_instance.owner.json";

#[derive(Debug)]
pub(crate) struct RepoInstanceLock {
    #[allow(dead_code)]
    file: std::fs::File,
    pub(crate) lock_path: PathBuf,
}

impl RepoInstanceLock {
    pub(crate) fn acquire(
        git_dir: &Path,
        owner: RepoInstanceOwner,
    ) -> Result<Self, RepoAttachError> {
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
