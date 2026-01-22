use std::path::{Path, PathBuf};

use redesmyn_protocol::artifacts::{ArtifactKind, ArtifactRef, StorageHint};
use tokio::io::AsyncWriteExt as _;

#[derive(Debug, thiserror::Error)]
pub enum ArtifactStoreError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Minimal local artifact store for daemon-owned artifacts.
///
/// Notes:
/// - Artifacts are stored on disk to avoid unbounded memory growth.
/// - Paths remain daemon-local; cross-boundary references use `blob_key` hints.
#[derive(Debug, Clone)]
pub struct LocalArtifactStore {
    root: PathBuf,
}

impl LocalArtifactStore {
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub async fn ensure_dirs(&self) -> Result<(), ArtifactStoreError> {
        tokio::fs::create_dir_all(self.root.join("artifacts")).await?;
        Ok(())
    }

    pub fn artifact_path(&self, artifact_id: redesmyn_ids::ArtifactId) -> PathBuf {
        self.root
            .join("artifacts")
            .join(format!("{artifact_id}.bin"))
    }

    #[must_use]
    pub fn blob_key(artifact_id: redesmyn_ids::ArtifactId) -> String {
        format!("artifact/{artifact_id}")
    }

    pub async fn create_writer(
        &self,
        kind: ArtifactKind,
        mime: Option<String>,
    ) -> Result<LocalArtifactWriter, ArtifactStoreError> {
        let artifact_id = redesmyn_ids::ArtifactId::new();
        let path = self.artifact_path(artifact_id);
        let file = tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .await?;

        Ok(LocalArtifactWriter {
            artifact_id,
            kind,
            mime,
            bytes_written: 0,
            file,
            path,
        })
    }

    pub async fn store_bytes(
        &self,
        kind: ArtifactKind,
        mime: Option<String>,
        bytes: &[u8],
    ) -> Result<ArtifactRef, ArtifactStoreError> {
        let mut writer = self.create_writer(kind, mime).await?;
        writer.write_all(bytes).await?;
        Ok(writer.finish())
    }

    pub async fn remove_artifact(
        &self,
        artifact_id: redesmyn_ids::ArtifactId,
    ) -> Result<(), ArtifactStoreError> {
        let path = self.artifact_path(artifact_id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err.into()),
        }
    }

    pub fn contains_artifact(&self, artifact_id: redesmyn_ids::ArtifactId) -> bool {
        self.artifact_path(artifact_id).exists()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

pub struct LocalArtifactWriter {
    artifact_id: redesmyn_ids::ArtifactId,
    kind: ArtifactKind,
    mime: Option<String>,
    bytes_written: u64,
    file: tokio::fs::File,
    path: PathBuf,
}

impl LocalArtifactWriter {
    pub async fn write_all(&mut self, bytes: &[u8]) -> Result<(), ArtifactStoreError> {
        self.file.write_all(bytes).await?;
        self.bytes_written = self
            .bytes_written
            .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        Ok(())
    }

    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    #[must_use]
    pub fn artifact_id(&self) -> redesmyn_ids::ArtifactId {
        self.artifact_id
    }

    #[must_use]
    pub fn finish(self) -> ArtifactRef {
        ArtifactRef {
            artifact_id: self.artifact_id,
            kind: self.kind,
            content_hash: None,
            byte_len: Some(self.bytes_written),
            mime: self.mime,
            storage_hint: Some(StorageHint::BlobKey {
                blob_key: LocalArtifactStore::blob_key(self.artifact_id),
            }),
        }
    }

    pub async fn discard(self) -> Result<(), ArtifactStoreError> {
        drop(self.file);
        tokio::fs::remove_file(&self.path).await?;
        Ok(())
    }
}
