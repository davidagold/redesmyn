//! Artifact references: stable addresses to out-of-band content (T-14).
//!
//! Protocol messages and durable session events must reference large content via
//! `ArtifactRef` rather than embedding full blobs directly.

use redesmyn_ids::ArtifactId;

/// Small, stable artifact kind enum. Additive changes are allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Log,
    Diff,
    Patch,
    FileSnapshot,
    Trace,
    /// A kind not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// Content hash for integrity/dedup (algorithm + digest bytes).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Hash {
    pub algorithm: String,
    pub digest: Vec<u8>,
}

/// Storage backend is an implementation detail; hints are best-effort.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StorageHint {
    LocalPath {
        local_path: String,
    },
    BlobKey {
        blob_key: String,
    },
    /// A storage hint not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

/// Stable reference to content stored out-of-band.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactRef {
    pub artifact_id: ArtifactId,
    pub kind: ArtifactKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<Hash>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_len: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_hint: Option<StorageHint>,
}
