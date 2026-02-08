use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("failed to create database directory: {path}")]
    CreateDbDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("legacy DB not found: {path}")]
    LegacyDbNotFound { path: PathBuf },

    #[error("failed to snapshot legacy DB: {path}")]
    LegacyDbSnapshot {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("legacy repository row not found for repo root: {repo_root}")]
    LegacyRepoNotFound { repo_root: PathBuf },

    #[error("conflict: {message}")]
    Conflict { message: String },

    #[error("invalid data: {message}")]
    InvalidData { message: String },

    #[error(transparent)]
    Migrate(#[from] sqlx::migrate::MigrateError),

    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
}
