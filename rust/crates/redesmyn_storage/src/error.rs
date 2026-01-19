use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("failed to create database directory: {path}")]
    CreateDbDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid data: {message}")]
    InvalidData { message: String },

    #[error(transparent)]
    Migrate(#[from] sqlx::migrate::MigrateError),

    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
}
