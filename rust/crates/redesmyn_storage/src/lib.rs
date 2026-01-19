//! Control-plane persistence (event log + projections).
//!
//! # Migrations
//!
//! SQLite migrations live in `migrations/` and are applied at runtime via
//! [`apply_migrations`]. Call [`open_sqlite_pool`] (file-backed) or
//! [`open_test_sqlite_pool`] (in-memory) to create a pool with migrations
//! applied.
//!
//! # ID mapping (ULID)
//!
//! Primary keys use [`redesmyn_ids`] newtypes stored as `BLOB(16)` (raw ULID
//! bytes). Enable `redesmyn_ids`'s `sqlx` feature to bind and decode IDs in
//! queries.

#![forbid(unsafe_code)]

mod error;

pub mod events;
pub mod sqlite;

pub use error::StorageError;
pub use sqlite::{apply_migrations, in_transaction, open_sqlite_pool, open_test_sqlite_pool};
