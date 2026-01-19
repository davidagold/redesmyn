# `redesmyn_storage`

Control-plane persistence (SQLite via `sqlx`).

## Migrations

- Migrations live in `rust/crates/redesmyn_storage/migrations/`.
- They are applied at runtime via `redesmyn_storage::apply_migrations`.

### Dev / prod

The control plane (server/desktop host) should call `open_sqlite_pool(db_path).await?` during
startup. This creates the SQLite pool and applies all pending migrations from `migrations/` (via
`sqlx::migrate!()` embedded migrations).

### Tests

Use `open_test_sqlite_pool().await?` for a deterministic in-memory database with migrations
applied.

### Adding migrations

Create a new `migrations/<timestamp>_<name>.sql` file (the format produced by `sqlx migrate add`).

## ID mapping (ULID)

All primary keys use `redesmyn_ids` newtypes and are stored in SQLite as `BLOB(16)` (raw ULID
bytes). Bind IDs directly in `sqlx` queries (requires the `redesmyn_ids/sqlx` feature).
