# `redesmyn_storage`

Control-plane persistence (SQLite via `sqlx`).

## Schema (v0)

Core tables:

- Graph: `workspaces`, `repositories`, `epics`, `tasks`, `task_relations`
- Orchestration: `commands`, `command_updates`, `events`
- Sessions: `session_events`, `artifacts`
- Presence: `hosts`, `daemon_presence`

Primary indices (selected):

- `tasks`: by `epic_id`, by `parent_task_id`, and unique `(epic_id, local_ref)` when `local_ref` is
  present.
- `commands`: by scope (`scope_kind`, `scope_workspace_id`, `scope_repo_id`), by `state`, and by
  `target_task_id`.
- `session_events`: by `session_id`/time and by `task_id`/time.
- `events`: by scope/time (used by the event log subscription surface).

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
