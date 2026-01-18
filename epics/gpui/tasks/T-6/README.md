---
epic: gpui
branch:
  suggested: rn/gpui/T-6-sqlx-storage-scaffold
rn:
  parent: T-2
---

# T-6 Control-plane storage scaffold (`sqlx` + SQLite) (Domain 0)

## Problem

The Rust control plane will own authoritative state (event log + projections). We want:

- good performance,
- clean schema evolution,
- and predictable query behavior.

We have selected `sqlx`.

## Goal

Create a `redesmyn_storage` crate that provides:

- a `sqlx`-managed SQLite database layer,
- migration scaffolding,
- clear boundaries so other crates don’t write SQL ad-hoc.

## Requirements

### 1) Pool + transaction helpers

- Provide a `SqlitePool` setup function and common transaction helpers.
- Provide deterministic test DB creation (temporary files or in-memory with migrations).

### 2) Migration strategy

- Add a migrations directory for the Rust DB schema.
- Document how migrations are applied in dev/test/prod.

We will keep Alembic for the legacy Python DB until we switch; this ticket does not migrate the existing schema.

### 3) Minimal schema stub (for validation)

Create the minimal tables needed to validate the scaffolding:

- e.g., an `events` table suitable for an append-only event log (schema can evolve later).
- do not over-design; keep it minimal but real.

### 4) Type integration

- Use `redesmyn_ids` newtypes for primary keys.
- Define clear mapping rules for ULIDs (`BLOB(16)`).

## Acceptance criteria

- `cargo test -p redesmyn_storage` applies migrations and runs at least one query.
- The crate provides a clean API surface so control-plane code doesn’t scatter SQL.
- Migration workflow is documented.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-1/README.md` (workspace + crate skeletons).
- Depends on `epics/gpui/tasks/T-2/README.md` (ULID/newtypes) for key types.

## Reference implementation (today; DB + migrations)

- Storage layer (Python today):
  - `redesmyn/db/models.py` (SQLAlchemy ORM; current domain source of truth).
  - `redesmyn/db/session.py` (engine/session creation; Alembic upgrade/init policy; SQLite pragmas).
  - `redesmyn/db/migrate.py` + `redesmyn/db/alembic/versions/` (migrations).
- Tests (Python today):
  - `tests/test_migrations_db.py`
  - `tests/test_sqlite_pragmas.py`
