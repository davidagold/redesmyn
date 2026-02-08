---
epic: gpui
branch:
  suggested: rn/gpui/T-17-control-plane-db-schema
rn:
  node:
    branch: rn/gpui/T-17-control-plane-db-schema
  parent: T-6
---

# T-17 Control plane DB schema + migrations (Domain 2)

## Problem

The control plane needs durable, queryable state for:

- epics/tasks/graph topology (desired state),
- command lifecycle (no silent actions),
- an append-only event log (audit + realtime),
- structured session events and artifact references (for the session viewer + diff UX),
- and daemon/host presence (projection).

We also require:

- strong typing (ULID/newtypes),
- fast queries (zippy CLI, smooth UI),
- and evolvable schema (migrations).

## Goal

Define the initial Rust/`sqlx` SQLite schema and migrations for the control plane, aligned with the Domain 1 protocol contracts (T-9..T-15).

This ticket is about schema + migration workflow, not about implementing every API.

## Requirements

### 1) ID strategy

- All primary identifiers are ULID newtypes.
- SQLite storage uses `BLOB(16)` for ULID keys (with helper functions in code for encode/decode).

### 2) Core tables (initial set)

Define and migrate the minimal-but-real set of tables:

- `workspaces`
- `repositories` (scoped by workspace)
- `epics` (scoped by repository)
- `tasks` (scoped by epic)
  - include:
    - `parent_task_id` (tree topology)
    - stable “human ref” (e.g., `local_ref` like `T-16`) if present
    - `branch_name` (desired branch backing)
    - state fields needed by UI (merge readiness, etc.) as they stabilize
- `task_relations` (optional in v0; if included, use for “after” constraints)
- `commands` (control plane authoritative record of user/daemon commands)
- `command_updates` (append-only state transitions / progress for observability)
- `events` (append-only domain event log; compact payloads + typed discriminator)
- `session_events` (structured, queryable agent/session output per T-14)
- `artifacts` (artifact refs + metadata per T-14)
- `hosts` / `daemon_presence` (projection; keep minimal)

Notes:

- Do not embed large blobs in `events` or `session_events`; use `artifacts` for big content.
- Prefer normalized columns for frequently queried fields; keep “extra JSON” as an escape hatch only where necessary.

### 2.1) DB-level consistency constraints (prefer constraints over app logic)

Prefer enforcing cross-column invariants in SQLite where practical so downstream code can assume the
graph is internally consistent.

In particular:

- **Repo scope columns are a single logical identity**:
  - Tables that store both `scope_workspace_id` and `scope_repo_id` should prefer a *composite* FK
    like `(scope_workspace_id, scope_repo_id) → repositories(workspace_id, id)` (instead of two
    independent FKs), to prevent “workspace A + repo B” impossible pairings.
- **Tree topology should be internally consistent**:
  - Prefer enforcing `tasks.parent_task_id` never crosses epics via a composite FK
    `(epic_id, parent_task_id) → tasks(epic_id, id)`.
- **Scope chains should be coherent where representable**:
  - When storing `session_events` with `scope_kind = epic|task`, prefer constraints that ensure
    referenced `epic_id`/`task_id` belong to the same repo scope (often via additional composite
    FKs or redundant columns).

Non-goals for v0 (unless required by real bugs):

- Invariants that require multi-table join logic (e.g. “`commands.target_task_id` must belong to
  `scope_repo_id`”) usually require triggers or denormalization; treat these as application-layer
  invariants until we choose the minimal redundant columns needed to enforce them cheaply.

### 3) Indexing policy

Add indices for the primary query patterns:

- tasks by epic, tasks by parent, tasks by `local_ref`
- commands by scope, commands by state, commands by target task
- session events by task/session/time
- events by scope/time

### 4) Migration workflow

- Migrations are applied automatically by the control plane at startup in dev/test.
- Provide a clear, documented workflow for adding migrations (no manual DB fiddling).

Split DB strategy:

- Rust migrations apply to the Rust DB file only: `<repo>/.redesmyn/redesmyn_rust.sqlite3`.
- Do not modify the legacy Alembic DB (`<repo>/.redesmyn/redesmyn.sqlite3`) from Rust code.
- Provide an explicit, testable import/cutover tool as a separate ticket (T-67).

### 5) Testability

- Provide a deterministic way to create a temporary DB and run migrations for tests.
- Add a small set of schema-level tests (migrations apply; basic inserts/selects work with ULID keys).

## Acceptance criteria

- `cargo test -p redesmyn_storage` (or equivalent) applies migrations and can insert/query core rows using ULID keys.
- The schema supports the contracts for:
  - command lifecycle observability,
  - session events + artifacts,
  - and graph queries.
- Indexes exist for the primary access patterns (documented in the ticket or code).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-2/README.md` (ULID/newtypes) and `epics/gpui/tasks/T-6/README.md` (sqlx scaffold).
- Informed by Domain 1 contracts (T-9..T-15).

## Reference implementation (today; schema/migration orientation only)

- ORM schema (Python today):
  - `redesmyn/db/models.py` (SQLAlchemy ORM; includes tables like `tasks`, `agent_sessions`, `merge_runs`, `events`, `repo_executor_leases`, etc.).
  - `redesmyn/domain/enums.py` (enum values persisted to DB today).
- Migrations (Python today, Alembic):
  - `alembic.ini`
  - `redesmyn/db/migrate.py`
  - `redesmyn/db/alembic/` (migration scripts)
  - `scripts/check_migrations.py`
- Tests (Python today):
  - `tests/test_migrations_db.py` (migrations apply and match ORM tables).
  - `tests/test_sqlite_pragmas.py` (SQLite settings expectations).
