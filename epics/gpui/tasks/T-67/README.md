---
epic: gpui
branch:
  suggested: rn/gpui/T-67-legacy-db-import
rn:
  node:
    branch: rn/gpui/T-67-legacy-db-import
  parent: T-17
---

# T-67 Legacy DB → Rust DB import + cutover tooling (Domain 2)

## Problem

During the port we will have **two** databases:

- Legacy Python DB (Alembic): `<repo>/.redesmyn/redesmyn.sqlite3`
- Rust control plane DB (sqlx): `<repo>/.redesmyn/redesmyn_rust.sqlite3`

We need a principled, low-risk path for:

- existing repos/users with legacy state to move to the Rust DB, and
- developers to test the Rust control plane without corrupting legacy data.

If we try to have two migration systems modify one DB, or if the Rust control plane silently “upgrades” the legacy DB, we will create non-determinism, hard-to-debug corruption, and irreproducible environments.

## Goal

Provide an explicit, testable import/cutover mechanism that:

- creates the Rust DB (if missing),
- applies Rust `sqlx` migrations,
- optionally imports selected legacy state (v0: graph + minimal session preview),
- and leaves the legacy DB untouched as a backup/forensics artifact.

This is a migration *tooling* ticket: a split-codebase bridge, not the full control plane implementation.

## Requirements

### 1) Discovery + safety rules

Define a deterministic discovery policy:

- Rust DB path (default): `<repo>/.redesmyn/redesmyn_rust.sqlite3`
- Legacy DB path (default): `<repo>/.redesmyn/redesmyn.sqlite3`

The `rn-rs` CLI also supports overriding these paths for dev workflows (e.g. importing into a
throwaway Rust DB file while preserving the “official” cutover DB).

Safety rules:

- Never write to the legacy DB.
- Never delete the legacy DB.
- Import must be explicitly triggered (CLI and/or a clearly visible UI affordance), not a silent background action.

Implementation note (v0): the importer snapshots the legacy SQLite file (and `-wal`/`-shm` if present) into a
temporary directory and opens the snapshot in read-only mode.

### 2) Import surface (AI-first + CLI-first)

Provide a developer-facing import surface, at minimum via Rust CLI:

- `rn-rs db import-legacy [--repo <path>] [--legacy-db-path <path>] [--rust-db-path <path>] [--dry-run] [--json]`

Requirements:

- `--dry-run` reports what would be imported (counts, tables, schema version) without writing.
- `--json` (or `--output json`) produces machine-readable output suitable for CI/agent verification.
- Exit codes follow the shared error conventions (T-3).

Future direction (not required in this ticket):

- a desktop UI flow that guides the user through import and clearly shows progress.

### 3) What to import (initial scope)

Start minimal. Import only what provides immediate value for continuity and UI/UX:

- workspace/repo/epic/task metadata needed to render the graph:
  - repo + epic slugs/titles,
  - task title, parent topology, branch name,
  - stable human refs like `T-123` when derivable from `local_path` (best-effort),
  - task state when a Rust `tasks.state` column exists (forward-compatible),
- one best-effort per-task session preview event (latest legacy session by id) using legacy `agent_preview.last_assistant_message_preview`,
  - plus the legacy `turn_id` when present.

Notes:

- The legacy DB uses integer primary keys; the Rust DB uses ULID newtypes. The importer must define an explicit mapping strategy.
- Future direction (not required in v0): import legacy `events` as best-effort history feed, preserving the original `event_type` string.

### 4) ID mapping strategy (explicit)

Define one of these strategies (pick one; document rationale):

1. Store legacy IDs as dedicated columns (e.g. `legacy_task_id INTEGER`) on imported core tables.
2. Maintain a separate `legacy_id_map` table keyed by `(table_name, legacy_id) -> new_ulid`.

Requirements:

- Import is idempotent (re-running does not duplicate rows).
- Mapping is queryable for debugging and support.

Implementation note (v0): `legacy_id_map` is used for:

- `repositories`, `epics`, `tasks` (graph core)
- `agent_sessions` (session id)
- `agent_session_preview_events` (distinct session event ids; do not reuse `session_id` as `event_id`)

Important: ULIDs are stable only as long as the Rust DB file (and therefore `legacy_id_map`) is reused.
If you import into a fresh Rust DB file, the newly generated ULIDs will differ (which is fine for a
throwaway dev DB).

### 5) Tests

Add a deterministic test that:

- creates a tiny legacy SQLite DB fixture with a small subset of tables/rows (or uses a minimal snapshot),
- runs the importer into a temp Rust DB,
- asserts imported counts and key invariants (graph topology preserved, mapping stable),
- and verifies the legacy DB is unchanged.

## Acceptance criteria

- A developer can run an explicit import command and end up with a valid Rust DB.
- Import emits deliberate `tracing` logs (start/end + counts + errors), without noisy per-row logs.
- The process is safe and repeatable (idempotent; no legacy DB mutation).

## How to run (dev)

- Dry run (plan only): `rn-rs db import-legacy --dry-run --json`
- Apply import: `rn-rs db import-legacy`
- Dev DB import (throwaway Rust DB file): `rn-rs db import-legacy --rust-db-path <repo>/.redesmyn/redesmyn_rust_dev.sqlite3`
- “Official”/cutover import (stable ULIDs across re-runs): `rn-rs db import-legacy --rust-db-path <repo>/.redesmyn/redesmyn_rust.sqlite3`
- From the repo root via Cargo: `cargo run -p rn -- db import-legacy --dry-run --json`

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-17/README.md` (Rust schema + migrations).
- Uses `rn-rs` as the initial entrypoint (T-8).
- Informs the desktop onboarding/runtime story once the control plane is embeddable (T-16).

## Reference implementation (today; behavior orientation only)

- Legacy DB schema + migrations:
  - `redesmyn/db/models.py`
  - `redesmyn/db/alembic/`
  - `scripts/check_migrations.py`
- Legacy “repo-scoped state dir” convention:
  - `<repo>/.redesmyn/` (contains DB, tasks/, worktrees/, etc.)
