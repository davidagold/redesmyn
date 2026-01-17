---
id: T-6
stacked_on: T-1
node:
  branch: rn/revise-architecture/T-6-merge-node-into-task
linear:
  issue_id: 531f7793-b301-4746-bdfb-b516946e2402
  identifier: RED-35
---

# T-6 Merge Node into Task (single graph primitive)

## Brief (local)

Consolidate “graph node” concepts by merging `Node` into `Task` so the graph has a single primitive:

- Move branch/topology/worktree/assignment metadata into `Task` (or `Task`-scoped companion tables where needed).
- Ensure git-derived “projection” fields used by the UI move with the graph primitive (e.g. `stackInSync` should not remain on a removed `Node` API).
- Migrate foreign keys and payloads that currently point at `nodes.*`:
  - sessions should target tasks
  - events referencing nodes should reference tasks
- Remove the `Node` API surface; the UI operates on tasks and their relationships.
- Preserve existing data through a DB migration; provide an upgrade path for local dev DBs.
- Invest in Alembic infrastructure (needed for this migration and future ones):
  - `just` targets for revision/upgrade/downgrade/history/current
  - a predictable migrations directory and a clear “dev DB” vs “test DB” story

## Acceptance Criteria

- There is no longer a separate “node” concept in public API/UI.
- The data model has a single graph primitive (tasks) with topology/branch metadata.
- Agent sessions and telemetry can be attributed to tasks without indirection.

## Updates

### 2025-12-31

- Recent work added `stackInSync` to the epic graph node payload to surface “left-behind/out-of-sync” child branches.
  Once Node is merged into Task, this field (and its derivation/telemetry) must be moved so the UI/API does not retain a node concept.

### 2026-01-01

- During stack merge/rebase work (with T-2 merged into `main`), this branch also refactored daemon presence + command delivery to reduce architectural drift while removing `Node`.
- Presence/liveness:
  - `/v1/daemons` now reports `connected` from the in-process WebSocket registry (no DB staleness heuristic).
  - Persisted presence is reduced to `hosts.last_seen_at` (rate-limited updates) so the UI can show “last seen” even when offline.
  - Added explicit multi-instance notes in the runtime registry about what must change for multi-server deployments.
- Identity alignment:
  - Daemon identity is `hosts.host_key` end-to-end (daemon WS hello, command routing, and daemon-emitted event attribution).
- Command ack data:
  - `DaemonCommand.data` remains the issued payload; daemon acks are stored in `DaemonCommand.ack_data` to avoid overwriting the original command.
  - Added Alembic migrations for daemon identity/presence and the command ack payload split (`0003_daemon_presence_host_key`, `0004_daemon_command_ack_data`).
