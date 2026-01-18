---
epic: gpui
branch:
  suggested: rn/gpui/T-28-observation-telemetry
rn:
  parent: T-26
---

# T-28 Repo observation + telemetry + snapshots (Domain 3)

## Problem

The UI and CLI depend on near-realtime, correct repo-derived state:

- ref movements and new commits
- worktree health (exists/dirty/branch mismatch)
- projections (merge base, trunk timeline, “stack in sync”, etc.)

This data must be produced by the daemon (repo executor) and sent to the control plane without the control plane touching the repo filesystem.

If telemetry is ad-hoc or polling-heavy, we risk:

- laggy UI,
- high CPU from repeated git calls,
- and brittle resync behavior.

## Goal

Implement a per-attached-repo observation loop that:

- computes a minimal set of repo-derived facts each tick,
- emits compact events and periodic snapshots via the daemon stream protocol,
- is efficient (batched git queries; minimal work when nothing changed),
- and supports resync for reconnect and backpressure recovery.

## Requirements

### 1) Observation state machine

Maintain per-repo observation state:

- last seen ref state (branch tips)
- last emitted event cursor (local)
- last snapshot time

Ensure:

- first tick can optionally emit a baseline snapshot,
- subsequent ticks emit only deltas (events) when possible.

### 2) Event set (initial)

Emit compact, typed events (no blobs), such as:

- `git.commit` (new commit observed on a task branch)
- `worktree.health` (worktree status changes)
- `repo.executor_status` (lease/primary + attachment status changes)

For unknown/new event types, include an `UnknownEvent` fallback (T-11/T-14).

### 3) Snapshots

Support snapshot emission and on-demand resync:

- periodic `TelemetrySnapshot` for key projections (schema minimal initially)
- respond to control-plane `ResyncRequest` with a snapshot

### 4) Performance

- Use the git backend abstraction (T-26) and batch queries where possible.
- Avoid scanning the entire repo on every tick.
- Tick interval should be configurable; default should be reasonable for interactive use.

### 5) Backpressure and loss recovery

- Telemetry publishing must be bounded (do not allow unbounded queues).
- If the control plane falls behind, emit a resync signal and fall back to snapshot recovery rather than trying to replay an unbounded backlog.

### 6) Testability

Provide deterministic tests that:

- set up a temp repo and worktree,
- make commits and ref moves,
- run one observation tick at a time,
- and assert emitted events/snapshots without sleeps.

## Acceptance criteria

- Observation loop emits correct events and snapshots for basic repo changes.
- Resync behavior works (on request and on backlog/overflow).
- Implementation is efficient and structured for future expansion (more projections, more event types).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on git backend abstraction (T-26) and repo attachment (T-24).
- Integrates with lease status (T-25) and worktree health (T-27).
- Consumed by control-plane ingestion and projections (Domain 2).

## Reference implementation (today; observation/telemetry orientation only)

- Observation loop (Python today):
  - `redesmyn/repo_observer.py` (polling loop + observation state).
  - `redesmyn/git_telemetry.py` / `redesmyn/git_projections.py` (projection computation and persistence).
  - `redesmyn/db/models.py` (projection tables such as `git_ref_states`, `git_trunk_timelines`, `git_merge_bases`, and by-instance variants).
- Telemetry transport today (Python):
  - `redesmyn/ws_protocol.py` (`DaemonEvent` with `event_type` + `data`).
  - `redesmyn/api.py` `daemon_ws()` handler ingests `DaemonEvent` and appends it to the event log for UI streaming.
- UI consumption today (TS):
  - `dashboard/src/hooks/useEventStream.ts` (live events)
  - `dashboard/src/api/useEpicCacheSync.ts` (event-driven cache invalidation)
- Tests (Python today):
  - There are limited direct “repo observer” tests today; closest coverage is:
    - `tests/test_epic_graph.py` (trunk timeline presence and host-key scoping),
    - `tests/test_daemon_ws_runtime_integration.py` (daemon events ingested and reflected in merge run state).
