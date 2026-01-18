---
epic: gpui
branch:
  suggested: rn/gpui/T-25-lease-primary-enforcement
rn:
  parent: T-23
---

# T-25 Lease/primary executor management + enforcement (Domain 3)

## Problem

We must avoid “two writers” executing git mutations against the same repo identity:

- multiple daemons could exist (two laptops, stale processes, etc.),
- only one must be allowed to execute mutating repo commands at a time.

We already have a conceptual lease/primary model. In the Rust port, this must be:

- explicit in the protocol,
- enforced in the daemon,
- and observable for UI/CLI (no confusing failures).

## Goal

Implement the daemon side of lease/primary behavior:

- acquire/refresh/lose primary status per repo scope,
- enforce “mutations require primary”,
- and emit clear status/telemetry about lease ownership.

This ticket focuses on daemon behavior and protocol usage; control-plane policy/routing is implemented in Domain 2/3 follow-ups.

## Requirements

### 1) Lease model

Define a lease model that includes:

- `repo_scope`
- `primary_host_id`
- `lease_expires_at`

Daemon must track:

- whether it is primary for each attached repo scope,
- when it must renew,
- and when it must stop executing mutating operations.

### 2) Acquisition/renewal protocol

Use the daemon stream protocol (T-11) to:

- request lease acquisition/renewal, OR
- receive lease assignments from the control plane.

Pick a single approach for v0 and document it.

Preference:

- control plane is authoritative; daemon requests renewal and control plane grants/denies.

### 3) Enforcement

For repo-mutating commands (merge/restack/worktree writes/etc):

- if daemon is not primary, it must reject the command with a structured error:
  - category: conflict or unavailable
  - message: “Not primary executor for repo; primary is <host_id>”
  - include enough detail for UI to render a helpful next step.

### 4) Graceful lease loss

If the daemon loses primary while executing:

- it must stop accepting new mutating commands immediately,
- and for in-flight operations:
  - finish the current safe boundary if possible, or
  - fail the command with a clear “lost lease” error.

### 5) Observability

Emit status updates that allow UI/CLI to display:

- current primary for each repo scope,
- lease freshness/expiry horizon,
- and whether the daemon is eligible to execute.

## Acceptance criteria

- Daemon enforces primary requirements correctly for mutating operations.
- Lease renewal logic is robust and testable.
- UI/CLI can surface “who is primary” and “why my command was rejected” without guesswork.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on daemon skeleton (T-23) and daemon protocol contract (T-11).
- Coordinated with Domain 2 command routing (T-19) once implemented.

## Reference implementation (today; lease orientation only)

- Lease model and helpers (Python today):
  - `redesmyn/db/models.py` (`RepoExecutorLease` table).
  - `redesmyn/repo_executor_leases.py` (`acquire_or_refresh_primary`, `get_primary_host_key`, etc.).
- Where leases are used today (Python):
  - `redesmyn/api.py`:
    - lifespan task periodically refreshes the local lease in `runner_mode=local`,
    - `GET /v1/epics/{epic}/graph` reacquires lease on-demand in local mode if missing.
    - `daemon_ws()` heartbeat handler refreshes lease when a daemon reports the repo as attached.
- Tests (Python today):
  - `tests/test_repo_executor_lease_local_fallback.py` (epic graph reacquires primary lease in local mode).
  - `tests/test_epic_graph.py` (trunk timeline keyed by host/lease).
