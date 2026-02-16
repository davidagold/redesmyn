---
epic: gpui
branch:
  suggested: rn/gpui/T-25-lease-primary-enforcement
rn:
  node:
    branch: rn/gpui/T-25-lease-primary-enforcement
  parent: T-23
---

# T-25 Lease/primary executor management + enforcement (Domain 3)

## Problem

T-24 already protects repo-local safety by enforcing one writer per **repo instance** via attach lock.

Additionally, some operations are **repo-scope global** (e.g. “the” merge-queue/trunk executor) and need a
single-writer policy even when multiple repo instances exist across different hosts.

We already have a conceptual lease/primary model. In the Rust port, this must be:

- explicit in the protocol,
- enforced in the daemon,
- and observable for UI/CLI (no confusing failures).

If a deployment has only one active executor for a repo scope (for example a single embedded local daemon), this
lease layer can be trivial or dormant; it must not block normal repo-instance-local operations.

## Goal

Implement the daemon side of lease/primary behavior for **repo-scope global** mutations:

- acquire/refresh/lose primary status per repo scope,
- enforce “commands that require primary must run on the primary”,
- and emit clear status/telemetry about lease ownership.

This ticket does **not** make “primary” a global mutex across all git work everywhere:

- repo **instance** exclusivity is enforced via attach locks (T-24),
- multiple daemons may work in parallel on different repo instances for the same `RepoScope`,
- and the control plane is responsible for routing commands to the intended executor/instance.

This ticket focuses on daemon behavior and protocol usage; control-plane policy/routing is implemented in Domain 2/3 follow-ups.

## Requirements

### 0) Terminology + scope

- **Repo scope**: logical identity `RepoScope { workspace_id, repo_id }`.
- **Repo instance**: a particular checkout on a host (repo root / `.git` directory).

Rules:

- Leases are for **repo-scope global** operations (where “only one primary” is a product decision).
- Leases are **not** a substitute for repo instance exclusivity (T-24).
- Command kinds must explicitly declare whether they require repo-scope primary (`requires_repo_primary`).
- Default policy is `requires_repo_primary = false` unless explicitly documented otherwise.

### 1) Lease model (for repo-scope singleton ops)

Define a lease model that includes:

- `repo_scope`
- `primary_host_id`
- `lease_expires_at`
- `lease_fencing_token` (monotonic; required to fence stale primaries)

Daemon must track:

- whether it is primary for each attached repo scope,
- when it must renew,
- and when it must stop executing `requires_repo_primary = true` operations.

When no active command kinds require repo-scope primary, lease maintenance may be a no-op.

Note: `lease_expires_at` alone is not sufficient to prevent “two writers” (clock skew + renew races). A
control-plane-issued fencing token (monotonic generation) allows deterministic rejection of stale primaries and
stale queued commands even when leases overlap in time.

### 2) Acquisition/renewal protocol

Use the daemon stream protocol (T-11) to:

- request lease acquisition/renewal, OR
- receive lease assignments from the control plane.

Pick a single approach for v0 and document it.

Preference:

- control plane is authoritative; daemon requests renewal and control plane grants/denies.

Lease grant/renewal messages must include the current `lease_fencing_token`. The daemon must treat it as part
of its “am I primary?” state, and it must be plumbed through mutating command routing so callers can be fenced
if their view of the lease is stale.

### 3) Enforcement

For commands that require a repo-scope primary (e.g. merge-queue/trunk mutations, “blessed” restack/merge
executor operations):

- if daemon is not primary, it must reject the command with a structured error:
  - category: conflict or unavailable
  - message: “Not primary executor for repo; primary is <host_id>”
  - include enough detail for UI to render a helpful next step.

For repo-instance-local mutations that do **not** require a repo-scope primary:

- do not reject solely because the daemon is not primary,
- but still require repo attachment + instance exclusivity (T-24),
- and fence execution via command-level preconditions (expected ref/sha “CAS-style fencing”) where applicable.

Additionally, for any mutating command with `requires_repo_primary = true` that includes a
`lease_fencing_token` in request metadata:

- if the token does not match the daemon’s current lease token, the daemon must reject with a clear “lost
  lease / stale lease” error (include expected vs received tokens).

### 4) Graceful lease loss

If the daemon loses primary while executing:

- it must stop accepting new `requires_repo_primary = true` mutating commands immediately,
- and for in-flight operations:
  - finish the current safe boundary if possible, or
  - fail the command with a clear “lost lease” error.

Repo-instance-local commands that do not require primary are not canceled solely due to lease loss.

### 5) Observability

Emit status updates that allow UI/CLI to display:

- current primary for each repo scope,
- lease freshness/expiry horizon,
- and whether the daemon is eligible to execute.

Status should distinguish:

- `primary_not_required` (for command kinds that do not require it),
- `primary_unknown`,
- `primary_owned`,
- `primary_other`.

## Acceptance criteria

- Daemon enforces primary requirements correctly for commands that require a repo-scope primary.
- Repo-instance-local mutations are not blocked by a global “primary” concept.
- Lease renewal logic is robust and testable.
- UI/CLI can surface “who is primary” and “why my command was rejected” without guesswork.
- Command kinds without `requires_repo_primary` execute under T-24 attachment/lock semantics even when no lease exists.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on daemon skeleton (T-23) and daemon protocol contract (T-11).
- Coordinated with Domain 2 command routing (T-19) once implemented.
- Required before enabling any command kind with `requires_repo_primary = true`.

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
