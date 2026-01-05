# T-2 Git query fastpath when a primary executor is available

## Metadata

```yaml
id: T-2
epic: backlog
stacked_on:
  - (none)
branch:
  suggested: rn/backlog/T-2-git-query-fastpath
```

## Problem

Today the API often relies on **projections** (telemetry snapshots) for git-derived data:

- trunk timelines / commit lists
- merge bases / ahead-behind counts
- ref state + “out of sync” heuristics

This is necessary when the control plane cannot run git and no executor is attached, but it creates a tension:

- In local-first mode (or when a primary executor is connected and fresh), we *could* answer many read-only queries directly against the repo checkout.
- Projections can be stale and introduce “why is the UI wrong?” moments during rapid iteration.
- Some actions (merge/restack) already require an executor; the UI would be better if read-only git queries could also leverage the executor when available.

## Goal

Evaluate and (if worthwhile) implement a **fastpath** for git-related read queries when the primary executor is available and fresh.

This should reduce perceived staleness without breaking the remote/control-plane architecture.

## Proposed approach (v0)

### 1) Define which queries are candidates

Start with queries that are:

- read-only
- needed for UI correctness (graph / commit UI)
- already present in projections

Likely candidates:

- trunk timeline (commit list + metadata)
- merge base computation for task edges
- ref state for branch tips

### 2) Add an executor-backed read seam

We already have the `RepoExecutor` abstraction for mutations.
Introduce an explicit way to perform read-only “repo queries”:

- either extend `RepoExecutor` with a small “query” surface (preferred if it fits),
- or add a dedicated `RepoQueryExecutor` interface that is implemented by:
  - local executor (direct git calls)
  - daemon executor (server issues a plan/query command; daemon responds with results)

Important: keep the surface minimal. This is not a general git RPC.

### 3) Routing rules

When serving a git-derived API response:

- If primary executor exists and telemetry is fresh:
  - prefer executor fastpath for the specific sub-queries that benefit
  - fall back to projections if the fastpath fails or times out
- If no executor is available:
  - use projections only (current behavior)

### 4) UX / performance considerations

- The fastpath should not make the UI *slower*; use timeouts and fallback.
- Cache results where reasonable (short TTL) to avoid repeated git work.
- Ensure results are consistent with projections when both are available (or be explicit about which is authoritative).

## Complexity / “is it worth it?”

This task should answer:

- What data is most responsible for user-visible staleness?
- How expensive are the git computations in practice (especially on large repos)?
- Does adding a read-query seam create meaningful surface area and maintenance cost?

If the fastpath is complex or doesn’t materially improve UX, we should not pursue it in v0.

## Acceptance Criteria

- A concrete recommendation: implement now vs defer (with reasoning).
- If implemented:
  - the UI shows fresher trunk/commit info when a primary executor is local/connected
  - the system still works correctly without an executor (projection fallback)
  - timeouts/failures do not degrade UX with noisy errors

