---
rn:
  node:
    branch: rn/revise-architecture/T-12-consolidate-backend-mechanisms
  linear:
    issue_id: a2b45498-9059-4e17-ac23-97ac060ab293
    identifier: RED-30
  parent: T-9
---

# T-12 Consolidate overlapping backend mechanisms (reduce surface area)

## Motivation

Redesmyn has grown quickly and several core responsibilities are implemented in multiple places. Most of this code is *doing real work* (not dead), but we can reduce the long-term maintenance burden by consolidating common patterns and establishing a single “way to do it” for:

- running git commands and interpreting failures/timeouts
- computing + persisting git-derived projections/events
- background task lifecycle and shutdown correctness (especially under xdist/tests)
- “plan → execute → emit events” plumbing for merge/restack and similar operations
- integration client structure and typing (Linear in particular)
- DB engine/session creation + SQLite configuration (timeouts/PRAGMAs) and consistent error handling

This task is intentionally a refactor/consolidation effort: it should *not* change product semantics.

## Why this is part of revise-architecture

Revise-architecture is about tightening boundaries between:

- control plane (API/UI + DB) and
- host-local execution/observation (repo executor / daemon).

Consolidation work supports that by:

- reducing the number of paths that “touch git” (and the number of subtly different implementations),
- making it easier to ensure “server does not run git” invariants,
- and making correctness + observability improvements (logging, lock handling, teardown) apply everywhere.

## Scope

### 1) Unify git subprocess execution helpers

Today we have multiple implementations of “run git with timeout, parse output, classify errors”:

- `redesmyn/repo.py` (various `git_*` helpers used by `git_telemetry.py`, `git_mechanics_v0.py`, etc.)
- `redesmyn/repo_observer.py` (its own `_run_git` and custom parsing)

Goal:

- Introduce a single internal abstraction (module or functions) for:
  - invoking git with consistent timeouts
  - capturing stdout/stderr deterministically
  - classifying common failure modes (non-repo, ref missing, in-progress rebase, etc.)
  - optional structured logging hooks
- Migrate `repo_observer.py` to use that abstraction so we do not maintain two parallel “git runners”.

### 2) Consolidate “git-derived projections” generation

We currently write multiple kinds of git-derived DB projections:

- ref tips per instance, merge bases, trunk timeline (`redesmyn/git_telemetry.py`)
- stack-in-sync + worktree health + other repo observations (`redesmyn/repo_observer.py`)

Goal:

- Factor shared logic into a single internal component responsible for:
  - “read git state” (refs, heads, merge base, commit summaries)
  - “write projections/events” with consistent attribution (`workspace_id`, `repo_id`, `host_key`)
  - debouncing + best-effort behavior (don’t crash on transient git/DB issues)
- Keep different “triggers” (periodic observer loop, after merge/restack, etc.) but route them through the same projection writer so we have one canonical implementation.

### 3) Standardize background task lifecycle / shutdown

We have several long-running background tasks (agent monitoring/supervision, WS loops, repo observation, etc.) and we already have patterns emerging for:

- registering tasks
- cancellation and awaited shutdown
- avoiding “task leaks” under tests/xdist

Goal:

- Make a small shared helper for spawning/registering/tearing down background tasks, and use it consistently across API + daemon runtime entrypoints.
- Ensure every background task is cancelled and awaited at shutdown; errors should be logged with context but not crash shutdown.

### 4) Deduplicate “plan → execute → event stream” plumbing

Merge/restack are converging on a shared planning model via `RepoExecutor`, but we still have repeated code around:

- effective base computation
- validation / early-error attribution
- run state persistence (`MergeRun` metadata)
- emitting progress/blocked/failed events and persisting “blocked” details

Goal:

- Introduce a shared internal orchestrator layer for “high-level operation intent”:
  - `plan_*` returns a typed plan + a human-readable summary
  - `execute_*` executes via a `RepoExecutor` and emits a shared event vocabulary
- Keep executor implementations focused on leaf operations (git/worktree mutation primitives), not orchestration.

### 5) Refactor Linear integration layout + typing

`redesmyn/integrations/linear.py` currently mixes:

- auth/token storage concerns
- a GraphQL transport client
- pagination utilities
- mapping to Redesmyn domain objects and sync semantics

Goal:

- Split into a small, legible module structure:
  - `linear_client.py` (GraphQL transport + retries/errors)
  - `linear_models.py` (typed payload models)
  - `linear_sync.py` (sync semantics + mapping)
- Minimize mocks: keep most tests at the integration boundary; only mock network.

### 6) Consolidate DB engine/session creation and SQLite tuning

We have multiple “create engine → init_db → sessionmaker” sequences across the codebase.
Given increasing SQLite lock contention, it’s important that:

- SQLite PRAGMAs and timeouts are set consistently
- lock errors are handled consistently and logged with actionable context

Goal:

- Introduce a single “DB context” helper for async session creation used by all long-running loops.
- Ensure SQLite config (busy timeout, WAL, etc.) is applied consistently.

## Non-goals

- Removing functionality or changing user-visible semantics (this is a consolidation/refactor task).
- Designing new policies (e.g., enforcing git mutability rules) unless required to preserve existing behavior.

## Success criteria

- There is one canonical way to run git subprocess calls (timeouts/logging/error handling consistent).
- Projection generation has a single “writer” implementation shared by telemetry/observer triggers.
- Background tasks are consistently registered/cancelled/awaited; no xdist task leaks.
- Merge/restack orchestration code is simplified via shared helpers, with less duplication across server/daemon paths.
- Linear integration code is easier to read/test, with stronger typing and smaller files.
- DB engine/session construction is consolidated; SQLite lock handling and logging are consistent.
