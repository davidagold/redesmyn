---
epic: gpui
branch:
  suggested: rn/gpui/T-21-epic-graph-query-model
rn:
  parent: T-17
---

# T-21 Epic graph query model + projection (Domain 2)

## Problem

The graph is the primary UI object and also the primary “control surface” for many operations.

We need a fast, well-typed query that returns the current “epic graph” view:

- tasks and topology,
- relevant execution/progress summaries (agents, merge runs, command state),
- and enough metadata for the UI to render callouts and in-flight state without guessing.

If we do this ad-hoc, we risk:

- slow queries,
- UI duplication of business logic,
- and fragile behavior when we evolve domain models.

## Goal

Define and implement the control plane’s “EpicGraph” query model and computation strategy.

This is a Domain 2 deliverable because:

- it defines the central read model for both UI and CLI,
- and it informs schema/index choices (T-17) and subscription payloads (T-18/T-20).

## Requirements

### 1) Typed query response

Define a typed response that includes (at minimum):

- `epic` metadata
- list of `tasks` with:
  - ids, titles, state, branch backing
  - parent linkage
  - merge-ready status
- `command_summaries` relevant to the epic (in-flight + recent)
- `daemon/executor` status relevant to the repo scope (presence + freshness)
- `session summaries` (enough to show “agent running/blocked” in the graph)

Rules:

- Keep it compact; do not embed large content.
- Include stable identifiers and “human refs” where helpful (e.g., local task ref).
- Prefer DB-level constraints that prevent cross-epic parent pointers (T-17), so graph traversal can
  treat `parent_task_id` as a safe in-epic edge.

### 2) Computation strategy

Choose a computation strategy that balances simplicity and performance:

- Prefer a small number of indexed queries over one huge join.
- Avoid expensive N+1 patterns.

It’s acceptable to start with a “good enough” approach as long as it’s structured so we can optimize later.

### 3) Caching / invalidation

Define how the control plane keeps this view fresh:

- event-driven invalidation or cached projections where appropriate,
- and a clear story for how clients can subscribe to updates instead of polling.

### 4) AI testability

The EpicGraph query must make it easy for tests to assert outcomes:

- expose command states and key fields that reflect user-visible progress,
- avoid requiring tests to parse freeform strings.

## Acceptance criteria

- The control plane can answer `GetEpicGraph` over the client API (T-20) using this model.
- Query performance is reasonable for interactive use on moderate graphs (document any known limits).
- The response is stable and typed, making it suitable for AI-driven tests.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on schema (T-17) and event/subscription patterns (T-18).
- Consumed by the client API server (T-20) and later by the GPUI UI.

## Reference implementation (today; epic graph orientation only)

- Control plane graph query (Python today):
  - `redesmyn/api.py` (`GET /v1/epics/{epic}/graph` implemented by `epic_graph()`).
  - `redesmyn/schemas/core.py` (`EpicGraphResponse`, `TaskResponse`, `AgentSessionResponse`, `MergeRunSummaryResponse`, `TrunkTimelineResponse`, etc.).
  - `redesmyn/repo_executor.py` (repo executor status surface included in graph response).
- Dashboard usage (TS today):
  - `dashboard/src/hooks/useGraph.ts`
  - `dashboard/src/routes/EpicView.tsx`
  - `dashboard/src/components/graph/GraphView.tsx`
- Tests (Python today):
  - `tests/test_epic_graph.py` (task topology, agent session overlay, merge run overlay, trunk timeline presence).
  - `tests/test_repo_executor_lease_local_fallback.py` (local-mode lease reacquisition behavior surfaced via epic graph).
  - `tests/test_daemon_ws_runtime_integration.py` (daemon attach affects epic graph repo executor status).
