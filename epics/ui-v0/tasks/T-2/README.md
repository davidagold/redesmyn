# T-2 Modernize dashboard query management (TanStack Query)

## Metadata

```yaml
id: T-2
epic: ui-v0
stacked_on: T-1
branch:
  suggested: rn/ui-v0/T-2-query-management
```

## Problem

The dashboard currently mixes several state sources:

- ad-hoc `fetch` calls
- WebSocket events (some state changes are “pushed”)
- local component state used as “optimistic” UI in a few places

This leads to predictable v0 issues:

- **Incoherent transitions** when a mutation triggers a refresh and the query invalidation races with UI state.
- **Stale UI** when a backend change occurs but the client doesn’t refresh the right data (or refreshes too aggressively).
- **Component bloat** because loading/error/refresh patterns are re-implemented repeatedly.

We want to converge on a modern, explicit model:

- Server state is cached and invalidated consistently.
- Mutations are optimistic where safe, and are reconciled cleanly.
- Event streams are used to keep the cache fresh without “poll everything”.

## Goal

Adopt **TanStack Query** (React Query) as the single, consistent server-state layer for the dashboard.

Make the UI:

- responsive (fast feedback on clicks)
- correct (reflects latest backend state)
- smooth (no jarring jumps when queries invalidate and re-render)

## Requirements

### 1) Introduce TanStack Query as the default pattern

- Add a `QueryClient` + provider at the app root.
- Move “server state” fetches to query hooks:
  - `useEpicGraph(epicSlug|id)`
  - `useConfig()`
  - `useDaemons()` / `useRepoExecutorStatus()` (as applicable)
  - `useTaskDetails(taskId)` (if separate from graph)
- Queries should be keyed in a way that makes invalidation easy and intentional.

### 2) Mutations must invalidate (or update) the correct queries

For user actions that mutate server state (examples):

- start/restart/stop agent
- merge/restack/resume merge run
- set “ready to merge”
- linear sync operations
- config updates

Implement a mutation hook per action that:

- updates the cache optimistically when safe (e.g. toggles)
- otherwise triggers invalidation of specific queries
- provides a single place to map “action → affected queries”

### 3) Event-driven query invalidation / cache updates

We already have a WS event stream; use it to keep the cache fresh:

- For “high-signal” events (merge run updates, agent session status changes):
  - prefer `queryClient.setQueryData` with a precise patch when feasible
  - otherwise `invalidateQueries` for the minimal affected keys
- Add debouncing/batching when multiple events arrive quickly to avoid jitter.

Important: do not create a second state system; the event layer should flow into query cache updates.

### 4) Smooth transitions for invalidation-driven updates

When a query refetch changes rendered UI:

- Prefer subtle transitions:
  - small fade/opacity changes on updated panels
  - gentle highlight/glow for “changed” items when appropriate
- Avoid layout “snap” where possible:
  - do not unmount/remount large sections unnecessarily
  - keep stable keys; update data in-place

This should remain elegant (no noisy loading spinners everywhere).

### 5) xdist/AI-driven development friendliness

Design goals for maintainability:

- Hooks should be strongly typed and colocated with API typings (`dashboard/src/api`).
- Provide an obvious pattern for “add a query” and “wire an event”.
- Avoid over-abstraction: the goal is readability + correctness, not a framework inside the app.

## Notes / design choices

- Prefer “invalidate then refetch” unless a patch update is clearly simpler and less bug-prone.
- If we keep some local UI state for immediate feedback, it must reconcile with query state without flicker.
- For “optimistic” flows that can fail, ensure the rollback is predictable and communicates failure succinctly.

## Acceptance Criteria

- The primary user flows (agent lifecycle, merge/restack/resume, config changes) no longer rely on ad-hoc refresh calls.
- The UI remains correct after a sequence of quick actions (no “looks like it happened but didn’t”).
- WS events keep relevant UI state fresh without full-page refresh or heavy polling.
- Query invalidation/refetch does not cause jarring visual jumps; transitions are subtle and intentional.
