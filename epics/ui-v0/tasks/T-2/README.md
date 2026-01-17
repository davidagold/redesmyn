---
id: T-2
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-2-query-management
---

# T-2 Modernize dashboard query management (TanStack Query)

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

- updates query cache from the mutation response when possible (authoritative, avoids refetch)
- otherwise triggers invalidation of specific queries
- uses true optimistic updates only when clearly safe + rollbackable (e.g. simple toggles like “ready to merge”)
- provides a single place to map “action → affected queries”

### 3) Event-driven query invalidation / cache updates

We already have a WS event stream; use it to keep the cache fresh:

- For “high-signal” events (merge run updates, agent session status changes):
  - prefer `queryClient.setQueryData` with a precise patch when feasible
  - otherwise `invalidateQueries` for the minimal affected keys
- Add debouncing/batching when multiple events arrive quickly to avoid jitter.

Important: do not create a second state system; the event layer should flow into query cache updates.

Note: for complex async operations (agent lifecycle, merge/restack/resume, bulk actions, linear sync), prefer authoritative updates via WS events and/or mutation responses rather than speculative optimistic state.

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

## Preparation (recommended before/while implementing)

These are small refactors that make the TanStack Query migration much more mechanical and reduce the risk of “two competing state systems” during the transition.

### A) Land T-3 extraction seams first (or use the same structure)

T-2 will be easier if the code already has clean seams for:

- a single “run toolbar model” (counts/buckets/targets)
- a persisted bulk-action state hook
- a dedicated orchestration config panel component
- centralized parsing for “running agents” conflicts / confirmations

If these don’t exist yet on the branch, prefer creating them as part of T-3 or early in T-2 so the Query migration doesn’t happen inside monolithic components.

### B) Make `api.ts` query-friendly (AbortSignal + consistent errors)

React Query will pass an `AbortSignal` to query functions; update `fetch*` helpers to accept an optional `signal` and forward it to `fetch(...)` so in-flight requests cancel cleanly.

Also, prefer a single consistent error shape (e.g. `ApiHttpError`) across API calls so query/mutation hooks don’t need per-endpoint error parsing.

### C) Establish query keys up front

Create a small `queryKeys` module (or equivalent) early so every hook uses the same keys and invalidation is intentional and discoverable:

- `epics`
- `epicGraph(epicId)` (or epic slug)
- `hosts`, `daemons`, `repoExecutorStatus` (as applicable)
- `config/orchestrationDefaults`
- any “task details” query if we split it from the graph later

### D) Centralize “WS event → cache updates” in one hook

Today, the event stream drives manual refresh calls. For T-2, move this wiring into a single hook (e.g. `useEpicCacheSync`) so:

- it can call `queryClient.setQueryData(...)` for high-signal events (agent session + merge run)
- and `invalidateQueries(...)` as a fallback for less precise events
- with debouncing/batching handled in one place

This avoids sprinkling query invalidation logic throughout components.

### E) Identify and reduce local state shadowing server state

Before adding optimistic mutations, audit where local state is used as a “server proxy” (examples: merge-ready toggles, agent status/callouts, selection-derived counts).

Prefer:

- derived values from query state (`useMemo`/pure functions), and
- optimistic updates implemented inside mutation hooks (with rollback) rather than in component-local state.

## Notes / design choices

- Prefer “invalidate then refetch” unless a patch update is clearly simpler and less bug-prone.
- Prefer authoritative cache updates (mutation response / WS event patches) over speculative optimistic updates.
- If we keep local UI state for immediate feedback, it must reconcile with query state without flicker.
- For “optimistic” flows that can fail, ensure the rollback is predictable and communicates failure succinctly.

## Acceptance Criteria

- The primary user flows (agent lifecycle, merge/restack/resume, config changes) no longer rely on ad-hoc refresh calls.
- The UI remains correct after a sequence of quick actions (no “looks like it happened but didn’t”).
- WS events keep relevant UI state fresh without full-page refresh or heavy polling.
- Query invalidation/refetch does not cause jarring visual jumps; transitions are subtle and intentional.
