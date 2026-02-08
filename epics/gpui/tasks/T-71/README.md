---
epic: gpui
branch:
  suggested: rn/gpui/T-71-wire-epic-graph-to-graph-view
rn:
  parent: T-46
  after:
    - T-20
    - T-21
    - T-50
---

# T-71 Wire epic selection to graph rendering (GetEpicGraph → GraphScene) (Domain 6)

## Problem

The desktop chrome (T-46) can select an epic, but the right pane still renders a demo graph
(`GraphView::new_demo`). This blocks a “real” wired experience:

- selecting an epic does not load or render its task graph,
- graph work cannot be validated against real data,
- and later AI-testability work (T-58) cannot “open an epic” deterministically.

## Goal

When the user selects an epic:

- the desktop app queries the control plane (`GetEpicGraph`) for that epic’s graph,
- converts the response into a `redesmyn_ui_graph::GraphScene`,
- and renders it in `GraphView`.

Switching epics must update the graph. The UI must show explicit in-flight state and error states
(no silent actions).

## Requirements

### 1) Desktop plumbing (no IO in `render()`)

- Selecting an epic triggers an async fetch of `GetEpicGraph`.
- Cancel/ignore stale in-flight requests if the selected epic changes.
- Show explicit progress state while loading (text-based; no spinner wheels).
- On failure, show an actionable error callout and allow retry.

### 2) ControlPlaneClient support

Extend the desktop `ControlPlaneClient` with:

- `get_epic_graph(epic_slug) -> EpicGraph` (or equivalent typed surface),
- error mapping consistent with existing `list_epics`/`status` patterns.

### 3) Deterministic mapping: `EpicGraph` → `GraphScene`

Define a stable mapping from `redesmyn_protocol::client::EpicGraph` to `redesmyn_ui_graph::GraphScene`:

- nodes keyed by stable ids (`TaskId` → `GraphNodeId::Task(TaskId)`),
- edges representing parent→child relationships (and any available metadata),
- task fields mapped where available (title, slug, branch name, state, merge readiness),
- agent/session status can be `Unknown` until real data is available.

Selection behavior:

- Preserve selection when possible (if the selected node still exists after reload).
- Otherwise clear selection deterministically.

### 4) Empty/degenerate graphs

- If an epic has zero tasks, render an explicit empty state (still within the graph host).
- Avoid pan/zoom bugs when bounds are empty or a single node exists.

### 5) Testability hooks (light)

Expose enough state for semantic snapshots/waits (T-15/T-58) to assert:

- selected epic slug,
- graph loaded state (loaded vs empty vs error),
- node/edge counts.

## Acceptance criteria

- Selecting an epic causes the graph pane to render the epic’s real tasks/edges (not demo data).
- Switching epics updates the graph and cancels/ignores stale fetches deterministically.
- Loading/error/empty states are visible and actionable.
- The fetch/mapping path is allocation-free in steady state (no per-frame allocations; rebuild only on data change).

## Dependencies / sequencing

- Depends on:
  - desktop chrome epic selection (T-46),
  - client API server + GetEpicGraph (T-20),
  - EpicGraph read model contract (T-21),
  - graph scene + view scaffolding (T-50).
- Informs/unblocks:
  - graph testability (T-58),
  - later graph UX work that needs real data (T-52..T-57).

