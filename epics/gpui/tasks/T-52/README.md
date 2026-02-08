---
epic: gpui
branch:
  suggested: rn/gpui/T-52-task-node-view
rn:
  node:
    branch: rn/gpui/T-52-task-node-view
  parent: T-50
  after:
    - T-44
---

# T-52 Task node view (compact/expanded; measurement; selection affordances) (Domain 6)

## Problem

The task node is the primary interactive unit of the graph.

In the desktop port we want:

- a readable, information-dense task card,
- fast expand/collapse,
- and measurement-driven layout (so expansion triggers relayout without collisions).

If node UI and layout are coupled, we will:

- make layout fragile,
- re-render too much,
- and struggle to keep performance smooth.

## Goal

Implement the GPUI task node view with:

- compact and expanded presentation states,
- clean selection/hover affordances,
- and a measurement contract that feeds node size into the layout engine (T-51).

This ticket focuses on node rendering and measurement; rich task actions can be stubbed initially.

## Requirements

### 1) Node content (v0)

Render at minimum:

- task title (and id),
- task state (blocked/in_progress/done),
- merge readiness indicator,
- agent/session status indicator (running/blocked/stopped/error),
- and small callouts where actionable (but avoid “busy border soup”).

### 2) Expand/collapse behavior

Support:

- collapsed (default) vs expanded (large “task inspector” surface),
- expansion triggers a node size change and requests a relayout (T-51),
- smooth animation for size/position transitions (layout animation can live in T-54).

Expanded content direction:

- the expanded card is the single home for “expanded task” UI (no separate sidebar drawer),
- and it is designed to host:
  - a SessionView slot (chat/history/composer) and
  - a structured details surface (T-55/T-64).

### 3) Selection affordances

- Click selects the node.
- Selected state is clearly visible (without thick borders everywhere).
- Keyboard focus behavior must be sane (no focus traps).

### 4) Measurement contract

Define how node size is measured and reported:

- node view reports its measured size to the graph scene,
- graph scene triggers layout recompute when size changes.

This must be incremental: changing one node should not force full re-measure of all nodes every frame.

### 5) “No silent actions” integration

Even if we stub actions, define the pattern:

- action buttons show immediate in-flight feedback,
- and disabling prevents duplicate clicks.

## Acceptance criteria

- Task nodes render in the GPUI graph with compact styling.
- Selecting a node updates selection state and highlights it.
- Expanding a node changes its size and triggers a relayout without overlapping nodes.
- Node measurement and layout integration is stable and does not cause infinite relayout loops.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on graph scene scaffolding (T-50) and UI foundations (T-44).
- Uses layout engine (T-51) for positions once integrated.

## Reference implementation (today; for behavior orientation only)

- Task card UI (web today):
  - `dashboard/src/components/graph/TaskCard.tsx`
  - `dashboard/src/components/graph/TaskCardCallouts.tsx`
- Graph node wrappers (web today):
  - `dashboard/src/components/graph/FlowBranchNode.tsx`
  - `dashboard/src/components/graph/TrunkNode.tsx`
