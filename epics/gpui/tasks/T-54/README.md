---
epic: gpui
branch:
  suggested: rn/gpui/T-54-graph-viewport-behaviors
rn:
  node:
    branch: rn/gpui/T-54-graph-viewport-behaviors
  parent: T-50
  after:
    - T-51
---

# T-54 Viewport behaviors (fit-to-view, pan-to-selection) (Domain 6)

## Problem

Graph usability depends heavily on camera behaviors:

- first-load fit-to-view,
- pan-to-selection (without hiding behind panels),
- and smooth animated transitions that feel responsive.

Without explicit camera policy, the graph will feel brittle and hard to use.

## Goal

Implement the key viewport behaviors for the GPUI graph:

- fit-to-view on initial layout settle,
- pan-to-selection with reserved UI space,
- and smooth animation of layout transitions (position interpolation).

## Requirements

### 1) Fit-to-view

Implement:

- compute graph bounds from layout output,
- choose zoom within min/max bounds,
- pan to center,
- and suppress fit when there is an explicit selection (so we don’t fight the user).

### 2) Pan-to-selection

When a node is selected:

- pan it into view,
- account for expanded node size (expanded task cards are large in this port),
- keep animation short and calm (no spinning).

### 3) Layout transition animation

When positions change (layout recompute):

- animate node positions between old and new positions with easing.
- Avoid per-frame allocations where possible.

### 4) Determinism and testability

- Camera behaviors expose enough state to assert via semantic UI snapshot (T-58).

## Acceptance criteria

- On first load with no selection, the graph fits into view.
- Selecting a node pans it into view reliably (including when the selected task is expanded).
- Layout recomputes animate smoothly without visible jank on small graphs.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on graph scene scaffolding (T-50) and layout engine bounds (T-51).
- Interacts with expanded task card behavior (T-52/T-55) for pan-to-selection ergonomics.

## Reference implementation (today; for behavior orientation only)

- Fit/pan behaviors (web today):
  - `dashboard/src/components/graph/GraphView.tsx` (fitView, pan-to-selection).
