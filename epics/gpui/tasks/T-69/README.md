---
epic: gpui
branch:
  suggested: rn/gpui/T-69-graph-node-virtualization
rn:
  node:
    branch: rn/gpui/T-69-graph-node-virtualization
  parent: T-50
  after:
    - T-51
    - T-52
---

# T-69 Graph node virtualization + LOD (viewport culling; node render modes) (Domain 6)

## Problem

The graph view will eventually render hundreds (or thousands) of task nodes. Mounting a rich
TaskCard view for every node is too expensive and will:

- make pan/zoom janky,
- increase per-frame allocations and layout work,
- and cause selection-driven resize/relayout (T-51/T-55) to thrash.

We already plan zoom-driven LOD for edges in T-53, but we need the equivalent for **nodes**:

- viewport-based culling/virtualization (only keep views for visible nodes), and
- zoom-based node LOD bands (use a cheaper representation when zoomed out).

## Goal

Implement node virtualization + node LOD for the GPUI graph so that:

- only nodes in/near the viewport are mounted as rich views,
- zooming switches between discrete node render modes without churn,
- and selection/expansion remains stable and interactive.

## Requirements

### 1) Viewport culling

- Compute visible world-space bounds from the camera state (T-50).
- Define configurable overscan (in screen px or world units) to avoid pop-in/out at the edges.
- Determine the visible node set via intersection of viewport bounds and node bounds produced by
  layout output (T-51).

### 2) Node virtualization policy

- Only instantiate `TaskNodeView`/TaskCard views for:
  - nodes in the visible set, and
  - pinned nodes (selected, hovered, focused, actively resizing).
- Unmounted nodes still exist in the scene model; selection/hit-testing must continue to work
  using cached bounds (T-50).
- Do not drop focused input mid-interaction: pinned nodes must preserve internal UI state.

### 3) Node LOD bands (zoom-driven)

Define discrete node LOD bands driven by camera zoom with hysteresis (in the spirit of T-53):

- **Low zoom**: minimal node representation (no rich subviews; optionally canvas-drawn placeholder).
- **Mid zoom**: compact/collapsed TaskCard with minimal content.
- **High zoom**: normal TaskCard; selected node may expand (T-55).

Band changes must be discrete: no continuous “add/remove details” behavior on tiny zoom deltas.

### 4) Measurement cache integration

- Maintain a cache of measured node sizes (at minimum: collapsed vs expanded) keyed by node id.
- When a node view is virtualized away, keep its last known size so layout stays stable.
- Ensure the “measure → layout → render” loop cannot oscillate/infinite-loop under virtualization
  (T-52 calls this out; this ticket enforces it at scale).

### 5) Allocation-free steady state

- Pan/zoom within a stable visible set must not allocate per-frame in the virtualization loop.
- Only allocate when:
  - nodes enter/leave the overscanned viewport, or
  - an LOD band boundary is crossed.

### 6) Testability hooks (light)

Expose enough semantic UI snapshot state (T-58) to assert:

- current node LOD band,
- total node count vs mounted/visible node count,
- selected node remains mounted.

## Acceptance criteria

- With a large synthetic graph (e.g., 500+ nodes), pan/zoom remains smooth because only nodes
  in/near the viewport are mounted as rich views.
- Node LOD banding works: zooming out transitions to a cheaper node representation; zooming in
  restores full cards; no flicker at thresholds.
- Selecting a node keeps it mounted and interactive; expanding it triggers relayout (T-51) without
  mass re-mount churn.
- Virtualization/culling loop is allocation-free in steady state (no per-frame allocations while
  panning within an unchanged visible set).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging`
  (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on graph scene + camera model (T-50), layout output bounds (T-51), and node view
  measurement contract (T-52).
- Interacts with expanded task card behavior (T-55) and edge LOD (T-53) but should not require
  either to land first.

## Reference implementation (today; for orientation only)

- Web graph perf constants and zoom policies:
  - `dashboard/src/components/graph/graphConfig.ts`
