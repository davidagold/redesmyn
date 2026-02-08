---
epic: gpui
branch:
  suggested: rn/gpui/T-53-graph-edges
rn:
  node:
    branch: rn/gpui/T-53-graph-edges
  parent: T-50
  after:
    - T-51
---

# T-53 Edge routing + rendering (orthogonal edges; hover/selection; LOD labels) (Domain 6)

## Problem

Edges communicate the orchestration topology.

We need edge rendering that is:

- legible,
- performant at ~100 edges (and beyond),
- and supports hover/selection hit testing.

The web implementation uses ReactFlow “smooth step” edges plus a commit-count LOD system to keep perf under control.

## Goal

Implement edge routing and rendering in GPUI:

- orthogonal/smooth-step polyline routing,
- hover/selection states,
- and a level-of-detail strategy for labels/ticks based on zoom.

## Requirements

### 1) Routing strategy

Start with a deterministic routing approach:

- route parent→child edges with a predictable orthogonal path,
- avoid pathological crossings where possible, but keep v1 simple.

If the layout engine (T-51) does not produce routes, edge routing can be derived from node positions/sizes in this ticket.

### 2) Rendering

- Draw edges beneath nodes.
- Use subtle styling (avoid overpowering lines).
- Selected/hover states must be visible but calm.

### 3) Hit testing

Implement coarse hit testing:

- treat edge as a polyline with an interaction width,
- return stable edge ids (for selection in T-55).

### 4) Zoom-based LOD

Implement discrete LOD bands (not continuous) so panning/zooming does not re-render everything on tiny zoom deltas:

- low zoom: edges only
- mid zoom: show lightweight labels (e.g. commit count)
- high zoom: show tick/dash affordances for commit ranges

The exact affordances can start minimal; the important part is the mechanism.

## Acceptance criteria

- Edges render correctly between nodes and remain stable across layout recomputes.
- Hovering an edge highlights it and selection works (coarse hit testing is acceptable).
- LOD bands work: labels/ticks appear/disappear based on zoom without janky rerenders.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on graph scene scaffolding (T-50).
- Should integrate with layout engine outputs (T-51) once available.

## Reference implementation (today; for behavior orientation only)

- Edge components and LOD (web today):
  - `dashboard/src/components/graph/CommitStringEdge.tsx`
  - `dashboard/src/components/graph/RoundedSmoothStepEdge.tsx`
  - `dashboard/src/components/graph/graphConfig.ts` (`edgeLodBand` + perf notes)

