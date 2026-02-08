---
epic: gpui
branch:
  suggested: rn/gpui/T-50-gpui-graph-scene
rn:
  parent: T-45
  after:
    - T-44
    - T-21
---

# T-50 Graph scene + renderer scaffolding (GPUI canvas, camera, hit-testing) (Domain 6)

## Problem

The web dashboard graph relies on:

- ReactFlow for rendering/interactions, and
- ELK for layout.

In the GPUI port we need:

- a Rust-native renderer (fast, deterministic, testable),
- a camera model (pan/zoom) and coordinate transforms,
- hit-testing for selection/hover,
- and a clean seam so layout/node/edge work can be parallelized.

If we don’t establish a stable “graph scene” architecture early, later work will entangle:

- layout, rendering, and business state,
- making it hard to iterate and test.

## Goal

Create the GPUI graph scaffolding:

- a `GraphScene` (model) with stable ids and selection state,
- a renderer that draws nodes and edges,
- camera/viewport transforms (pan/zoom),
- and hit-testing primitives.

This ticket focuses on architecture and scaffolding; the layout algorithm and rich node UI land in separate tickets.

## Requirements

Architecture note: follow the GPUI state architecture policy in `epics/gpui/README.md` §3.7 (Presentation Model; explicit + testable). In particular: keep domain/business state in the control plane; treat `GraphScene` and camera/selection state as UI presentation models; keep IO out of `render()`.

### 1) Crate/module structure

Create a dedicated Rust module/crate for the graph UI (names illustrative):

- `redesmyn_ui_graph` (views + rendering)
- optionally `redesmyn_graph` (pure view-model types) if it helps parallelism

Rule: keep layout logic in its own crate (T-51) so it can be tested independently.

### 2) Graph scene model (typed)

Define typed structures for:

- nodes (task nodes, and optional “trunk” node placeholder),
- edges (at minimum: parent→child edges),
- selection state:
  - selected node id (optional),
  - selected edge id (optional),
  - multi-selection set (optional; wired in later).

The scene must be able to update incrementally when the EpicGraph read model changes.

### 3) Camera model

Implement a camera/viewport model:

- world↔screen coordinate transforms,
- pan and zoom,
- min/max zoom policy,
- and a “fit-to-view” operation stub (completed in T-54).

### 4) Hit testing

Implement hit-testing primitives that work with the camera:

- hit node bounding boxes,
- hit edge segments (coarse is fine initially; refined in T-53),
- and return stable ids for selection.

### 5) Rendering strategy (GPUI)

Establish the rendering approach:

- draw edges behind nodes,
- avoid per-frame allocations where possible,
- define an explicit invalidation strategy (when to re-render).

### 6) No silent actions (hook)

Even in scaffolding, establish the UI pattern:

- selection changes are immediate,
- action triggers (later) must produce visible in-flight state (wired via T-44 conventions).

## Acceptance criteria

- The desktop app can render a basic graph scene in the right pane (placeholder nodes/edges are fine).
- Pan/zoom works and is smooth on a small graph.
- Clicking a node updates selection state (reflected in a debug label or semantic snapshot).
- The architecture cleanly separates:
  - graph scene state,
  - layout (T-51),
  - node views (T-52),
  - and edge rendering (T-53).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on the desktop split layout (T-45) and UI foundations (T-44).
- Depends on the EpicGraph read model contract (T-21) for the data shape (even if stubbed initially).

## Reference implementation (today; for behavior orientation only)

- Graph host and behaviors (web today):
  - `dashboard/src/components/graph/GraphView.tsx` (ReactFlow + layout orchestration).
  - `dashboard/src/components/graph/graphConfig.ts` (camera + perf constants; fit/pan policies).
- Graph data shape (web today):
  - `dashboard/src/lib/graph-utils.ts` (EpicGraph-derived types).
