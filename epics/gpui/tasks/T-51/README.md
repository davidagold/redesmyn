---
epic: gpui
branch:
  suggested: rn/gpui/T-51-graph-layout-engine
rn:
  parent: T-2
---

# T-51 Deterministic layout engine v1 (variable node sizes; expand/collapse relayout) (Domain 6)

## Problem

We need a Rust-native layout engine that is:

- deterministic,
- fast,
- simple to maintain and extend,
- and supports dynamic node sizes (expand/collapse) without the UI becoming janky.

The current web implementation uses ELK (`mrtree`) with fixed node sizes and large spacing gaps; it cannot truly “relayout on expand/collapse” based on measured sizes.

If we do not build a clear layout engine interface early, graph work will:

- hardcode geometry assumptions,
- bake in rendering-specific quirks,
- and be difficult to test.

## Goal

Implement a pure Rust layout engine crate that takes:

- nodes (ids + sizes + hierarchy),
- edges (at minimum parent→child),

and produces:

- deterministic node positions,
- and (optionally) edge routes.

It must support variable node sizes and recomputation when nodes expand/collapse.

## Requirements

### 1) Pure crate with strong typing

Create a layout crate (names illustrative):

- `redesmyn_graph_layout`

It must:

- depend only on core types (`redesmyn_ids` / `redesmyn_domain` as needed),
- have no GPUI dependencies,
- be unit-testable without a GUI.

### 2) Inputs: hierarchy-first

Start with the primary structure we have today:

- a rooted forest defined by `parent_task_id` (tree edges).

Keep an extension seam for future non-tree edges (blockers, merge constraints), but do not over-design v1.

### 3) Variable node sizes

Layout must accept per-node width/height (in logical pixels).

Nodes have discrete size states (collapsed/expanded); a size change triggers recomputation.

### 4) Determinism guarantees

Explicitly guarantee determinism by:

- stable node ordering,
- stable traversal rules,
- and avoidance of hash-map iteration nondeterminism.

Add tests that assert:

- identical input → identical positions,
- positions stable across runs (golden snapshots are acceptable for pure layout).

### 5) Algorithm choice (v1)

Implement a simple, maintainable algorithm appropriate for a tree/forest:

- a tidy-tree style layout (e.g., Reingold–Tilford style), or
- a deterministic layered layout with row packing.

We optimize for:

- correctness + clarity first,
- then performance (no superlinear behavior on common graphs).

### 6) Output: positions + bounding box

Return:

- `HashMap<NodeId, Point>` (or equivalent typed map),
- and the graph bounds (for fit-to-view).

Optional (can be deferred to T-53):

- edge routes (orthogonal polylines).

### 7) Layout invalidation strategy

Define a minimal invalidation story:

- layout is recomputed when:
  - topology changes,
  - node sizes change,
  - or focus mode changes (subset layout; see T-54).

The layout engine itself remains pure; the UI decides when to call it.

## Acceptance criteria

- A pure layout crate exists with deterministic unit tests.
- Layout supports variable node sizes and produces stable, collision-free positions for a forest.
- Layout is fast enough for the v0 perf target (qualitative):
  - ~100 nodes / ~100 edges layout within a frame budget on a modern laptop in release mode.

## Dependencies / sequencing

- Depends on ULID/newtypes (T-2) for typed ids (or equivalent typed identity strategy in Rust).
- Used by the GPUI graph renderer (T-50) and viewport behaviors (T-54).

## Reference implementation (today; for behavior orientation only)

- ELK-based layout (web today):
  - `dashboard/src/components/graph/elkLayout.ts` (deterministic ordering + `mrtree`).
- Fallback deterministic layout (web today):
  - `dashboard/src/components/graph/flowLayout.ts` (DFS traversal layout).
- Config constants (web today):
  - `dashboard/src/components/graph/graphConfig.ts` (node sizes + spacing).

