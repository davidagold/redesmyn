---
rn:
  slug: graph-viz
  name: Graph Visualization
  root_branch: main
  linear:
    project_id: null
---

# Graph Visualization Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Graph Visualization** epic: intent, v0 spec, invariants, and key decisions. Keep it current.

## 1) Vision

Upgrade the dashboard graph from a static nested list to a Dagster-like interactive graph viewport:

- Pan + zoom + fit-to-view.
- Highly stylable nodes/edges (shadcn/Tailwind + custom rendering).
- Smooth transitions for layout changes and live updates.
- “Commit strings” are **first-class** (edges represent a selectable, inspectable sequence of commits).

### Layout vision (target)

- The epic root branch (usually `main`) is rendered as a horizontal **trunk** across the top.
- Each node gets a horizontal “lane” below; lanes flow **left → right**.
- Parent → child edges branch **down** from the parent lane/trunk and curve into the child lane.
- Horizontal distance is **commit-count scaled** (with summarization/clamping for long ranges).
- Root nodes (stacked on the trunk) attach at the **merge-base commit** on the trunk (“commit-accurate”).

## 2) Why a dedicated epic?

The graph is the primary UI object in Redesmyn. This epic exists to keep the migration plan and sequencing clear, so we can dogfood it safely and iterate without derailing unrelated v0 work.

## 3) Approach (v0)

### 3.1 Library strategy

- Use **XYFlow / React Flow** as the interactive viewport substrate (pan/zoom, selection model, DOM-based node rendering).
- Use **ELK** for deterministic, controllable layout (tree/stack layouts).
- Build a custom **edge renderer** for “commit strings” and expand its fidelity over time (level-of-detail by zoom).

### 3.2 Migration plan (phased)

1. **Viewport foundation**
   - Replace the current graph list with an XYFlow-based view (nodes + parent edges).
   - Preserve current node card styling via custom node renderers.
2. **Interaction model**
   - Node/edge selection, keyboard shortcuts (Esc clears), background click to clear.
   - Maintain deterministic state and keep styling consistent with the rest of the dashboard.
3. **Deterministic layout**
   - Integrate ELK and compute stable positions from the branch topology.
   - Add fit-to-view and re-layout on resize/data changes.
4. **Commit strings (data)**
   - Extend the daemon API to expose commit-range metadata per node and edge.
5. **Commit strings (rendering)**
   - Implement a first-class edge renderer (ticks/segments, selection/hover, inspectability).
6. **Polish**
   - Layout/transition animations, LOD/perf budget, and “Dagster feel” refinement.

## 4) Scope boundaries (v0)

- Start with the **branch graph** (nodes + parent edges).
- Commit-string edges start minimal (e.g., count + highlights), then grow toward full per-commit inspectability.
- Avoid premature optimization; prefer clear architecture so we can later swap render strategies if needed.
