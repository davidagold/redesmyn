---
id: T-1
node:
  branch: rn/graph-viz/task-1-xyflow-foundation
---

# T-1 Graph viewport foundation (XYFlow)

## Brief (local)

- Establish the interactive graph “canvas” so users can efficiently explore the **map of work**
  (tasks/nodes/agents) via pan/zoom and selection (Dagster-like ergonomics).
- Introduce an XYFlow/React Flow based graph viewport to replace the current nested list rendering.
- Render branch graph nodes/edges with full styling control (custom node components).
- Support pan/zoom/fit-to-view and selection events (node click + background click to clear).

## Acceptance Criteria

- Selecting the Graph Visualization epic in the dashboard renders an XYFlow-based graph viewport.
- Pan + zoom work smoothly (trackpad/mouse): drag to pan, wheel/pinch to zoom, and a “fit view” action exists (v0 can be automatic on load).
- Nodes render using our existing visual language (shadcn/Tailwind tokens; no hard-coded colors).
- Selection is clean and non-redundant:
  - Clicking a node selects it and opens the Details panel.
  - Clicking empty space (or pressing Esc) clears selection.
- The code is structured to make follow-on work straightforward:
  - A single mapping layer converts `EpicGraphResponse` → XYFlow nodes/edges.
  - Custom node + edge types exist even if the initial edge renderer is simple.

## Notes / Contracts

- Inputs should come from the existing `/v1/epics/{epic}/graph` endpoint; avoid adding API surface area in this task.
- Initial layout can be simple/temporary, but must not block deterministic layout work (T-3). Prefer a layout function with explicit constants (spacing/direction).
- Treat edges as first-class interactive objects from day one (IDs, click/hover plumbing), even before “commit strings” are fully implemented.
