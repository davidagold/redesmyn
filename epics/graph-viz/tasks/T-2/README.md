---
rn:
  node:
    branch: rn/graph-viz/task-2-interactions-selection
  parent: T-1
---

# T-2 Graph interactions + selection model

## Brief (local)

- Make the viewport feel “native” and empathetic: predictable focus, minimal chrome, and no redundant information competing with the graph.
- Define a consistent interaction model:
  - Node selection, edge selection, hover states.
  - Keyboard shortcuts (Esc clears selection; future: search/command palette hooks).
- Ensure Details panel integration stays clean (selection drives focus; no redundant UI).

## Acceptance Criteria

- Node selection behavior is consistent across the graph:
  - Single selection (v0): selecting a new object clears the previous selection.
  - Esc clears selection and closes the Details panel.
  - Clicking the background clears selection.
- Edge selection is supported (even if edges are still rendered simply):
  - Hover conveys interactivity without adding visual noise.
  - Click selects the edge and opens Details (or updates Details content) without changing the viewport unexpectedly.
- Selection state is URL-agnostic in v0, but designed so we can add deep-linking later without rewrites.

## Notes / Contracts

- Establish an explicit selection model (`selectedNodeId | selectedEdgeId`) so downstream work (commit strings) can target edges without hacks.
- Prefer subtle affordances (opacity, underline, slight glow) over heavy borders; avoid “busy” linework.
