---
rn:
  node:
    branch: rn/graph-viz/task-7-focus-mode-diagonal-bias
  parent: T-2
  after:
  - T-3
---

# T-7 (Bonus) Focus mode: diagonal branch bias

## Brief (local)

When the user focuses a specific branch/stack (centers + zooms it in the viewport), consider a “diagonal bias” layout:

- Bias the focused path top-left → bottom-right so the primary information display remains horizontal (legible),
  while still leaving horizontal space adjacent to nodes for contextual panels/annotations without overlapping the path.

## Acceptance Criteria

- Optional/behind a toggle in v0: a “focus mode” view exists for a selected node/stack.
- The focused path remains readable (no label rotation), and adjacent space for annotations is increased versus the default layout.
- The implementation does not compromise the default layout or introduce hard-to-maintain special cases (keep it isolated as an alternate layout strategy).
