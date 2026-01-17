---
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-8-graph-selection-simplify
rn: {}
---

# T-8 Remove graph on-select viewport behavior + diagonalization repositioning

## Problem

The graph currently applies automatic viewport changes during selection/focus workflows (zoom/pan), and also includes
“diagonalization” node repositioning logic. In practice:

- the behavior is not reliably producing the intended UX,
- it adds complexity and “motion” that can feel surprising, and
- it makes it harder to iterate on the core graph interactions.

Reference request: remove the behavior for now and simplify as much as possible.

## Goal

Simplify graph interactions by removing:

- all on-select zoom/pan/viewport animation behavior, and
- diagonalization (node repositioning) behavior.

Keep selection itself (single-select, multi-select, focus mode) working, but without automatic viewport movement.

## Requirements

- Remove any code paths that pan/zoom/fit/animate the viewport in direct response to selection changes.
- Remove diagonalization logic and any derived layout changes tied to it.
- Keep selection highlighting and Details panel behavior intact.
- Keep graph layout stable across refreshes (no new jitter).

## Acceptance Criteria

- Selecting nodes/edges never triggers an automatic viewport change.
- Focus mode does not introduce diagonal repositioning of nodes.
- Graph remains usable for typical dogfooding flows (select, multi-select, inspect details).
- `just check` remains green.
