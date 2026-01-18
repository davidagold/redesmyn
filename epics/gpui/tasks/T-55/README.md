---
epic: gpui
branch:
  suggested: rn/gpui/T-55-graph-details-panel
rn:
  parent: T-46
  after:
    - T-44
---

# T-55 Details panel (drawer) + selection model integration (Domain 6)

## Problem

The graph is “graph-first”, but selection needs a place for actionable detail:

- task details and controls,
- edge details (e.g. merge base/head refs),
- and eventually session/diff entrypoints.

In the web UI, selection drives a right-side details panel/drawer. The desktop UI needs an equivalent pattern without consuming permanent width.

## Goal

Implement a GPUI details panel that:

- opens when a node or edge is selected,
- integrates with the graph selection model,
- and provides a stable home for actions (even if many actions are stubbed initially).

## Requirements

### 1) Drawer behavior

- Panel is closed by default.
- Selecting a node/edge opens the panel.
- Clearing selection closes the panel.
- The panel width is stable and affects pan-to-selection reserved space (T-54).

### 2) Content (v0)

Provide placeholder but structured content:

- for tasks: title, state, agent status, merge readiness, identifiers
- for edges: from/to labels, and any available metadata

Avoid path leakage:

- do not require local filesystem paths to render.

### 3) Action pattern (no silent actions)

Even if the real actions land later:

- implement the “in flight” UI pattern for at least one dummy action (wired to a no-op command) to ensure the UX pattern is correct.

### 4) Accessibility

- panel open/close must be keyboard accessible,
- no focus traps,
- preserve user input on errors.

## Acceptance criteria

- Selecting a node opens the details panel; clearing selection closes it.
- Pan-to-selection reserves space when the panel is open (T-54).
- The panel content updates correctly as selection changes.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on desktop chrome and selection state wiring (T-46) and UI foundations (T-44).
- Integrates with the graph selection model from Domain 6 (T-50).

## Reference implementation (today; for behavior orientation only)

- Web details panel (today):
  - `dashboard/src/components/layout/DetailsPanel.tsx`
  - Selection routing in `dashboard/src/routes/EpicView.tsx` (node/edge selection).

