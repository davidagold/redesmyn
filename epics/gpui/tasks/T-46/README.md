---
epic: gpui
branch:
  suggested: rn/gpui/T-46-gpui-epic-header-chrome
rn:
  node:
    branch: rn/gpui/T-46-gpui-epic-header-chrome
  parent: T-43
  after:
    - T-44
---

# T-46 Epic header + chrome (epic selector, status, refresh, settings entrypoints) (Domain 5)

## Problem

With no navigation sidebar, the desktop app still needs:

- a way to select an epic,
- a way to see daemon/host connectivity status,
- a way to trigger refresh/resync and configuration,
- and a home for global actions (theme toggle, command palette entrypoint).

If we postpone chrome decisions, we will end up scattering “random buttons” across views and reintroduce silent actions.

## Goal

Implement a desktop “header/chrome” surface that is:

- minimal,
- graph-first,
- and consistent with “no silent actions”.

This ticket focuses on the shell: the specific graph/session views can be placeholders.

## Requirements

### 1) Epic selection

Provide an epic selector UI that:

- lists available epics (via control plane query),
- allows changing the selected epic,
- and shows the selected epic’s primary identifiers (name/slug).

### 2) Status cluster

Show daemon/control-plane status in a compact, calm way:

- control plane running (in-proc),
- daemon connected / telemetry freshness (as available),
- and actionable “how to start” help text where relevant.

No spinners; use subtle, accessible indicators and explicit text.

### 3) Refresh / resync entrypoint

Provide a refresh action:

- triggers an explicit control plane query/refresh,
- shows immediate in-flight feedback,
- and prevents accidental double refresh while in flight.

### 4) Global actions without a sidebar

Provide entrypoints (UI affordances can evolve later):

- theme toggle,
- settings/config panel entrypoint (if we keep one),
- command palette entrypoint (optional; see T-49).

## Acceptance criteria

- When an epic is selected, the header updates and remains stable.
- Refresh has immediate in-flight indication and disables itself while running.
- Status surfaces are visible and actionable without a sidebar.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on GPUI app bootstrap (T-43) and UI foundations (T-44).
- Depends on control plane query surfaces (Domain 2; at minimum `ListEpics` / `Status`).

## Reference implementation (today; for behavior orientation only)

- Epic header (web today):
  - `dashboard/src/routes/EpicView.tsx` (header with `EpicSelector`, `ConnectionsCluster`, refresh, config).
  - `dashboard/src/components/layout/EpicSelector.tsx`

