---
epic: gpui
branch:
  suggested: rn/gpui/T-45-gpui-main-split-layout
rn:
  parent: T-43
  after:
    - T-44
---

# T-45 Main split layout (left session pane + right workspace; resizable + collapsible) (Domain 5)

## Problem

The web UI currently spends persistent width on a left navigation sidebar, but we are not using most of those destinations.

In the desktop app we want to spend that width on work:

- a persistently visible, collapsible **left session pane** (epic-scoped),
- and a graph-first right pane (graph + details).

If the layout is not designed early, later work (session viewer, diff viewer, graph) will fight the shell and cause costly refactors.

## Goal

Implement the top-level desktop layout:

- no persistent navigation sidebar,
- a horizontal split with a **left session pane** and **right workspace pane**,
- the left pane is **resizable** and **collapsible**,
- state is persisted locally (pane width + collapsed state).

This ticket implements layout only; the session viewer and graph content can be placeholders.

## Requirements

### 1) Layout structure

- Root window contains a horizontal split:
  - Left: `EpicSessionPaneHost` (placeholder view; later replaced by Domain 7).
  - Right: `WorkspacePaneHost` (placeholder; later replaced by Domain 6).

### 2) Left pane sizing + persistence

- Left pane defaults to a “minority of width” but materially wider than the old sidebar.
- Support:
  - drag resize,
  - collapse toggle,
  - sensible min/max widths,
  - persistence across restarts.

### 3) Accessibility and interaction

- Collapsing must not trap focus.
- Keyboard shortcuts are allowed (optional), but not required for v0.
- No spinner wheels.

### 4) No implicit “targeting”

Do not introduce any implicit coupling where the left pane targets the selected task by default.

We keep this as an explicit future UX improvement.

## Acceptance criteria

- Desktop app shows the split layout reliably.
- Left pane can be resized and collapsed and the state persists.
- Placeholders render without performance issues.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on GPUI app bootstrap (T-43).
- Depends on shared UI foundations (T-44) for consistent controls.

## Reference implementation (today; for behavior orientation only)

- Root layout (web today):
  - `dashboard/src/components/layout/RootLayout.tsx` (currently renders `Sidebar` + `Outlet`).
  - `dashboard/src/components/layout/Sidebar.tsx` (we remove this entire concept in desktop).

