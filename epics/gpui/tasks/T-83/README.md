---
epic: gpui
branch:
  suggested: rn/gpui/T-83-session-pane-resize-perf
rn:
  parent: T-82
---

# T-83 UI: session pane resize performance (SplitPane deferred resize) (Domain 6)

## Problem

Resizing the Sessions pane (left side of the main split) currently has **terrible performance**.
The primary culprit is per-pixel live resize causing expensive, repeated reflow in pane contents
(notably SessionView/chat history + markdown/text shaping), making the divider drag gesture janky.

## Goals

1. **Smooth divider dragging**
   - Dragging the split divider should stay responsive in debug and release builds.
   - The UI should provide a clear affordance for the *target* size while dragging.

2. **Preserve correctness + persistence**
   - Persist the committed split size only when the drag ends (mouse up), as today.
   - Do not regress collapse/expand behavior or state persistence.

3. **Reusable primitive**
   - The optimization should be implemented as a `SplitPane` feature (not a one-off hack in the
     desktop root view), since other split layouts may be added later.

## Approach

- Add `SplitPaneResizeMode`:
  - `Live`: current behavior (resize panes while dragging).
  - `Deferred`: keep pane layout fixed during drag, render a “ghost” divider at the preview
    position, and apply the new size on mouse up.
- Use `Deferred` for the main desktop split (Sessions pane ↔ Workspace pane).

This preserves the ability to resize while making the drag gesture cheap (only the ghost line
updates during the drag), avoiding per-pixel reflow of expensive content.

## Acceptance criteria

- Sessions pane divider drag is smooth (no obvious stutter) in debug builds and release builds.
- Divider still:
  - double-click collapses/expands,
  - emits `SplitPaneEvent::StateChanged` with the committed state,
  - respects min/max constraints.
- No visual/layout regressions in either pane.
