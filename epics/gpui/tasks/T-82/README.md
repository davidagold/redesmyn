---
epic: gpui
branch:
  suggested: rn/gpui/T-82-header-chrome-polish
rn:
  node:
    branch: rn/gpui/T-82-header-chrome-polish
  parent: T-80
---

# T-82 UI: header chrome polish (connections menu + calmer toolbar) (Domain 6)

## Problem

The desktop app header still feels “webby” / heavy:

- The full-width elevated bar consumes visual attention and makes the app feel less native.
- The control plane + daemon status presentation is verbose / “button-y”, even though most of the
  detail is only occasionally useful.
- Manual “refresh” is a web-app affordance; the desktop UI should stay current via realtime updates
  and explicit state/subscription events.
- Global actions (theme/settings) are fine, but the chrome is not yet cohesive with the
  calmer filter system.

## Goals

1. **Calmer top toolbar**
   - Keep a single header row for now, but reduce its visual weight (less “gray slab”).
   - Preserve clarity and contrast in both dark/light modes.
   - Avoid adding new always-visible controls.

2. **Connections as an inspectable menu (not chips)**
   - Replace the “Control plane / Daemon” chips with a single ghost control: **Connections**.
   - The control should remain compact and non-“button-y”.
   - On click (or hover, later), show a small menu with richer detail (control plane status, daemon
     mode, host id, etc.).
   - No user-visible action should be “silent”: if connection state is being recomputed, show an
     in-progress affordance.

3. **Remove manual refresh**
   - Remove the header refresh button.
   - Keep any internal refresh/reload logic, but rely on event-driven updates for correctness.

4. **No coupling to the graph**
   - This is global chrome; it must not assume the graph view is the only workspace surface.
   - Reuse existing UI primitives where possible.

5. **Keyboard-friendly**
   - Do not regress existing keyboard workflows (filters, command palette, selection actions).
   - Prefer action-based wiring (not ad-hoc key listeners).

## Acceptance criteria

- Header looks calmer and more native than the current `surface_elevated` slab.
- Control plane + daemon status is represented via a single **Connections** ghost control with a
  detail menu.
- Header does not expose a manual refresh button.
- No layout regressions in Sessions pane / Workspace pane.
- `cargo run -p redesmyn_desktop` works in debug and `--release`.
