---
epic: gpui
branch:
  suggested: rn/gpui/T-82-header-chrome-polish
rn:
  parent: T-80
---

# T-82 UI: header chrome polish v1 (compact status chips + calmer toolbar) (Domain 6)

## Problem

The desktop app header still feels “webby” / heavy:

- The full-width elevated bar consumes visual attention and makes the app feel less native.
- The control plane + daemon status text is verbose and always-visible, even though most of the
  details are only occasionally useful.
- Global actions (refresh/theme/settings) are fine, but the chrome is not yet cohesive with the
  calmer filter system.

## Goals

1. **Calmer top toolbar**
   - Keep a single header row for now, but reduce its visual weight (less “gray slab”).
   - Preserve clarity and contrast in both dark/light modes.
   - Avoid adding new always-visible controls.

2. **Compact connection/status chips**
   - Replace the verbose status text cluster with compact chips.
   - Chips should show a minimal, scan-friendly summary (e.g. running / embedded).
   - On hover, show a richer “hovercard” with the detailed status (host id, mode, etc.).

3. **No coupling to the graph**
   - This is global chrome; it must not assume the graph view is the only workspace surface.
   - Reuse existing UI primitives where possible.

4. **Keyboard-friendly**
   - Do not regress existing keyboard workflows (filters, command palette, selection actions).
   - Prefer action-based wiring (not ad-hoc key listeners).

## Acceptance criteria

- Header looks calmer and more native than the current `surface_elevated` slab.
- Control plane + daemon status is represented as compact chips with hover detail.
- No layout regressions in Sessions pane / Workspace pane.
- `cargo run -p redesmyn_desktop` works in debug and `--release`.

