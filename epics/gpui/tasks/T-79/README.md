---
epic: gpui
branch:
  suggested: rn/gpui/T-79-linear-filters
rn:
  parent: T-71
---

# T-79 UI: Linear-style task filters (shared, prune graph)

## Problem

The Rust/GPUI desktop UI needs a **first-class filtering experience** for tasks that:

- feels **native / production-quality** (not “debug bar” UI),
- is **KISS** (avoid lots of configurators),
- matches a **Linear-like** workflow (Filter button → popover → categories → multi-select values → grouped chips),
- and is **not coupled to the graph view** so it can be reused for upcoming list-like views.

Today, the graph renders with a “debug” style header inside `GraphView`:

- `rust/crates/redesmyn_ui_graph/src/view.rs:2115` (`debug_bar`), currently showing selection + pan/zoom hints.

We want to replace this direction with a real filter bar and popover.

## Goals

1. **Linear-like filter UX**
   - A **Filter** button in the workspace header area (shamelessly copy Linear’s interaction model).
   - Clicking it opens a **popover** with:
     - an “Add Filter…” input (search/filter categories),
     - a left pane listing filter categories,
     - a right pane listing values for the selected category,
     - multi-select values with checkmarks,
     - a clear/close affordance (and click-outside closes).
   - Active filters show as **grouped chips** (one chip per category) with labels similar to Linear:
     - “`<Category> is any of <N> <category plural>`”
     - Clicking a chip reopens the popover focused on that category.
     - Each chip has an `×` to clear that category.

2. **Keyboard shortcuts**
   - `f` opens/focuses the filter popover (“Filter” shortcut).
   - `cmd-f` / `ctrl-f` is reserved for a future “Find” feature (do **not** bind it to filters).

3. **Prune semantics (default; no “dim vs prune” toggle)**
   - Filtering should **prune** the visible graph to make large graphs inspectable.
   - Always-on policy: **collapse ancestors**.
   - Context: include **one level of children** under matching nodes.

4. **Reusable across views**
   - The filter model and UI must **not live in `GraphView`** (no GraphView-owned chrome).
   - The same filter model + UI components should be usable by:
     - graph view (prune the scene),
     - and a future list view (filter a list).

## Non-goals (for this ticket)

- Server-side filtering / query parameters (filters are client-side for now).
- Advanced filter types (date ranges, custom predicates, saved views).
- “Dim” mode, or a UI toggle between multiple filtering modes.

## Design / architecture notes

### Presentation model (shared)

Introduce a **shared task filter model** (presentation state) that lives at the **workspace layer**, not in `GraphView`.

- Graph view consumes the filter model to produce a pruned scene (layout should operate on visible nodes only).
- Future list view consumes the same model to filter rows.

### Popover/overlay implementation

Use the same “absolute overlay + click-outside to close” pattern as the epic selector menu:

- `rust/crates/redesmyn_desktop/src/root_view.rs:1784` (`epic_menu_overlay`)

Key idea: render an `.absolute().inset_0().occlude()` backdrop and a positioned card on top.

### Aesthetics + UX

- Avoid “debug strip” styling.
- Use the existing theme tokens and shared button/input components from `redesmyn_ui`.
- Keep borders minimal; rely on spacing, subtle shadows, and muted separators.

## Acceptance criteria

- Filter bar appears in the workspace UI (not inside `GraphView`) with:
  - Filter button
  - grouped chips
  - visually cohesive styling with the rest of the desktop chrome.
- Filter UI does **not** include graph-debug affordances (zoom level, pan hints, etc.).
- `f` opens the popover and focuses “Add Filter…”.
- Selecting filter values prunes the graph:
  - ancestors collapsed,
  - 1-level children included,
  - selection remains stable when possible (clears if the selected node disappears).
- Graph view no longer renders the existing `debug_bar` header.
- Logging:
  - add deliberate spans/events for open/close and filter application (no per-frame spam).
