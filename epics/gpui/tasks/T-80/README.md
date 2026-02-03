---
epic: gpui
branch:
  suggested: rn/gpui/T-80-filter-ui-polish
rn:
  parent: T-79
---

# T-80 UI: filter chrome polish (tab background, cascaded menus, fade) (Domain 6)

## Problem

T-79 shipped a functional, Linear-inspired task filter system, but the UI still feels a bit
“demo-ish” compared to Linear:

- The filter row takes vertical space, visually “cutting off” the graph.
- The filter popover is a single large panel; Linear uses cascaded menus where the secondary menu
  appears on hover.
- The popover appears/disappears abruptly (no opacity transition).

## Goals

1. **Tab-like filter chrome**
   - The filter affordance (Filter button + active filter chips) should be a “tab” overlay that
     sits above the graph rather than consuming vertical layout space.
   - The tab background should be translucent (“frosted glass”-ish), with no hard border.
   - The tab should size to its contents (chips can grow/shrink without forcing a full-width bar).

   Note: true per-element backdrop blur / custom shaders may not be available via the current GPUI
   public API. If blur isn’t feasible, use a tasteful translucent surface + shadow that still feels
   intentional and native.

2. **Linear-like cascaded filter menus**
   - Primary menu: “Add filter…” + categories list.
   - Secondary menu: values for the hovered category (appears only after hovering a category row).
   - Secondary menu should not render until a category is hovered (matches Linear behavior).

3. **Menu fade-in/out**
   - When opening/closing the filter menu, fade the menu surfaces in/out via an opacity transition
     (no spinner wheels).
   - Click-outside still closes immediately, but the visual should fade smoothly.

4. **Keep filters reusable**
   - Filter model remains shared in `redesmyn_ui::task_filters`.
   - UI implementation stays at the workspace layer (not graph-owned chrome), so future list views
     can reuse the same constructs.

## Acceptance criteria

- Graph no longer gets “pushed down” by a full-width filter bar.
- Filter tab looks native and calm (translucent surface; minimal borders).
- Cascaded menus behave like Linear (secondary appears on hover).
- Filter popover fades in/out (opacity transition) and remains keyboard-accessible:
  - `f` opens and focuses “Add filter…”
  - `esc` closes.
