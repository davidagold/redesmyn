---
epic: gpui
branch:
  suggested: rn/gpui/T-83-cascading-menu-polish
rn:
  parent: T-81
---

# T-83 UI: cascading menu polish + shared menu primitives (Domain 6/7)

## Problem

We now have multiple menu surfaces in the GPUI port (task filters, session settings, and more to
come). The initial cascading menu work is functional, but:

- styling/metrics are duplicated between menus,
- option selection indicators are reimplemented per menu,
- multiple menus can be opened simultaneously (overlapping UX),
- UI-driver smoke flows duplicate launch/connect boilerplate and are annoying to extend.

## Goals

1. **Shared styling + metrics**
   - Standardize cascading menu styling in `redesmyn_ui::components::cascading_menu` so all callers
     inherit compact padding/typography and consistent hover/keyboard highlight treatment.

2. **Single-open menu policy**
   - Enforce “only one cascading menu open at a time” via global state, so opening one menu closes
     any other open cascading menu.

3. **Shared option indicators**
   - Provide reusable checkbox/radio indicators for cascading-menu option rows (task filters,
     session settings) so highlight/contrast behavior stays consistent across the app.

4. **UI-driver smoke dedupe**
   - Introduce a tiny helper harness inside the UI-driver implementation to keep smoke commands
     concise and deterministic (no repeated launch/connect boilerplate).

## Acceptance criteria

- Filter + Session Settings menus share styling and option row behavior without per-site
  duplication.
- Opening a cascading menu automatically closes any other open cascading menu.
- Option rows show consistent hover/keyboard highlight and indicator contrast.
- UI-driver smoke commands stay small/readable and are easy to extend.

