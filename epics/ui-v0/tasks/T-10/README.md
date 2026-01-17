---
id: T-10
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-10-epic-menu-hover-full-width
---

# T-10 Epic menu hover background should be full width

## Problem

In the epic switcher dropdown, the hover/active background does not span the full menu width.
This makes the menu feel visually misaligned and reduces the perceived click target.

Reference screenshot: `codex-clipboard-DY2nXq.png`.

## Goal

Make the hover (and keyboard-focus) background fill the full width of each menu row while preserving the current layout and spacing.

## Requirements

- Hover background spans the full available menu width for each item.
- Selected state styling remains consistent and legible.
- Keyboard navigation/focus states should match hover width (no “partial row” highlight).
- Avoid adding extra borders/lines; keep the menu feeling clean and calm.

## Acceptance Criteria

- Hovering any epic shows a full-width background highlight.
- The clickable area feels consistent with the visual highlight.
- `just check` remains green.
