---
id: T-9
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-9-left-align-commit-titles
---

# T-9 Left-align commit titles (keep trunk spacing invariant)

## Problem

In the trunk commit list view, commit titles are effectively right-aligned against a fixed edge near the trunk.
This makes the text harder to scan and read because each line starts at a different horizontal position.

Reference screenshot: `codex-clipboard-UJR4dn.png`.

## Goal

Improve readability by left-aligning commit titles while preserving the existing “column” / trunk spacing behavior.

## Requirements

- Commit title column width should remain unchanged.
- Maintain the current title-to-trunk spacing invariant:
  - For titles that span the full allowed width (i.e. they hit the truncation/ellipsis behavior), the distance from the trunk should be unchanged.
- For titles that do **not** span the full allowed width, render them left-aligned (consistent start edge).

## Acceptance Criteria

- Short commit titles are left-aligned (readable, easy to scan).
- Long/truncated titles preserve the existing positioning relative to the trunk and SHA column.
- No regressions in node layout/interaction.
- `just check` remains green.
