---
epic: gpui
branch:
  suggested: rn/gpui/T-77-collapsed-task-card-parity
rn:
  parent: T-52
---

# T-77 Collapsed task card parity (header/title/preview/status) (Domain 6)

## Problem

The GPUI graph’s collapsed task cards are currently too minimal (slug-only). The web app’s task
cards provide a compact, information-dense summary that helps users scan the graph quickly:

- task slug + title,
- a short “latest activity” preview,
- and clear state affordances (status + agent).

## Goal

Bring the collapsed GPUI task card up to parity with the web task card as much as is practical
given the current Rust read model.

## Requirements

- Render:
  - branch slug (muted header) and
  - task title (primary; wraps to max 2 lines, ellipses only on the final line).
- Show a single-line “latest activity” preview when available.
  - Source: epic graph `session_summaries` per task (message preview).
  - Render as inline markdown (no raw backticks; no link navigation in this preview).
- Use state affordances consistent with the web view:
  - Border color reflects task/merge readiness (no status badge).
  - Agent status is shown as a small dot in the header corner.
- Hover interactions:
  - Hover target is the entire card.
  - Quick action icons fade in/out via opacity and do not cause layout shifts.
- Keep layout stable under camera zoom (no overflow/flicker).

## Acceptance criteria

- A collapsed task node is information-dense and legible at normal zoom:
  - slug + title + preview + state affordances.
- Tasks without a session still render cleanly (no empty placeholder noise).
