---
epic: gpui
branch:
  suggested: rn/gpui/T-77-collapsed-task-card-parity
rn:
  parent: T-52
---

# T-77 Collapsed task card parity (preview + agent badges) (Domain 6)

## Problem

The GPUI graph’s collapsed task cards are currently too minimal (slug-only). The web app’s task
cards provide a compact, information-dense summary that helps users scan the graph quickly:

- task slug + title,
- a short “latest activity” preview,
- and small agent/session badges.

## Goal

Bring the collapsed GPUI task card up to parity with the web task card as much as is practical
given the current Rust read model.

## Requirements

- Render both:
  - task slug (muted) and
  - task title (primary).
- Show a single-line “latest activity” preview when available.
  - Source: epic graph `session_summaries` per task (message preview).
- Show agent/session badges when available:
  - `turn_id` (e.g. `a-149`) and
  - `kind` (e.g. `Codex`).
- Keep layout stable under camera zoom (no overflow/jank).

## Acceptance criteria

- A collapsed task node is information-dense and legible at normal zoom:
  - slug + title + preview + badges.
- Tasks without a session still render cleanly (no empty placeholder noise).
