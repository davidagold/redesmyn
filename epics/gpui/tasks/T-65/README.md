---
epic: gpui
branch:
  suggested: rn/gpui/T-65-interactive-session-placeholder
rn:
  node:
    branch: rn/gpui/T-65-interactive-session-placeholder
  parent: T-59
  after:
    - T-44
    - T-36
---

# T-65 Interactive (tmux) session placeholder UX (Domain 7)

## Problem

Shell/tmux agents remain a compatibility path in the port, but:

- the native session viewer is not a terminal emulator,
- and we should not fake a transcript by duplicating raw logs into the UI.

Without a clear interactive-session UX, users will:

- expect chat history where none exists,
- be confused about how to “see what the agent is doing”,
- and lose confidence in the product’s responsiveness (especially around attach).

## Goal

Provide a clear, delightful placeholder view for interactive/tmux sessions in the native session viewer surfaces (left pane and task details), with strong affordances to attach via tmux and with “no silent actions” guarantees.

## Requirements

### 1) Placeholder visual design (recommended)

When `SessionView` is asked to show an **interactive** session:

- show a compact header:
  - “Interactive session (tmux)”
  - current status (running/blocked/stopped/error)
  - session age / started_at
- show a callout body explaining:
  - “This session runs in tmux. The native viewer does not render its terminal output yet.”
  - “Attach to view/interact.”
- actions (buttons):
  - **Attach** (primary)
  - **Copy attach command**
  - optional: **Stop session** / **Restart** (if these actions are present in the surface)

Avoid heavy borders; prefer spacing + subtle separators (UI conventions).

### 2) Attach behavior (no silent actions)

Since the desktop app can’t assume an embedded terminal emulator:

- “Attach” should:
  - immediately show a pending state (e.g. “Preparing attach…”),
  - then either open a terminal command via OS integration (if supported), or
  - fall back to copying the command + showing a small “Copied” confirmation.

The canonical attach command should match `rn`:

- `rn agent attach --task <task_id>`

### 3) Optional: show log artifact link (nice-to-have)

If the daemon produces a log artifact reference for interactive sessions (recommended in T-36/T-14):

- show a “Open log” link that opens the artifact in an appropriate viewer.

Do not inline the full log content in the session viewer for the port.

### 4) Testability

Expose in the semantic UI snapshot:

- whether the current session view is “interactive placeholder” vs “chat feed”,
- whether attach is pending,
- and the exact attach command string (so AI can run it).

## Acceptance criteria

- Interactive sessions show a clear, non-confusing placeholder with attach actions.
- Clicking attach produces immediate visible progress and deterministic state transitions in the UI snapshot.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on:
  - Shell/tmux runtime plan (T-36),
  - UI foundations (T-44),
  - session viewer scaffolding (T-59).

## Reference implementation (today; for behavior orientation only)

- Attach UX precedent:
  - `dashboard/src/components/graph/TaskCard.tsx` copies `rn agent attach --task ...`
