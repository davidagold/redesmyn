---
epic: harness-interface-v0
branch:
  suggested: rn/harness-interface-v0/T-13-surface-external-session-id
rn:
  linear:
    issue_id: null
    identifier: null
  parent: T-12
---

# T-13 UI: surface external session id (thread/session) when available

## Problem

For structured harnesses, Redesmyn often has an **external resume handle** for the running conversation:

- Codex: `thread_id` (and sometimes a cursor like `turn_id`)
- Claude Code: `session_id`

This is stored as `AgentSession.external_session_ref` and used for resume-by-id turns (T-11) and task-card messaging
(T-12), but the UI does not surface it anywhere.

That makes debugging and operational workflows harder:

- users can’t quickly confirm whether a task is “bound” to a specific conversation
- users can’t copy the id to correlate with harness-native logs/UIs
- it’s harder to reason about “resume vs new session” behaviors when messaging

## Goal

When an agent session has an external session/thread id, show it in the task card chrome (near the agent kind badge),
using a short, scannable presentation with a one-click copy affordance.

This is a UI-only feature; it should not change session selection rules (T-12) or resume semantics (T-11).

## Naming (v0)

We need one user-visible name for this construct.

Proposal:

- **Session id** (UI label)
- internal field name remains `external_session_ref`
- tooltip/secondary text clarifies the source: “Codex thread id” / “Claude session id”

## Requirements

### 1) Display rules

- If the currently active/relevant `AgentSession` for a task includes `external_session_ref`, render an id chip.
- If `external_session_ref` is absent (interactive-only sessions), omit the chip entirely.
- If `external_session_ref` is present but unknown/unparseable, omit the chip and log/debug only (no UI noise).

### 2) Formatting + copy

- Render a short, stable abbreviation (e.g. `abcd1234…`) with the full id available:
  - on hover (tooltip), and
  - via a copy button that copies the full id.
- Copy behavior must be explicit and non-destructive:
  - no modal, no spinner
  - show a small “Copied” confirmation inline/toast consistent with dashboard conventions

### 3) Placement

Place the id chip in the same horizontal row as agent identity/kind badges (e.g. `a-<task_id>`, `Codex`), matching
the visual intent in the provided mock (id chip adjacent to the agent kind).

### 4) Accessibility

- Copy control is keyboard reachable and has an aria-label that includes the full label, e.g. “Copy session id”.
- Tooltip content is readable in both light/dark and does not trap focus.

## Acceptance criteria

- For structured Codex sessions, the UI shows the Codex thread id (abbreviated) and allows copying the full id.
- For structured Claude Code sessions, the UI shows the Claude session id (abbreviated) and allows copying the full id.
- For interactive-only sessions (no `external_session_ref`), the UI shows nothing extra.
- The feature does not introduce additional API calls or alter message/session routing behavior.
