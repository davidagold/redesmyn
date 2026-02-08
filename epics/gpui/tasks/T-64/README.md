---
epic: gpui
branch:
  suggested: rn/gpui/T-64-task-session-view-in-expanded-card
rn:
  parent: T-55
  after:
    - T-59
    - T-61
    - T-62
---

# T-64 Expanded task card: show latest task session via SessionView (Domain 7)

## Problem

The UI is graph-first, but users need to inspect and interact with an agent’s work.

Today, the web UI supports:

- a task card preview of the last assistant message,
- a composer to send a message to the task’s agent,
- and attach commands for interactive sessions.

In the GPUI port we want a more coherent session viewer:

- a scrollable conversation history,
- with turn boundaries and durable events,
- integrated into the expanded task card.

## Goal

Integrate the reusable `SessionView` into the expanded task card (T-55) so users can:

- see the latest task session’s history,
- send messages with the same conflict semantics as v0,
- and observe turn lifecycle/status updates.

We intentionally start with “latest session only” to avoid UI bloat; session history browsing is follow-up work.

## Requirements

### 1) “Latest session only” semantics

For a selected task:

- fetch `latest_session_id` (or null),
- if null: show an empty state (“No session yet”) plus actions:
  - “Start agent” (if we include start here) or
  - “Send message” (which can start a session implicitly per T-41 semantics).

Optional (nice): a lightweight “New session” action that corresponds to stop-and-start-new semantics for structured agents.

### 2) Embed SessionView

Render `SessionView` within the expanded task card’s Session column:

- history + live updates,
- markdown rendering,
- composer,
- and “new messages” indicator behavior.

### 3) Interactive session affordances

If the latest session is interactive/tmux:

- show the tmux placeholder view (T-65) in-line,
- include attach/copy actions.

### 4) No silent actions

All mutations in this surface (send/stop/restart/attach) must:

- show immediate in-flight feedback,
- disable the triggering control while pending,
- and be assertable via semantic UI snapshot (T-48).

### 5) Keyboard + focus

At minimum:

- focusing the expanded card allows tabbing to the composer,
- hitting “Send” keeps focus predictable (return to input).

### 6) Camera framing

The graph view should guide focus to the expanded card:

- When the user selects a task: animate pan/zoom so the expanded task card is centered and fills ~80% of the viewport.
- When the user collapses the task card (clears selection): restore the zoom level that was active when the task was selected (no need to restore pan).

## Acceptance criteria

- Selecting a task shows its latest session (or empty state).
- User can send a message and immediately sees pending state and subsequent events.
- Interactive sessions show a clear attach placeholder.
- Selecting a task animates the camera to frame the expanded card; collapsing restores the previous zoom level.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on:
  - expanded task card surface (T-55),
  - session viewer pipeline + list + composer (T-59/T-61/T-62).

## Reference implementation (today; for behavior orientation only)

- Task card preview + composer today:
  - `dashboard/src/components/graph/TaskCard.tsx`
- Send message semantics today:
  - `redesmyn/task_agent_messaging.py`
