---
epic: gpui
branch:
  suggested: rn/gpui/T-63-left-pane-pinned-chat-session-view
rn:
  node:
    branch: rn/gpui/T-63-left-pane-pinned-chat-session-view
  parent: T-47
  after:
    - T-59
    - T-61
    - T-62
---

# T-63 Left pane: pinned chat session viewer (no “overseer” naming) (Domain 7)

## Problem

Domain 5 establishes the desktop shell layout and the existence of user-managed chat sessions pinned to epics.

But without a real session viewer in the left pane:

- the primary “always-visible orchestration chat” is not functional,
- we can’t validate performance or UX of long chat histories,
- and we can’t test “no silent actions” end-to-end in the desktop UI.

## Goal

Populate the desktop shell’s left pane with the reusable `SessionView` component, showing the current epic’s pinned chat session (0/1) with full functionality:

- create/pin,
- unpin,
- close,
- and compose messages.

The UI must not introduce a new named construct (“overseer” is internal shorthand only).

## Requirements

### 1) Epic → pinned chat (0/1)

When an epic is selected:

- if it has a pinned chat session, display it in the left pane
- otherwise show an empty state with clear CTA(s):
  - “Create chat” (creates a new chat session and pins it to the epic)
  - optionally “Pin existing…” (future/optional; see below)

### 2) Pinned chat actions (no silent actions)

From the left pane, support:

- create+pin chat to current epic
- unpin (clears the epic’s pin; chat remains available in history)
- close (marks the chat session closed; does not delete history)

All actions must:

- show immediate in-flight feedback (no spinner wheels),
- prevent duplicate requests while pending,
- and surface actionable errors without discarding drafts.

### 3) Optional: “pin existing chat”

If cheap, include a minimal UX to pin an existing chat session to the current epic:

- modal/palette listing recent chat sessions (title + last updated),
- selecting one sets/replaces the epic’s pin.

If not cheap, leave as follow-up work; the port can ship with create+pin only.

### 4) SessionView integration

Embed the reusable `SessionView`:

- history + live subscription,
- markdown rendering,
- composer,
- and scroll behaviors.

### 5) Testability surfaces

Extend semantic UI snapshot (T-48) to include:

- current epic id,
- pinned chat session id (or null),
- left pane visible/collapsed,
- composer pending/error state.

## Acceptance criteria

- Selecting an epic shows its pinned chat or a useful empty state.
- Creating a chat immediately pins it and opens the composer.
- Unpin/close work and are reflected in the UI snapshot.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on:
  - left pane layout + epic chrome (T-45/T-46),
  - chat sessions + pin semantics (T-47),
  - session viewer pipeline + UI (T-59/T-61/T-62).
