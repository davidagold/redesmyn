---
epic: gpui
branch:
  suggested: rn/gpui/T-47-epic-scoped-session-pane
rn:
  parent: T-46
  after:
    - T-40
---

# T-47 User-managed chat sessions + epic pins (no “overseer” naming) (Domain 5)

## Problem

We want the desktop shell to host a persistently visible left session pane with a “check-in/review/orchestration” chat.

Earlier we considered “exactly one chat session per epic”. That is too rigid:

- users should be able to create/close these chats freely,
- and optionally pin them to epics (so multiple epics can be active concurrently without mixing context).

If we don’t define how these chats are identified, pinned, and persisted, later work will either:

- invent ad-hoc UI-only state (hard to test and not shareable across clients), or
- accidentally make “overseer” a first-class user-facing construct (we don’t want that).

## Goal

Implement **user-managed chat sessions** (a kind of session that is not task-bound) and a **pinning model** that associates these sessions to epics.

The UI continues to present this as “just a session/chat view” without introducing a new named concept.

## Requirements

### 1) Session scoping model (task vs chat)

Extend session identity to support a scope like:

- `Task { task_id }` (normal task sessions)
- `Chat` (not task-bound; user-managed conversation)

This is still “just a session”; it is not a new user-facing entity.

### 2) Pinning model (chat ↔ epic)

Support pinning a chat session to an epic:

- a chat can be pinned to **zero or more** epics,
- an epic can have **at most one** pinned chat session (0/1),
- pin/unpin is a lightweight metadata operation (does not end the session),
- pins are persisted in the control plane DB.

Semantics:

- pinning a chat session to an epic is **set/replace**:
  - if the epic already has a pinned chat, it is replaced,
  - the replaced chat remains open (not closed) unless the user explicitly closes it.

### 3) Control plane API surface

Expose methods (names illustrative) over the client API:

- `CreateChatSession { title? } -> { session_id }`
- `CloseChatSession { session_id }` (marks closed/ended; does not delete history)
- `ListChatSessions { filters… } -> { sessions… }` (at minimum: list sessions pinned to an epic)
- `PinChatSessionToEpic { session_id, epic_id }` (set/replace; epic has 0/1 pinned)
- `UnpinChatSessionFromEpic { epic_id }` (clears the epic’s pin)

Semantics:

- chat sessions are durable and queryable (session id stable across restarts),
- closing a chat session is idempotent,
- the epic’s pinned chat session is deterministic and stable.

### 4) Persistence

Persist the mapping in the control plane (preferred), e.g.:

- store sessions in `agent_sessions` (with `scope_kind = chat`),
- store pins in a small table keyed by epic (e.g. `session_pins(epic_id PRIMARY KEY, session_id)`),
- optionally store a `title` and `closed_at` for chat sessions.

Do **not** store this as UI-only local state; we want:

- multi-client consistency (desktop + `rn` + future clients),
- and AI-first testability (stable ids and queryable state).

### 5) UI integration

When the selected epic changes:

- UI shows the epic’s pinned chat session if one exists, otherwise an empty state.
- UI supports creating a new chat session and pinning it to the current epic.
- UI supports pinning an existing chat session to the current epic (replacing any existing pin).
- UI supports closing the currently displayed chat session.

Do **not** implement default targeting of selected tasks from this pane in the port.

### 6) Future direction (not in this ticket)

We may later add explicit UX affordances to “act on selected task(s)” from the epic session, but this is out of scope for the port.

## Acceptance criteria

- The control plane supports creating/closing chat sessions and pinning them to epics.
- Selecting an epic shows a stable list of pinned chat sessions and allows picking one for the left pane (placeholder content is fine).
- Pins and session metadata are persisted in control plane storage and are queryable.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on control plane session persistence (T-40) and client query surfaces (T-12/T-20).
- Builds on desktop shell chrome (T-46) for epic selection integration.

## Reference implementation (today; for orientation only)

- There is no user-managed/pinnable “orchestration chat” today; this is new.
- Task-scoped messaging and session hints today:
  - `dashboard/src/components/graph/TaskCard.tsx` (task message composer).
  - `redesmyn/task_agent_messaging.py` (send-message semantics).
