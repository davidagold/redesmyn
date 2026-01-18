---
epic: gpui
branch:
  suggested: rn/gpui/T-59-session-viewer-foundations
rn:
  parent: T-45
  after:
    - T-14
    - T-20
    - T-40
    - T-44
---

# T-59 Session viewer foundations: event→view model + pagination + subscriptions (Domain 7)

## Problem

We want a native session viewer that is:

- delightful and fast,
- reusable across UI surfaces (left pane chat + task session view),
- and correct under “session == conversation; turns are events” semantics.

If we do not define a stable “event → UI view model” pipeline early, we will:

- bake UI-specific assumptions into storage/protocol,
- struggle with pagination + live updates + scroll stability,
- and make AI-first testability brittle.

## Goal

Define and implement the foundational session viewer data pipeline:

- control-plane query/subscription shapes (as consumed by the UI),
- a typed session viewer view-model,
- pagination semantics (older history),
- and live append semantics (new events),

so that later UI work (markdown rendering, composer, pinned chats, task details) can proceed in parallel.

## Requirements

### 1) Session semantics: session == conversation; turns are events

The session viewer must treat `SessionId` as a **conversation id**:

- in interactive mode, and
- in structured mode for agents that expose a resumable conversation id (Codex at minimum; ideally CC too).

Turn boundaries are represented in the session event stream (not via new session rows), consistent with:

- `epics/harness-interface-v0/tasks/T-15/README.md`
- `epics/gpui/tasks/T-14/README.md` (port contract)

### 2) Control plane query surfaces (consumer-driven)

Specify (and implement in the Rust control plane) the minimal query/subscription APIs needed by the session viewer.

Names are illustrative; exact API belongs in the client protocol (T-12).

**History**

- `GetSessionEvents { session_id, before?: Cursor, limit, kinds?: [...] } -> { events, next_cursor }`
  - Cursor should be stable and deterministic. Recommended: `(created_at, event_id)` tuple.
  - Must support backwards pagination (load older).

**Live**

- `SubscribeSessionEvents { session_id, after?: Cursor } -> stream<SessionEvent>`
  - Must guarantee ordering within a session.
  - Must allow resync signaling (if events were missed).

**Session lookup**

- Task details: `GetLatestTaskSession { task_id } -> { session_id? }`
- Left pane: `GetEpicPinnedChatSession { epic_id } -> { session_id? }` (0/1 pinned)
- Optional: `ListChatSessions { ... }` (for “pin existing chat” UX)

### 3) View-model types (UI-internal, strongly typed)

Define a UI view-model layer that is the *only* thing the GPUI views render.

Suggested types:

- `SessionEventRow` (event id, created_at, kind, payload, derived preview fields)
- `SessionTimelineItem` (renderable item):
  - message blocks (user/assistant),
  - turn boundary separators,
  - tool invocation/result blocks,
  - status updates,
  - “load older” affordance row.
- `SessionFeedState`
  - ordered items,
  - cursor bookkeeping,
  - “at bottom” + scroll anchoring state,
  - “loading older” / “live syncing” states (no spinner wheels; use calm progress affordances).

Rule: the view-model must be deterministic and easy to snapshot for AI-first tests (T-48).

### 4) Event ordering + scroll stability

Define and implement scroll semantics that avoid “jumping”:

- When new events arrive:
  - if the user is at bottom (or within a small threshold), auto-scroll to bottom,
  - otherwise, do not move the scroll position; show a “new messages” indicator row/button.
- When older events are loaded and prepended:
  - preserve the user’s visible anchor (no jump).

### 5) Delta/chunk policy (durability vs live UI)

Session viewer must support:

- durable events (persisted; used for history),
- optional live-only delta/chunk events (not persisted) as ephemeral UI updates.

Rule: durable history must be renderable without any delta events.

### 6) Crate/module boundaries

Add a dedicated UI crate/module (names illustrative):

- `redesmyn_ui_session` (views + view-model + rendering)
- optionally `redesmyn_session_view_model` (pure view-model logic) if it helps parallelism/testing

Keep it independent from the graph UI crate; integrate via composition in later tickets.

## Acceptance criteria

- A minimal GPUI “SessionView” scaffold can:
  - request history pages,
  - subscribe to live events,
  - render a placeholder list of timeline items (no markdown yet),
  - and preserve scroll stability when prepending/append.
- View-model logic is unit tested without GPUI.
- The consumer-driven API requirements are reflected in the client protocol + control plane plan (T-12/T-40), with explicit cursor semantics.

## Dependencies / sequencing

- Depends on:
  - session event contract (T-14),
  - client protocol over UDS (T-20),
  - session persistence/query surfaces (T-40),
  - GPUI UI foundations (T-44).
- Unblocks:
  - markdown rendering (T-60),
  - virtualized UI and composer (T-61/T-62),
  - pinned chat + task details integrations (T-63/T-64),
  - testability surfaces (T-66).

## Reference implementation (today; for behavior orientation only)

- Durable-ish event emission today:
  - `redesmyn/agent_driver.py` emits `agent.turn_started`, `agent.turn_completed`, `agent.assistant_message`.
  - `dashboard/src/hooks/useEventStream.ts` defines the stream message shapes.
- Composer UX + “no silent actions” precedent:
  - `dashboard/src/components/graph/TaskCard.tsx` (pending “Sending…” states; disable while in flight).
