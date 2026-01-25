---
epic: gpui
branch:
  suggested: rn/gpui/T-62-session-composer-and-conflicts
rn:
  parent: T-59
  after:
    - T-41
    - T-44
---

# T-62 Session composer: send message + conflict/confirm UX (Domain 7)

## Problem

The session viewer is not useful without a composer:

- users need to send messages to agents (task sessions) and to user-managed chats,
- the UI must preserve “no silent actions” (instant in-flight indicators),
- and we must preserve conflict semantics from v0 (fail vs interrupt vs stop-and-start-new).

If we re-invent these semantics at the UI layer, we will:

- diverge from `rn`,
- break remote-daemon readiness,
- and make testability harder.

## Goal

Implement the session viewer composer UI and hook it up to the headless control plane’s agent-message APIs, preserving v0 conflict behavior and UX affordances.

## Requirements

### 1) Composer UX (baseline)

- multi-line text input
- “Send” action
- disabled while sending (prevent duplicates)
- draft preserved on errors
- visible pending indicator (“Sending…”, no spinner wheels)

### 2) Conflict semantics (must match v0)

Implement the same confirm flows we have today:

- structured turn in progress → offer “Interrupt & send”
- incompatible/other session conflict → offer “Stop session & send”

The UI must key off **stable conflict codes** returned by the control plane (see T-41).

### 3) Scope support: task vs chat sessions

Support composing into:

- task-scoped sessions (normal task agent sessions)
- chat sessions (user-managed, pinned to epics)

Recommendation for v1:

- chat sessions are **structured-only** (Codex by default; see “Agent defaults” below).
- if an implementation chooses to allow interactive chat sessions, the UI may show the “tmux-only” placeholder instead of a transcript (T-65).

### 4) Persistence expectations

When the user sends a message:

- a durable `UserMessage` session event is appended (T-14/T-40),
- the corresponding turn lifecycle events occur (`TurnStarted` / `TurnCompleted`) for structured agents,
- assistant responses become durable `AssistantMessage` events,
- delta/chunk emissions (if any) may render live but are not persisted.

Event source note:

- For structured agents, assistant/turn events arrive via the session event subscription stream (T-59), sourced from daemon-emitted `DaemonMessage::SessionEventBatch` frames (not as inline responses to `SendSessionMessage`).

### 5) Agent defaults for chat sessions (recommendation)

For user-managed chats (left pane), prefer a separate “chat harness” default so we can keep tasks Shell/tmux-compatible while keeping the epic chat structured:

- default: **Codex (structured)** for chat sessions,
- configurable separately from task harness defaults (in `redesmyn_config` in the Rust port),
- allow an advanced override at chat creation time (future; not required for the port UI).

### 6) AI-first testability

Expose enough semantic UI state so tests can assert:

- draft text,
- send pending state,
- last error/conflict prompt visible,
- and message count / last message preview in the view-model.

Integrate with UI driver (T-48) in T-66.

## Acceptance criteria

- A user can send a message from the session viewer and immediately sees a pending state.
- Conflict prompts appear with the correct options (interrupt vs stop-and-start-new), and user choice is honored.
- Draft is not lost on error.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on:
  - control plane agent command semantics + conflict codes (T-41),
  - UI foundations (T-44),
  - session viewer pipeline (T-59).

## Reference implementation (today; for behavior orientation only)

- UX + confirm dialogs:
  - `dashboard/src/components/graph/TaskCard.tsx` (composer + pending states).
  - `dashboard/src/components/agents/AgentMessageConfirmDialog.tsx`.
- Semantics:
  - `redesmyn/task_agent_messaging.py` (conflict classification and behaviors).
