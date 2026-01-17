---
epic: harness-interface-v0
branch:
  suggested: rn/harness-interface-v0/T-15-session-as-conversation
rn:
  linear:
    issue_id: null
    identifier: null
  parent: T-14
---

# T-15 Session semantics: `AgentSession` == conversation, turns are events

## Motivation

T-10/T-11/T-12 introduced a “one process per structured turn” runtime model so:

- structured turn completion is deterministic (process exit + `agent.turn_completed`),
- we can enforce “one turn at a time” for a given external conversation id, and
- automation (e.g. T-5 conflict assist) can safely correlate the prompt it sent to a specific completion boundary.

However, the current v0 persistence shape is discordant:

- In **interactive** mode, `AgentSession` behaves like a *conversation/session*:
  - long-lived, attachable (tmux), and “running” represents the live conversation.
- In **structured** mode, we currently create a new `AgentSession` DB row per turn-run, and reuse `external_session_ref` as the real continuity handle.

This creates both conceptual and practical friction:

- “Session” in the UI/API reads as “conversation”, but in structured mode it functions as “turn/run record”.
- “Turn completed but agent still running” becomes confusing (the underlying tmux/process lifecycle can diverge from the conversation lifecycle).
- Active-session selection has to reconstruct the “real conversation” by grouping rows by `external_session_ref`.
- The system ends up with many `AgentSession` rows that are not actually sessions, complicating:
  - stop/restart semantics,
  - attach semantics,
  - status summaries,
  - and future timeline views.

## Decision

For Harness Interface v0:

- `AgentSession` represents an **agent conversation** (one external session/thread, or one interactive harness session).
- A “turn” is represented in the existing append-only **Event stream**:
  - `agent.turn_started`, `agent.turn_completed`, and `agent.assistant_message`.

We explicitly choose **event-as-turn** (no new DB `AgentTurn` table) as the canonical turn timeline.

## Goal

Unify semantics so that:

- sending a message in structured mode continues an existing *conversation* without creating a new `AgentSession` row,
- the UI and API can reliably treat `agent_session_id` as “conversation id”,
- turn lifecycle is observable and correlatable purely from events, and
- we retain the operational benefits of “one process per turn” while keeping persistence coherent.

## Requirements

### 1) `AgentSession` invariants

Define and enforce these invariants:

1. **Conversation identity**
   - Interactive: an `AgentSession` represents the tmux-backed harness conversation for a task.
   - Structured: an `AgentSession` represents the external conversation handle (`external_session_ref`) for a task.

2. **At most one active conversation per task per interface mode**
   - A task may have historical `AgentSession`s, but at most one *active* conversation for a given `task_id` and `agent_interface_mode`.
   - “stop” ends the conversation (sets `ended_at`), it does not end a single turn-run.

3. **`started_at` / `ended_at` represent conversation lifecycle**
   - `started_at`: first time the conversation was created/started.
   - `ended_at`: explicit stop, or terminal failure of the conversation (not merely the end of a structured turn-run).

4. **`external_session_ref` is conversation state**
   - Structured conversations MUST persist a resumable `external_session_ref` once observed.
   - Structured “send message” MUST reuse the same `external_session_ref` for the conversation unless the user explicitly chooses “stop session and start new”.

### 2) Turn-as-event contract (minimum event payload)

Standardize the event payloads required for event-as-turn to work end-to-end.

1. `agent.turn_started`
   - MUST include:
     - `task_id`
     - `agent_session_id` (conversation id)
     - `turn_id` (string; stable within the external harness when available)
     - `interface_mode`
   - SHOULD include:
     - `external_session_ref` (when structured)
     - `log_offset_bytes` (size of the session log file before the turn-run starts; used later for log slicing)
     - `idempotency_key` (for dedupe/retry in automation)

2. `agent.turn_completed`
   - MUST include:
     - `task_id`
     - `agent_session_id` (conversation id)
     - `turn_id` (string; matches the started event when available)
     - `interface_mode`
   - SHOULD include:
     - `external_session_ref` (when structured)
     - `exit_code` and/or a structured error detail (when the turn-run process failed)

3. `agent.assistant_message`
   - MUST include:
     - `task_id`
     - `agent_session_id` (conversation id)
     - `turn_id` (best-effort; null allowed if the harness cannot provide it)
     - message text (bounded/truncated per v0 policy)

### 3) Structured runtime: keep “one process per turn”, but do not create new sessions

Refactor the structured turn runner path so that:

- Each structured turn still runs as a distinct process invocation (v0 design constraint).
- Output is appended to the **same** per-conversation log path (session log file is a conversation artifact).
- Turn boundary events (`agent.turn_started` / `agent.turn_completed`) are emitted and refer to the **conversation** `agent_session_id`.
- The conversation `AgentSession` row is updated in place:
  - `agent_semantic_status.turn_state`
  - `agent_preview` (derived from assistant message events)
  - `external_session_ref` (if newly observed)

### 4) Active session selection for messaging

Update “send message” orchestration (T-12 endpoint handler path) so that:

- When structured and a resumable `external_session_ref` exists, the message is delivered as a new turn in the **same** `AgentSession` conversation.
- When structured and no resumable id exists, the system:
  - starts the conversation, captures/persists the external id onto the conversation, and
  - subsequent messages reuse it.
- When a conflicting conversation is active and the caller requests `stop_session_and_start_new`:
  - end the active conversation (stop), create a new conversation row, and send into it.

### 5) Stop/restart semantics must be conversation-correct

Ensure endpoints and UI affordances remain coherent:

- `stop` ends the conversation (sets `ended_at`, updates semantic status accordingly).
- A structured turn completing does NOT set `ended_at`.
- UI should be able to render:
  - “session exists, idle” (turn completed, conversation active),
  - “session busy” (turn running),
  - “session ended” (stopped).

### 6) Concurrency: one turn at a time per conversation (event-based)

Enforce “no concurrent structured turns” for a given conversation:

- Preferred mechanism: gate on conversation-level “turn in flight” snapshot derived from events (or a single session field updated by AgentDriver).
- The source of truth remains events; any session snapshot must be derivable from them.

### 7) Migration and compatibility (v0)

Provide a safe incremental migration plan:

- Existing DBs may have multiple structured `AgentSession` rows for a single `external_session_ref`.
- Define a deterministic reconciliation rule for selecting the canonical conversation row (e.g. latest active row, else latest row).
- Ensure API/UI continue to function during migration:
  - response payloads remain compatible,
  - existing historical rows remain queryable as “past conversations” or are treated as legacy turn-runs.

### 8) Tests

Add tests that lock the new semantics:

- Sending multiple structured messages for a task:
  - reuses a single `AgentSession` conversation row,
  - appends `agent.turn_started`/`agent.turn_completed` events per message,
  - updates preview from message events.
- Conflict handling:
  - `stop_session_and_start_new` ends the conversation and creates a new one.
- Turn gating:
  - in-flight turn blocks a new one unless explicitly interrupted (when supported).

## Acceptance criteria

- A structured message send does not create a new `AgentSession` row when it is continuing an existing conversation.
- `agent_session_id` in API responses is stable across structured turns for the same conversation.
- Turn timeline is fully representable from events (no new “turn table”).
- The UI can show “turn completed” while the session remains “idle/ready” (not “stopped”).
