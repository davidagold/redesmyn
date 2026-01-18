---
epic: gpui
branch:
  suggested: rn/gpui/T-40-control-plane-session-store
rn:
  parent: T-17
  after:
    - T-14
---

# T-40 Control plane session persistence + query surfaces (sqlx) (Domain 4)

## Problem

We want a native session viewer and “overseer” chat that are:

- delightful,
- fast,
- and queryable.

That requires durable, structured session event storage.

We also have an explicit product requirement:

- Persist all **non-delta / non-chunk** session emissions needed to render conversation history.

If we store raw logs only, we cannot build rich UI affordances (filtering by tool events, linking artifacts, resuming sessions, etc.).

## Goal

Extend the Rust control-plane persistence layer to store:

- agent sessions, and
- structured session events (T-14),

in a way that is:

- performant (indices, compact storage),
- easy to query,
- and supports subscriptions for live UI updates.

## Requirements

### 1) Schema (SQLite via sqlx)

Add tables (names illustrative):

- `agent_sessions`
  - `session_id` (ULID BLOB(16))
  - **scope**:
    - `scope_kind` (`task` | `chat`)
    - `task_id` (ULID BLOB(16), nullable)
  - `repo_id`/`workspace_id` (ULID BLOB(16) / string per our final decision)
  - `agent_kind` (enum string)
  - `interface_mode` (enum string)
  - `status` (running/blocked/stopped/error)
  - `external_session_ref` (small JSON or typed fields; keep query intent in mind)
  - optional metadata for chat sessions:
    - `title` (nullable)
    - `closed_at` (nullable)
  - timestamps
- `session_events`
  - `event_id` (ULID BLOB(16))
  - `session_id`, `task_id`, `repo_id` (indexed)
  - `kind` (small enum string for filtering)
  - `created_at`
  - `payload` (protobuf bytes; JSON optional for debug)
  - preview columns (optional, for fast list rendering)
- `artifacts` (if not already present in Domain 2 schema evolution)
- `session_pins`
  - `session_id` (ULID BLOB(16))
  - `epic_id` (ULID BLOB(16))
  - (unique constraint on `session_id, epic_id`)
  - index by `epic_id` for “list pinned chats for epic”

### 2) Durability policy: no deltas in DB

Persist:

- user messages,
- assistant messages (final messages / completed items),
- tool invocations/results,
- status updates,
- artifact references.

Do **not** persist:

- streaming token deltas / chunked output intended only for live UI.

If a daemon emits delta events, the control plane must either:

- drop them from persistence, or
- coalesce into a final durable event before persisting.

### 3) Storage API (typed)

Expose storage methods (names illustrative):

- `insert_agent_session(...)`
- `append_session_event(...)`
- `list_task_sessions(task_id, pagination)`
- `create_chat_session(title?) -> session_id`
- `close_chat_session(session_id)`
- `list_chat_sessions(filters...)`
- `pin_chat_session_to_epic(session_id, epic_id)`
- `unpin_chat_session_from_epic(session_id, epic_id)`
- `get_session_events(session_id, pagination, filters)`

### 4) Client query surfaces

Define control-plane query methods (over the client API protocol) sufficient for the session viewer:

- list sessions for a task (and current active session)
- manage chat sessions (create/close/list)
- manage pins to epics (pin/unpin/list pinned for epic)
- fetch session event history with pagination
- subscribe to new session events for a session or task

Exact method names belong to the client API schema (T-12), but this ticket must specify what data is required and how it’s indexed.

### 5) Performance constraints

- Queries should be stable O(log n) where possible (proper indices).
- Enforce size limits for persisted payloads; redirect large data to artifacts.

## Acceptance criteria

- Migrations apply cleanly and create the session tables.
- A test can:
  - create a session,
  - append events,
  - query them back with pagination,
  - and filter by kind.
- Storage format follows the epic rules:
  - ULIDs as `BLOB(16)`,
  - payload as protobuf bytes (JSON debug optional).

## Dependencies / sequencing

- Depends on control-plane schema/migrations foundations (T-17) and the session event contract (T-14).
- Used by control-plane agent command layer (T-41) and the session viewer domain (Domain 7).

## Reference implementation (today; for behavior orientation only)

- Agent session persistence (Python today):
  - `redesmyn/db/models.py` (`AgentSession` row; preview + external_session_ref persisted).
- Session-ish event emission (Python today):
  - `redesmyn/agent_driver.py` emits:
    - `agent.turn_started`, `agent.turn_completed`, `agent.assistant_message`
    - and `task.agent_session_update` events.
- UI expectations (TS today):
  - `dashboard/src/components/graph/TaskCard.tsx` (message composer; pending states).
  - `dashboard/src/hooks/useEventStream.ts` (agent events in the stream).
