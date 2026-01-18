---
epic: gpui
branch:
  suggested: rn/gpui/T-41-control-plane-agent-commands
rn:
  parent: T-19
  after:
    - T-40
    - T-12
    - T-11
    - T-32
---

# T-41 Control plane agent commands + “send message” semantics (conflicts/resume/interrupt) (Domain 4)

## Problem

Agent orchestration is core product behavior. We must preserve key semantics from today:

- resume-by-id turns for structured agents,
- “send message” conflict handling (fail vs interrupt vs stop-and-start-new),
- and interrupt semantics.

If these semantics live in the GUI (or ad-hoc daemon calls), we will:

- make `rn` inconsistent with the desktop UI,
- break remote-daemon readiness,
- and make AI-first testability harder.

## Goal

Implement agent orchestration as **control-plane-owned business logic** exposed over the client API:

- `StartAgent`
- `StopAgent`
- `RestartAgent`
- `SendTaskAgentMessage` (structured + interactive)
- `AttachAgentSession` (where supported)

The control plane persists command lifecycles (T-19) and routes execution to the daemon via the daemon stream protocol (T-11), while persisting session events (T-40).

## Requirements

### 1) Define command kinds + payloads

Define typed command payloads for daemon execution, including:

- start session (structured vs interactive)
- stop session
- interrupt session/turn
- send message (structured vs interactive)
- resume-by-id turn (structured)

Constraints:

- command payloads are repo-scoped and contain only stable ids (no paths).
- command results must be representable as durable session events + command lifecycle updates.

### 2) Preserve “send message” semantics (port behavior)

Port the current semantics precisely:

- Determine desired interface mode from configured harness (or explicit preference).
- Structured mode:
  - If a resumable structured session exists (external session ref present), send via resume-by-id.
  - If a structured turn is in progress for that resumable session:
    - on_conflict=fail → 409 conflict (“turn in progress”)
    - on_conflict=interrupt_turn → interrupt then resume-by-id
  - If no resumable session exists and any session is active:
    - on_conflict=interrupt_turn → 400 (cannot interrupt without resumable session)
    - on_conflict=fail → 409 conflict (“session conflict”)
    - on_conflict=stop_session_and_start_new → stop then start new structured session
- Interactive mode:
  - If an interactive session is active: send keystrokes (optional interrupt).
  - If an incompatible session is active: conflict unless stop-and-start-new.

Return a typed response including:

- delivery kind (structured_started/structured_resumed/interactive_started/interactive_sent),
- conversation continuity (kept/broken),
- warnings (tuple/list of strings).

Error reporting:

- Conflicts must be structured (T-9 error category `conflict`) and include stable conflict codes so the UI can offer confirm flows.

### 3) “No silent actions” guarantee

- All commands must create a `command_id` and immediately become observable via:
  - command lifecycle state, and
  - session event subscriptions.
- Clients must be able to show “Sending…”/“Starting…” immediately and prevent duplicate requests.

### 4) Session event persistence

Ensure that, as part of these flows:

- user messages and assistant messages become durable session events (T-14),
- and are persisted (T-40),
- while delta/chunk events are not persisted.

### 5) AI-first testability hooks

The implementation must support deterministic tests:

- waiting primitives (T-15) must work for:
  - command terminal states, and
  - “session event X observed” conditions.

## Acceptance criteria

- The control plane exposes agent command APIs over the client protocol (T-12).
- A deterministic integration test can:
  1) create a task,
  2) start a structured session,
  3) send a message and observe either:
     - resume-by-id, or
     - conflict behavior,
  4) interrupt when requested,
  5) and query back durable session history from the DB.
- Conflict codes and behaviors match the current Python semantics closely enough that existing UI confirm flows can be reproduced without inventing new logic.

## Dependencies / sequencing

- Depends on command engine (T-19) and client API surface (T-12).
- Depends on daemon stream protocol (T-11) and session persistence (T-40).
- Uses kind/mode/resume-by-id utilities (T-32).

## Reference implementation (today; for behavior orientation only)

- Core behavior (Python today):
  - `redesmyn/task_agent_messaging.py` (authoritative send-message semantics).
  - `redesmyn/agent_turn_transport.py` (resume-by-id argv builder).
  - `redesmyn/agent_runtime.py` (start/stop/send text; resume-by-id turn runner).
- API surface (Python today):
  - `redesmyn/api.py` (`POST /v1/tasks/{task_id}/agent/message`).
- UI confirm flows (TS today):
  - `dashboard/src/components/graph/TaskCard.tsx` (conflict prompts, pending “Sending…” state).

