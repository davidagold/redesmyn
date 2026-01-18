---
epic: gpui
branch:
  suggested: rn/gpui/T-14-artifacts-and-session-events-contract
rn:
  parent: T-10
---

# T-14 Artifact references + structured session events (contract) (Domain 1)

## Problem

We want to persist agent/session output in a structured, performant-to-query format that enables delightful UI/UX.

We also want to avoid shoving large blobs (full logs, full diffs) into the core protocol stream or event log.

If we don’t define a clear “what is an event vs what is an artifact” contract early, we will:

- accidentally bake large payloads into events,
- end up duplicating logs in multiple places,
- and make it hard to build fast diff/session views later.

## Goal

Define the foundational contract for:

- **artifact references** (addresses to content stored elsewhere), and
- **structured session events** (durable, queryable records of agent/session activity).

This ticket defines schema and invariants; detailed agent/runtime implementation lands in later domains.

## Requirements

### 1) Artifact reference model

Define an `ArtifactRef` type with:

- `artifact_id: ArtifactId` (ULID)
- `kind: ArtifactKind` (small enum; e.g. log, diff, patch, file_snapshot, trace)
- `content_hash: Option<Hash>` (for integrity/dedup; algorithm specified)
- `byte_len: Option<u64>`
- `mime: Option<String>`
- `storage_hint: Option<StorageHint>` (e.g. local path, blob store key, etc.)

Rules:

- Protocol messages and durable events should reference artifacts via `ArtifactRef`, not embed full content.
- Artifact storage backends are an implementation detail; the ref is stable.

### 2) Structured session event baseline

Define a `SessionEvent` union with (at minimum) these categories:

- `SessionStarted` / `SessionEnded`
- `TurnStarted` / `TurnCompleted` (turn-as-event; see “Session semantics” below)
- `UserMessage` / `AssistantMessage` (chat-like events; include short preview fields)
- `ToolInvocation` / `ToolResult` (structured tool events)
- `StatusUpdate` (turn state, blocking, progress)
- `ArtifactEmitted` (ties artifacts to a session/scope)

Rules:

- Session events must be queryable by:
  - `session_id`, time range, and event kind, and
  - **scope** (task-scoped sessions and user-managed chat sessions are both supported).
- Keep payloads compact; link to artifacts for large content.
- Preserve forward compatibility via an `UnknownSessionEvent` fallback.

### 2.1) Session semantics: session == conversation; turns are events

We want the UI/API to treat `session_id` as a **conversation id**:

- in interactive mode (tmux), and
- in structured mode **at least for Codex** (and ideally for other structured agents too, if they expose a resumable conversation handle).

That means (for agents that support it):

- a structured “send message” creates a **new turn**, not a new session,
- turn lifecycle is represented by events (`TurnStarted` / `TurnCompleted`),
- and the session viewer can render a correct “timeline” without reconstructing conversations from multiple session rows.

This aligns with (and should stay consistent with) the v0 design note:

- `epics/harness-interface-v0/tasks/T-15/README.md`: Session semantics: `AgentSession` == conversation, turns are events (event-as-turn).

Notes for Claude Code (CC):

- Claude Code “print mode” (`--output-format stream-json`) appears to expose a stable `session_id` and supports `--resume`, so we *expect* this same model to work for CC.
- If CC cannot reliably provide resumable conversation ids in practice, we can fall back to “new conversation per message” **for CC only**, but the event contract and session viewer must still support turn boundaries and durable message history.

Minimum event payloads (names illustrative; exact fields belong in `.proto`):

- `TurnStarted`
  - `session_id`
  - `turn_id: Option<String>` (best-effort; stable within an external harness when available)
  - `interface_mode`
  - `external_session_ref` (when structured; best-effort)
  - optional: `idempotency_key`, `log_offset_bytes`
- `TurnCompleted`
  - `session_id`
  - `turn_id: Option<String>`
  - `interface_mode`
  - `external_session_ref` (best-effort)
  - optional: `exit_code` / structured error
- `UserMessage` / `AssistantMessage`
  - `session_id`
  - `turn_id: Option<String>` (null allowed)
  - message body (bounded) + preview fields

### 3) Event log vs session events vs telemetry

Clarify storage intent:

- The control plane event log is authoritative for orchestration events and UI sync.
- Session events are durable, structured records primarily for session viewer UX.
- Raw agent logs may exist (tmux logs, harness logs) but are not duplicated wholesale; instead we store structured events + artifact refs.

### 4) Implications for protocol

Daemon → control plane should be able to emit:

- “small” structured session events on the stream, and/or
- artifact refs that can be fetched out-of-band.

Client subscriptions should be able to stream:

- orchestration events and command lifecycle updates, and
- session events for active sessions (with backpressure/limits).

## Acceptance criteria

- `.proto` definitions exist for `ArtifactRef` and the `SessionEvent` baseline union, including forward-compatible “unknown” variants.
- The contract explicitly forbids embedding large blobs in durable events.
- The schema is sufficient to design the native session viewer and diff viewer without inventing new primitives.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-10/README.md` (schema/codegen).
- Informs Domain 2 (storage schema), Domain 4 (agent runtime), and Domain 7/8 (session viewer + diff systems).

## Reference implementation (today; sessions + messaging)

- Agent/session runtime (Python today):
  - `redesmyn/agent_driver.py` (agent session lifecycle; emits assistant message events and updates).
  - `redesmyn/agent_turn_transport.py` (structured turn transport for agent interfaces).
  - `redesmyn/task_agent_messaging.py` (repo-scoped “send message” behavior and conflict handling).
- Control plane API (Python today):
  - `redesmyn/api.py` (`POST /v1/tasks/{task_id}/agent/message`).
- Dashboard UI (TS today):
  - `dashboard/src/components/graph/TaskCard.tsx` (message composer; “no silent actions” pending states).
  - `dashboard/src/hooks/useEventStream.ts` (`agent.assistant_message` and turn events in the stream).
