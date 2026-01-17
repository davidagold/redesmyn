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
- `UserMessage` / `AssistantMessage` (chat-like events; include short preview fields)
- `ToolInvocation` / `ToolResult` (structured tool events)
- `StatusUpdate` (turn state, blocking, progress)
- `ArtifactEmitted` (ties artifacts to a session/task)

Rules:

- Session events must be queryable by:
  - `task_id`, `session_id`, time range, and event kind.
- Keep payloads compact; link to artifacts for large content.
- Preserve forward compatibility via an `UnknownSessionEvent` fallback.

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

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-10/README.md` (schema/codegen).
- Informs Domain 2 (storage schema), Domain 4 (agent runtime), and Domain 7/8 (session viewer + diff systems).

