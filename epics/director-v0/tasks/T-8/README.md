---
rn:
  node:
    branch: rn/director-v0/T-8-task-agent-messaging
  parent: null
---

# T-8 Task agent messaging + history (rn surfaces)

## Problem

Director orchestration frequently needs to do more than start/merge:

- request changes after review,
- ask a task agent to clarify status, rerun a check, or make a targeted edit,
- and ground decisions in the task session transcript without bloating controller wake payloads.

Without a durable, control-plane-routed messaging/read surface, v0 devolves into manual relay by the
conductor and makes director runs harder to debug and reproduce.

## Goal

Add minimal `rn` CLI surfaces that let the director (and humans) interact with task agents through the
control plane:

- send a message to a task agent with optional intent metadata, and
- read task session history (transcript) deterministically.

These surfaces must use the same control-plane request paths as the desktop UI so results are durable
and visible without refresh.

## Requirements

### 1) Send message

Add `rn task send`:

- Inputs:
  - `--epic <slug>` (default only if exactly one local epic exists)
  - `--task <T-n>` (local ref)
  - `--message <text>` (required)
  - `--intent <value>` (optional; default `unspecified`)
  - optional conflict/interrupt flags if needed (keep minimal in v0).
- Semantics:
  - routes through control plane (no direct daemon calls from CLI).
  - delivery semantics follow existing control-plane planner:
    - resume structured session if resumable,
    - else start a new session (with normal conflict policy).
  - message is persisted as a `SessionEvent` and any resulting command is visible in command history.

Notes:

- `--intent` is metadata for the director/control plane/UI to interpret; it should not silently mutate
  queue state by itself.
- Director-level actions like `request_changes` should be modeled explicitly in director code/docs and
  implemented using `rn task send --intent request_changes` plus any separate queue/task state updates
  required by the director run semantics.

### 2) Read history

Add `rn task history`:

- Resolves the task’s latest session via control plane (`GetLatestTaskSession`).
- Fetches session events via control plane (`GetSessionEvents`) with a bounded, paginated read.
- Default output is a transcript-friendly subset:
  - `user_message`, `assistant_message`, `status_update`.
- Options:
  - `--limit N`
  - `--all` (includes tool/reasoning events) or equivalent explicit flags.

The intent is to let the director pull history on-demand for review/request-changes without forcing the
controller to include large transcripts in wake payloads.

## Acceptance Criteria

- `rn task send` delivers a message through control plane and the message appears in the task session
  history as a durable `SessionEvent`.
- `rn task history` prints a deterministic transcript view and supports bounded pagination.
- Both commands are observable:
  - command lifecycle is persisted and reported,
  - failures include actionable error messages.

