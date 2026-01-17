---
epic: gpui
branch:
  suggested: rn/gpui/T-19-command-engine
rn:
  parent: T-18
---

# T-19 Command engine (persisted lifecycle + routing to daemon) (Domain 2)

## Problem

We have a strict UX requirement:

- **no silent actions**: every user-visible mutation must show progress and resolve to a clear outcome.

Architecturally:

- the control plane must be authoritative about command intent and lifecycle,
- the daemon executes repo-local commands and reports progress/acks back,
- and clients (`rn`, UI) must be able to trigger commands and wait for outcomes.

If commands are “just RPC calls”, we lose auditability and make retries/resync brittle.

## Goal

Implement the control-plane command system:

- commands are durable DB objects,
- command state transitions are append-only updates,
- routing to daemons happens through the daemon transport (T-11),
- and clients can observe command state via subscriptions and wait primitives.

## Requirements

### 1) Command model

Define a command model that includes:

- `command_id` (ULID)
- `scope` (repo scope for repo-local actions)
- `kind` (typed command kind)
- `idempotency_key` (optional, but supported for clients)
- `created_by` (client id/session id; optional)

Persist:

- the initial command request (compact payload),
- and append-only state updates.

### 2) State machine

Implement a small, explicit state machine aligned with Domain 1:

- `queued` / `accepted` / `running` / `blocked` / `resumable` / `succeeded` / `failed` / `canceled`

Rules:

- state transitions are validated (no illegal jumps).
- the daemon can only update states for commands it is executing (routing/authorization via connection identity).

### 3) Routing to daemon

When a repo-scoped command is issued:

- the control plane selects the appropriate daemon/executor (lease/primary rules live in Domain 3, but the control plane must have the seam).
- dispatches a `CommandDispatch` message (T-11) over the daemon stream.

### 4) Receiving updates

Daemon command updates are:

- persisted to `command_updates`,
- emitted as domain events (T-18),
- and delivered to subscribed clients.

### 5) Wait primitives (AI-first)

Implement server-side support for:

- `WaitForCommand` and `WaitForIdle` primitives (T-15),

so tests and `rn` can wait without polling.

## Acceptance criteria

- Commands are durable and observable end-to-end:
  - create command → dispatch → updates → terminal outcome.
- Client can issue a command and wait for completion deterministically (no sleeps).
- The control plane never executes repo-local actions directly; it only dispatches.

## Dependencies / sequencing

- Depends on Domain 1 daemon stream contract (T-11) and client contract (T-12/T-15).
- Depends on Domain 2 storage and event hub (T-17, T-18).

