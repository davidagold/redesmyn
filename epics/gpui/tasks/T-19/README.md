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

Schema note:

- Prefer DB-level constraints that keep scope columns coherent (see T-17), so command lifecycle code
  can assume scoped rows are internally consistent without re-validating on every query.

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

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on Domain 1 daemon stream contract (T-11) and client contract (T-12/T-15).
- Depends on Domain 2 storage and event hub (T-17, T-18).

## Reference implementation (today; command/lifecycle orientation only)

- Command and lifecycle persistence (Python today):
  - `redesmyn/db/models.py` (`Command` and `DaemonCommand` tables; `MergeRun` as the durable merge/restack run record).
  - `redesmyn/merge_runs.py` (merge run state updates + event emission).
- Routing to daemon (Python today):
  - `redesmyn/ws_protocol.py` (daemon ↔ server message types like `DaemonHello`, `DaemonEvent`, `DaemonCommandAck`, `ServerCommand`).
  - `redesmyn/ws_runtime.py` (`DaemonConnectionRegistry` for connection registry + send routing).
  - `redesmyn/api.py`:
    - `POST /v1/daemons/{host_key}/commands` issues a `DaemonCommand` and sends a `ServerCommand` over the WS registry.
    - `/v1/daemon/ws` consumes daemon hello/heartbeat/event/command_ack messages.
- Repo executor seam (Python today):
  - `redesmyn/repo_executor.py` (local vs “remote” executor seam; routes via `DaemonConnectionRegistry` in remote mode).
  - `redesmyn/runner_backend.py` (local runner backend vs remote stub for agent actions).
- Tests (Python today):
  - `tests/test_daemon_ws_runtime_integration.py` (command routing over daemon WS + merge.run event ingestion).
  - `tests/test_api_integration.py` (merge/restack endpoints create merge run records; “no primary executor” guidance contract).
