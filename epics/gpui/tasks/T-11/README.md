---
epic: gpui
branch:
  suggested: rn/gpui/T-11-daemon-stream-protocol
rn:
  parent: T-10
---

# T-11 Daemon ↔ control plane stream protocol (Domain 1)

## Problem

The daemon/control-plane boundary is the primary architectural split.

We need a stream protocol that works for:

- embedded daemon (in-proc transport),
- future remote daemon (network transport),
- reconnection and resync,
- and “no silent actions” UX via explicit command lifecycle updates.

Without a well-specified stream protocol, we risk a fragile, special-cased “local mode” again.

## Goal

Define the daemon ↔ control plane stream protocol message set and semantics, including:

- handshake and capabilities,
- repo attachment and routing,
- telemetry/event emission patterns,
- command dispatch + lifecycle reporting,
- and reconnect/resync behavior.

This contract is independent of the concrete transport:

- remote mode: framed network transport with Protobuf codec,
- embedded mode: in-proc transport passing typed messages (optionally with codec loopback for embedded dev/tests).

## Requirements

### 0) Canonical message schema

- All message types for this protocol live in `.proto` and generate canonical Rust types in `rust/crates/redesmyn_protocol`.
- Both daemon and control plane implementations depend on those typed messages via the transport trait (T-7), not bespoke JSON.

Framing rule:

- All stream traffic is carried as `DaemonFrame { envelope: ProtocolEnvelope, message: oneof ... }` (see `rust/proto/daemon.proto`).
  - `envelope.msg_id` is the cross-boundary dedupe key (at-least-once delivery).
  - Repo-scoped messages must set `envelope.scope = RepoScope`.
  - Protocol-level errors may be sent as an `ErrorEnvelope` frame payload (T-9), then the connection is closed.

### 1) Handshake

Define a handshake sequence:

1. Daemon connects (outbound) to control plane.
2. Daemon sends `DaemonHello` including:
   - `host_id` (stable identity)
   - `host_instance_id` (ephemeral process identity)
   - `capabilities` (explicit list)
   - supported `protocol_major/minor`
3. Control plane responds with `ControlPlaneHelloAck` including:
   - accepted protocol version
   - optional server capabilities / configuration

Rules:

- Major mismatch must be rejected with a structured error (T-9).
- Minor mismatch may be accepted; unknown fields must not break.

### 2) Repo attachment

Define explicit repo-scoped routing:

- Daemon may send `RepoAttach { repo_scope, repo_root_hint? }` (the hint is local-only and must not be required by the control plane).
- Control plane may request `RepoAttach`/`RepoDetach` by stable repo identity (workspace_id + repo_id), never by filesystem path.

### 3) Presence / heartbeats

- Daemon periodically sends `DaemonHeartbeat` with:
  - current attached repo scopes,
  - optional per-scope “telemetry freshness” markers.

### 4) Telemetry and events

We need a policy that balances correctness with simplicity:

- Events are delivered at-least-once; receivers dedupe best-effort by `msg_id`.
- For loss recovery, use explicit resync:
  - control plane may request a snapshot for a scope,
  - daemon can emit snapshots periodically or on request.

Define message types:

- `TelemetryEventBatch { scope: RepoScope, events: [...] }`
- `TelemetrySnapshot { scope: RepoScope, snapshot: ... }` (schema may start minimal)

Event typing:

- Prefer a typed union for known event kinds (git/worktree/agent/merge-run), plus:
  - `UnknownEvent { event_type: string, json_payload: bytes/string }`

### 5) Command dispatch + lifecycle

Control plane sends:

- `CommandDispatch { command_id, scope, command_kind, payload }`

Daemon responds with lifecycle updates:

- `CommandUpdate { command_id, state, message?, progress?, detail? }`

Rules:

- Daemon must be able to reject commands it cannot execute (not attached, not primary, invalid).
- Command updates must be sufficient for UI/CLI to show “in progress” and final outcome (no silent actions).
- Command IDs are ULIDs and must be stable for dedupe/retry.

### 6) Resync

Define a minimal resync mechanism:

- Control plane can request resync: `ResyncRequest { scope, reason }`.
- Daemon responds with `TelemetrySnapshot` and/or replays recent events as appropriate.

Do not over-design sequencing/acks in v1:

- Start with message-level idempotency + snapshot-based recovery.
- Add per-scope sequence/acks only if we prove snapshots are insufficient.

## Acceptance criteria

- `.proto` message definitions exist for the handshake, attachment, telemetry, commands, and resync.
- Semantics are documented clearly enough that:
  - daemon implementers and control-plane implementers can build independently and interoperate.
- The design keeps the control plane free of repo filesystem assumptions.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-10/README.md` (protobuf schema pipeline).
- Informs later implementation work in Domain 2 (control plane core) and Domain 3 (daemon core).

## Reference implementation (today; daemon stream behavior)

- Daemon stream (Python today):
  - `redesmyn/api.py` (`/v1/daemon/ws` websocket handler `daemon_ws()`).
  - `redesmyn/ws_runtime.py` (`DaemonConnectionRegistry`; routing/broadcast mechanics).
  - `redesmyn/ws_protocol.py` (message types exchanged today).
- Tests (Python today):
  - `tests/test_daemon_ws_runtime_integration.py`
