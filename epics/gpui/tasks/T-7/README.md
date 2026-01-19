---
epic: gpui
branch:
  suggested: rn/gpui/T-7-transport-codecs
rn:
  parent: T-2
---

# T-7 Transport + codec scaffolding (Protobuf default + JSON debug) (Domain 0)

## Problem

We are embedding control plane + daemon in a single desktop process *without* allowing colocated assumptions.

We also want long-term performance and throughput for:

- high-frequency telemetry,
- exec/app-server style agent events,
- future remote daemon support.

If we delay transport/codec design, we risk:

- leaking daemon internals into control-plane call sites,
- cementing an inefficient JSON-only protocol,
- making remote-daemon support a painful retrofit.

## Goal

Introduce a typed transport abstraction and codec plan that supports:

- in-proc embedding (no serialization required),
- remote daemon connections (binary codec by default),
- JSON as an opt-in dev/diagnostic mode with good tooling.

## Requirements

### 0) Canonical protocol types

Define a single canonical cross-boundary message schema:

- Rust types live in `rust/crates/redesmyn_protocol` (generated from `.proto` in Domain 1).
- Transports move typed protocol messages (envelopes + payloads), not ad-hoc JSON dicts.

### 1) Transport traits

Define typed traits for communication such as:

- control plane → daemon: command dispatch
- daemon → control plane: events, presence/heartbeats, command acks

Key invariants:

- control plane code must not call daemon internals directly.
- embedded daemon must be reachable only through the same trait used for remote daemons.

### 2) Codecs

Implement two codecs over the same typed message structures:

- Protobuf codec (default for network)
- JSON codec (debug/diagnostic)

Notes:

- We optimize for performance long-term but keep developer observability.
- We should avoid putting giant blobs (e.g., full diffs) in protocol messages; use artifact references/streaming instead.

### 3) In-proc transport

Provide an in-proc transport implementation suitable for the desktop app:

- typed channels (no encoding),
- a clean lifecycle (start/stop, backpressure),
- testable in isolation.

Also support an optional “codec loopback” mode:

- send path: encode via Protobuf/JSON → decode back into typed messages → deliver,
- so embedded dev/tests can exercise the on-wire codec without requiring a network daemon.

### 4) Wiretap/observability hooks

Add minimal tooling hooks:

- ability to decode protocol frames to JSON for inspection,
- stable tracing spans per message (`msg_id`, `command_id`, `run_id`, etc.).

## Acceptance criteria

- A loopback integration test sends a typed “ping/hello” roundtrip via:
  - in-proc transport, and
  - a framed transport using both codecs (protobuf + json).
- Versioning/idempotency fields exist in the message envelope.
- Control-plane and daemon crates can depend on the transport crate without violating boundaries.

## Resolved decisions

- Default network codec: **Protobuf**.
- Debug codec: **JSON** (opt-in).
- In-proc: typed messages, no serialization.
  - Optional: codec loopback mode for embedded dev/tests.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-1/README.md` (workspace + crate skeletons).
- Depends on `epics/gpui/tasks/T-2/README.md` (ULID/newtypes) for `msg_id`/ids in envelopes.
- Should coordinate with `epics/gpui/tasks/T-4/README.md` (message tracing + wiretap spans).

## Reference implementation (today; transports)

- UI event stream (Python + TS today):
  - `redesmyn/event_stream.py` + `redesmyn/ws_runtime.py` (server-side WS runtime for `/v1/ws`).
  - `redesmyn/ws_protocol.py` (JSON message schema).
  - `dashboard/src/hooks/useEventStream.ts` (client-side WS protocol/types).
- Daemon ↔ control plane (Python today):
  - `redesmyn/api.py` (`/v1/daemon/ws` websocket handler).
  - `tests/test_daemon_ws_runtime_integration.py` (integration behavior expectations).
