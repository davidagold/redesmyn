---
epic: gpui
branch:
  suggested: rn/gpui/T-12-client-api-over-uds
rn:
  parent: T-10
---

# T-12 Client ↔ control plane API protocol over Unix socket (Domain 1)

## Problem

We want:

- a headless control plane service (desktop UI is just one client),
- a very fast `rn` CLI,
- and a clean separation from the daemon.

If we treat the “GUI as the control plane” or let `rn` bypass the control plane, we reintroduce split-brain state and make remote daemons harder.

## Goal

Define a client ↔ control plane API protocol that:

- runs over a **local Unix domain socket** for low overhead,
- can later be exposed over TCP/TLS if needed without redesigning semantics,
- supports request/response queries and streaming subscriptions,
- and uses the same envelope/versioning rules as the rest of the system (T-9).

## Requirements

### 1) Transport

- Control plane hosts a UDS server endpoint (path defined by config; secure permissions).
- Clients (`rn`, desktop UI when not embedded) connect via UDS.

No “direct daemon” path is required.

### 2) API message model

Define message types (in `client.proto`) for:

- `Request { request_id, method, payload }`
- `Response { request_id, status, payload | error }`
- `Subscribe { subscription_id, topic, filter }`
- `Event { subscription_id, event }`
- `Unsubscribe { subscription_id }`

Rules:

- `request_id` and `subscription_id` are ULIDs.
- The protocol must support multiple outstanding requests (multiplexing) without requiring multiple sockets.

### 3) Initial method set (minimal, but real)

Define a minimal set of methods sufficient for early integration testing and tooling:

- `Health` / `Status` (returns basic control-plane state)
- `ListEpics`
- `GetEpicGraph` (schema can be stubbed initially; the contract is what matters)
- `SubscribeEventLog` (stream new events)

We do not need to define every feature yet, but we must set the pattern.

### 4) Authentication / authorization (local-only v0)

For local UDS:

- rely on filesystem permissions for v0, and
- keep a clear seam for adding auth tokens later if we expose TCP.

### 5) Diagnostics

- Allow forcing JSON codec for the client API in debug mode (optional).
- Ensure request/response correlation is easy to trace (`trace_id`/`request_id` in spans).

## Acceptance criteria

- `.proto` definitions exist for the client API transport and the initial method set.
- The protocol supports multiplexed requests and streaming event subscriptions.
- The design is compatible with:
  - embedded in-proc clients (no serialization), and
  - out-of-proc clients over UDS (protobuf by default).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-10/README.md` (protobuf schema pipeline).
- Informs Domain 2 (control plane API implementation) and Domain 8 (CLI integration).

## Reference implementation (today; client API surfaces)

- HTTP API (Python today):
  - `redesmyn/api.py` (`/v1/*` REST endpoints; serves dashboard and websocket endpoints).
  - `openapi/openapi.json` (client-facing REST surface).
- CLI client (Python today):
  - `redesmyn/cli.py` (uses `httpx` to call the control plane HTTP API; includes local fallbacks for some ops).
- Dashboard client (TS today):
  - `dashboard/src/api.ts` + `dashboard/src/api/v1.ts` (HTTP client + generated types).
  - `dashboard/src/hooks/useEventStream.ts` (WS client for `/v1/ws`).
