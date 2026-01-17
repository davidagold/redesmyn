---
epic: gpui
branch:
  suggested: rn/gpui/T-20-client-api-server
rn:
  parent: T-18
---

# T-20 Client API server over UDS (requests + subscriptions) (Domain 2)

## Problem

Clients (`rn`, GPUI UI) need a fast, local, testable way to interact with the control plane.

We’ve chosen:

- a client protocol over Unix domain socket (T-12),
- multiplexed requests,
- and streaming subscriptions for realtime updates.

If we default to HTTP+JSON, we risk:

- unnecessary overhead,
- duplicated API surfaces,
- and brittle GUI testing.

## Goal

Implement the control plane’s client-facing API server over UDS, including:

- request/response handling,
- streaming subscriptions (event log, command updates),
- and the initial method set needed for early integration and AI testing.

## Requirements

### 1) UDS server implementation

- Bind to configured UDS path with safe permissions.
- Support multiple concurrent clients.
- Multiplex multiple in-flight requests per connection.

### 2) Initial method set

Implement at least:

- `Health/Status`
- `ListEpics`
- `GetEpicGraph` (can start as minimal schema; must be stable and typed)
- `SubscribeEventLog`
- `WaitForCommand` (T-15)

### 3) Client library

Create a small Rust client crate/module used by:

- `rn` (Rust CLI),
- the desktop UI when connecting out-of-proc,
- and integration tests.

The client library should:

- hide framing/codec details,
- expose typed request methods,
- support subscriptions and wait primitives.

### 4) Error handling

- Use the structured error model from T-3/T-9.
- Do not leak internal stack traces.
- Errors must be actionable for CLI users and renderable in UI.

### 5) Testability

- Provide integration tests that start a control plane instance, connect via UDS, and perform:
  - request/response,
  - subscription receive,
  - wait primitive success and timeout.

## Acceptance criteria

- A local client can talk to the control plane over UDS and perform the initial method set.
- Subscriptions are stable and do not require polling.
- Tests are deterministic and do not use `sleep` for correctness.

## Dependencies / sequencing

- Depends on Domain 1 client protocol contract (T-12, T-15) and envelope/schema (T-9/T-10).
- Depends on Domain 2 event hub (T-18) and DB schema (T-17).

