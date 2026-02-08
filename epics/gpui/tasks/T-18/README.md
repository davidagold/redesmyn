---
epic: gpui
branch:
  suggested: rn/gpui/T-18-event-log-and-subscriptions
rn:
  node:
    branch: rn/gpui/T-18-event-log-and-subscriptions
  parent: T-17
---

# T-18 Event log append + subscription hub (Domain 2)

## Problem

Redesmyn relies on realtime updates:

- UI must reflect daemon telemetry and command lifecycle changes quickly.
- CLI must be able to “wait until X” without polling or sleeps.

We also want an auditable history:

- an append-only event log for “what happened?”.

If event emission/publishing is bolted on ad-hoc, we risk:

- missed updates,
- expensive polling,
- and inconsistent semantics between UI and CLI.

## Goal

Implement the control-plane event pipeline:

- append compact domain events to the DB,
- publish them to subscribers in-process,
- and serve them over the client API subscriptions (T-12) in a deterministic, test-friendly way.

## Requirements

### 1) Event append API (single source of truth)

Expose an internal API like:

- `append_event(scope, event) -> EventId`

Rules:

- Persist first, then publish (so subscribers can resync by querying the DB).
- Events are compact; large data is referenced via artifacts (T-14).
- Unknown event types are supported (forward compatibility).
- Prefer relying on DB-level constraints for scope consistency (see T-17) so the append path cannot
  create “impossible” scoped rows (e.g. mismatched workspace/repo ids).

### 2) Subscription hub

Implement an in-process subscription hub with:

- backpressure (bounded queues),
- drop policy (explicit, documented),
- and a resync signal when a client falls behind.

This hub is used by:

- client API subscriptions (UDS),
- and internal “wait until” primitives (T-15).

### 3) Cursoring/resume

Support cursor-based resume:

- clients can subscribe “from event id N”.
- control plane can detect gaps and instruct resync when needed.

### 4) Deterministic testing hooks

Provide a test harness that can:

- start the control plane,
- subscribe to events,
- append events,
- and deterministically assert ordering/receipt without sleeps.

## Acceptance criteria

- There is a single internal append path used by all event emission.
- Subscriptions work in-process and over the client API with cursor resume.
- Tests can “wait for event” deterministically using event-driven primitives.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-17/README.md` (events table + schema).
- Builds on Domain 1 envelope/subscription contract (T-9, T-12, T-15).

## Reference implementation (today; event pipeline orientation only)

- Event persistence + broadcast (Python today):
  - `redesmyn/db/models.py` (`Event` table).
  - `redesmyn/api.py` (`_append_event`, `_broadcast_event`).
- Realtime event stream to UI (Python today):
  - `redesmyn/event_stream.py` (WebSocket stream protocol: hello/event/error/resync/pong).
  - `redesmyn/ws_runtime.py` (`JsonWebSocketHub` fanout + sender loop).
  - `redesmyn/api.py` (`@v1.websocket("/ws")`).
- Dashboard consumption (TS today):
  - `dashboard/src/hooks/useEventStream.ts`
  - `dashboard/src/api/useEpicCacheSync.ts`
- Tests (Python today):
  - `tests/test_api_integration.py` (event payloads are JSON-serializable).
  - `tests/test_daemon_ws_runtime_integration.py` (broadcast path and WS payload shape).
