---
rn:
  node:
    branch: rn/agent-orchestration/T-6-websocket-stream
  parent: T-4
---

# T-6 WebSocket: live graph updates (activity + presence)

## Brief (local)

- Add a WebSocket endpoint that streams:
  - agent/session status changes
  - new commits / ref movements
- Define (but do not implement here) extensibility points for message/command events (handled in `epics/messages-commands/README.md`).
- Define subscription scoping (at least by repo/epic) and backpressure behavior.
- Keep the UI simple initially (consume events to refresh projections), with an upgrade path to incremental updates.

## Acceptance Criteria

- The dashboard no longer relies on manual refresh for core activity signals.
- WebSocket contract supports future bidirectional chat without redesign.

## Updates

- Added `/v1/ws` WebSocket stream (server → client) that emits DB `events` as `{type:"event", event:{…}}` and supports `ping`/`subscribe`.
- Dashboard connects per-epic and auto-refreshes the graph on streamed events (debounced).
