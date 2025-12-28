# T-2 Server: daemon connection + presence + command delivery

## Metadata

```yaml
id: T-2
stacked_on: T-1
node:
  branch: rn/revise-architecture/T-2-server-daemon-connection
```

## Brief (local)

- Add a server endpoint for daemon connectivity (WebSocket recommended).
- Persist/track daemon presence:
  - daemon host identity + capabilities
  - last_seen + connection status
- Implement command delivery primitives:
  - send commands to a connected daemon
  - daemon acks + retries (idempotency)
- Accept daemon-emitted events and persist them into the control plane event log so the existing UI WebSocket stream can broadcast updates.

## Acceptance Criteria

- A daemon can connect, authenticate, and be visible as “online” in the control plane.
- A basic “ping → pong” + heartbeat updates daemon `last_seen`.
- A daemon-emitted event is persisted and observed by the dashboard via the existing `/v1/ws` stream.
