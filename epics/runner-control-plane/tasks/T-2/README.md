# T-2 Server: runner connection + presence + command delivery

## Metadata

```yaml
id: T-2
stacked_on: T-1
node:
  branch: rn/runner-control-plane/T-2-server-runner-connection
```

## Brief (local)

- Add a server endpoint for runner connectivity (WebSocket recommended).
- Persist/track runner presence:
  - runner host identity + capabilities
  - last_seen + connection status
- Implement command delivery primitives:
  - send commands to a connected runner
  - runner acks + retries (idempotency)
- Accept runner-emitted events and persist them into the control plane event log so the existing UI WebSocket stream can broadcast updates.

## Acceptance Criteria

- A runner can connect, authenticate, and be visible as “online” in the control plane.
- A basic “ping → pong” + heartbeat updates runner `last_seen`.
- A runner-emitted event is persisted and observed by the dashboard via the existing `/v1/ws` stream.
