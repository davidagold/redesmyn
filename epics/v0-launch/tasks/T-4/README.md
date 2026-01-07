# T-4 Daemon agent lifecycle: start/stop/restart via protocol

## Metadata

```yaml
id: T-4
stacked_on: T-3
node:
  branch: rn/v0-launch/T-4-daemon-agent-lifecycle
```

## Brief (local)

Make the daemon the sole owner of agent lifecycle:

- Add daemon command types for:
  - `agent.start` (task-scoped)
  - `agent.stop`
  - `agent.restart`
  - (optional v0) `agent.logs`/`agent.attach` guidance payloads
- Implement the corresponding handlers in the daemon runtime using existing `agent_runtime` primitives.
- Replace the control plane’s direct process management with a runner backend that sends commands to the daemon and waits for ack/result.

## Acceptance Criteria

- Starting/stopping/restarting agents works with the server running in pure mode (no local execution).
- Results are persisted in the DB and streamed to the dashboard as events/updates.
- Failure cases are explicit and actionable:
  - daemon not connected
  - daemon connected but repo not attached
  - requested host is not primary executor (when relevant)

