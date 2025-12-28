# T-3 Runner daemon: connect + telemetry + orchestration loop

## Metadata

```yaml
id: T-3
stacked_on: T-2
node:
  branch: rn/runner-control-plane/T-3-runner-daemon
```

## Brief (local)

- Implement a long-lived runner process that:
  - maintains an outbound connection to the control plane
  - emits telemetry/events (git commits, worktree health, session status)
  - receives commands (start/stop/restart sessions, assignment changes) and executes them locally
- Refactor the current repo observer to support a “sink” abstraction:
  - local dev sink: write directly to a local DB (optional; compatibility)
  - cloud sink: emit events to the control plane API/WS
- Ensure robust lifecycle:
  - reconnect/backoff
  - resync on reconnect
  - bounded polling + timeouts

## Acceptance Criteria

- `rn runner run` can connect to a remote control plane and stream telemetry.
- The control plane can issue a minimal command and the runner executes it and reports status/events back.
- “Observer” is no longer required as a separate user-facing process in the cloud model (it becomes part of the runner).
