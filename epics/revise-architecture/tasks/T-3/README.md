# T-3 Daemon: connect + telemetry + orchestration loop

## Metadata

```yaml
id: T-3
stacked_on: T-2
node:
  branch: rn/revise-architecture/T-3-daemon
```

## Brief (local)

- Implement a long-lived daemon process that:
  - maintains an outbound connection to the control plane
  - emits telemetry/events (git commits, worktree health, session status)
  - receives commands (start/stop/restart sessions, assignment changes) and executes them locally
- Refactor the current repo observer to support a “sink” abstraction:
  - **default** sink: emit events to the control plane API/WS (even in co-located local dev)
  - debug-only sink: write directly to a local DB (explicit flag; compatibility only)
- Ensure robust lifecycle:
  - reconnect/backoff
  - resync on reconnect
  - bounded polling + timeouts
- Remove epic-scoped observation as a primary mode: one daemon per repo (epic scoping only as a debug/perf option, if at all).

## Acceptance Criteria

- `rn daemon run` can connect to a remote control plane and stream telemetry.
- In normal operation, the daemon never writes to the control plane DB directly; all telemetry and state mutations flow through API/WS.
- The control plane can issue a minimal command and the daemon executes it and reports status/events back.
- “Observer” is no longer required as a separate user-facing process in the cloud model (it becomes part of the daemon).
