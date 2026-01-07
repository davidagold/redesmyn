# T-5 Daemon owns observation + agent monitoring

## Metadata

```yaml
id: T-5
stacked_on: T-4
node:
  branch: rn/v0-launch/T-5-daemon-observation
```

## Brief (local)

Move continuous repo/agent reconciliation loops out of the server:

- Repo observer loop runs in the daemon and emits normalized events to the control plane.
- Agent monitor/driver runs in the daemon and emits:
  - session status transitions
  - logs/presence/heartbeat signals as applicable
- Remove server flags/env like “run observer in server” except for explicit dev-only debug hooks (if retained at all).

## Acceptance Criteria

- With the server running alone, repo/agent state does not change unless a daemon is connected and attached.
- With the daemon connected, the dashboard receives:
  - git/worktree telemetry events
  - agent/session state updates
- There is no user-facing `rn observer` command; observation is implicit with daemon lifecycle.

