# T-5 Daemon owns observation + agent monitoring

## Metadata

```yaml
id: T-5
stacked_on: T-4
node:
  branch: rn/v0-launch/T-5-daemon-observation
```

## Plan

The daemon already emits repo telemetry (via `observe_repo`) but the control plane still runs monitoring loops in “local”
mode, and the CLI still exposes `rn observer`. This task makes the daemon the single source of truth for continuous
reconciliation, and ensures the control plane is just persistence + routing + UI.

### Work

1) Observation loop ownership

- Ensure `rn daemon run` always runs repo observation for attached repos (on by default).
- Remove any requirement for a separate observer process (this is a prerequisite for removing `rn observer`).

2) Agent monitoring ownership

- Move the “agent driver” loop (session supervision, status updates, log tailing metadata) into the daemon.
- Ensure status transitions are persisted/streamed so UI stays live without server-side monitoring.

3) Control plane becomes a passive receiver

- The control plane should accept and persist daemon-emitted events.
- The control plane should not attempt to infer repo state by reading git/worktree paths.

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
