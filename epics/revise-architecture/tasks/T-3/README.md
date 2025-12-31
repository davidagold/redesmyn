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
  - acts as the **repo executor** for local repos by executing git/worktree mutations on behalf of the control plane’s high-level intent:
    - merge/restack plans (including resumable conflict handling)
    - git-derived projections needed by the UI (e.g. “stack in sync with upstream”)
- Refactor the current repo observer to support a “sink” abstraction:
  - **default** sink: emit events to the control plane API/WS (even in co-located local dev)
  - debug-only sink: write directly to a local DB (explicit flag; compatibility only)
- Ensure robust lifecycle:
  - reconnect/backoff
  - resync on reconnect
  - bounded polling + timeouts
- Remove epic-scoped observation as a primary mode: one daemon per host, at most one observation loop per repo (epic scoping only as a debug/perf option, if at all).
  - when multiple daemons could attach to the same `workspace_id + repo_id`, participate in an explicit “primary executor” lease so git-mutating commands have a single writer

## Acceptance Criteria

- `rn daemon run` can connect to a remote control plane and stream telemetry.
- In normal operation, the daemon never writes to the control plane DB directly; all telemetry and state mutations flow through API/WS.
- The control plane can issue a minimal command and the daemon executes it and reports status/events back.
- “Observer” is no longer required as a separate user-facing process in the cloud model (it becomes part of the daemon).

## Updates

### 2025-12-31

- Recent work added `stackInSync`/“out-of-sync” surfacing in the dashboard, but it is currently computed by the control plane via direct git.
  This task must instead make the daemon emit that projection (event/snapshot) as part of its git/worktree telemetry.
- **Coordination note:** The updates in **T-2 + T-3 + T-7** must be considered in concert to enable server-driven merges:
  the daemon should be able to receive a merge intent, execute it locally (including conflict → resume), and emit progress/results
  for the control plane to persist and broadcast.
