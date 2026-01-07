# T-4 Daemon agent lifecycle: start/stop/restart via protocol

## Metadata

```yaml
id: T-4
stacked_on: T-1
node:
  branch: rn/v0-launch/T-4-daemon-agent-lifecycle
```

## Plan

To remove “local mode”, the daemon must become the sole owner of agent lifecycle. Today:

- The control plane can start agents directly in-process (local runner backend).
- The daemon protocol does not include agent lifecycle commands.

This task fills that gap: agent lifecycle becomes a daemon command surface, and the control plane calls it via protocol.

### Work

1) Extend the daemon protocol with agent lifecycle commands

- Define command types (names are illustrative; pick a consistent namespace):
  - `agent.start` (task-scoped)
  - `agent.stop`
  - `agent.restart`
- Define request/response payloads:
  - required: `task_id`, harness command/config, detach mode, prelude override, agent kind selection
  - response: created/updated agent session id, attach info, resolved configuration, warnings/errors
- Ensure idempotency semantics are clear enough for retries (v0 can be “best effort”, but avoid “double-start” footguns).

2) Implement daemon handlers using existing primitives

- The daemon should call existing `agent_runtime` code to:
  - start/stop/restart sessions
  - write logs/attach metadata
- The daemon should emit events (or persist state) so the dashboard updates without a server-side monitor loop.

3) Replace control-plane direct execution with a remote backend

- Implement a real “remote runner backend” that sends commands to the daemon and waits for `command_ack`.
- Remove/retire the existing “remote runner backend is not implemented” (currently 501).

4) Error messages and UX contracts

- When the daemon is not connected/attached, errors must include the exact next command a user should run
  (typically `rn daemon up` and attach guidance).
- When a non-primary executor is targeted for a canonical action, be explicit about primary executor selection and how to
  deliberately take over.

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
