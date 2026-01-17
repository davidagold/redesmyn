---
rn:
  node:
    branch: rn/revise-architecture/T-2-server-daemon-connection
  linear:
    issue_id: a4a477b4-0c9f-4830-a4e8-5bde1c308107
    identifier: RED-31
  parent: T-1
---

# T-2 Server: daemon connection + presence + command delivery

## Plan

- Add a server endpoint for daemon connectivity (WebSocket recommended).
- Define repo attachment on a host-scoped daemon:
  - attach == activate repo observation + reconciliation on the daemon
  - allow the control plane to request attach/detach by `workspace_id` + `repo_id` (multiplexed over one host WebSocket)
  - attach/detach must not include host filesystem paths; the daemon resolves repo roots from its local registry
- Persist/track daemon presence:
  - daemon host identity + capabilities
  - last_seen + connection status
  - treat presence as a projection derived from connection/heartbeat events
- Implement command delivery primitives:
  - send commands to a connected daemon
  - daemon acks + retries (idempotency)
  - route repo-mutating commands to the correct executor:
    - in v1, executor identity is `hosts.host_key` (one daemon per host_key)
    - prefer an explicit executor target for repo-mutating commands or an explicit “primary executor” lease per `workspace_id + repo_id`
- Accept daemon-emitted events and persist them into the control plane event log so the existing UI WebSocket stream can broadcast updates.
  - include daemon-emitted git-derived projections used by the UI (e.g. “stack in sync with upstream”)
  - include merge/stack-restack progress events so merge UX can be server-driven while git executes on the repo executor

## Acceptance Criteria

- A daemon can connect, authenticate, and be visible as “online” in the control plane.
- A basic “ping → pong” + heartbeat updates daemon `last_seen`.
- A daemon-emitted event is persisted and observed by the dashboard via the existing `/v1/ws` stream.

## Updates

### 2025-12-31

- Recent work added `stackInSync` to the epic graph response and currently computes it in the control plane via direct git calls.
  Under the revised architecture, T-2 must persist/broadcast this as a **daemon-emitted projection** instead.
- **Coordination note:** The updates in **T-2 + T-3 + T-7** must be considered in concert to enable server-driven merges:
  - T-2 delivers/routs merge intents to the correct repo executor and persists/broadcasts merge progress events.
  - T-3 executes the git plan locally and emits projections/progress.
  - T-7 removes server git execution and defines the git/projection contracts required by both sides.
