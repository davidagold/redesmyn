# T-8 Dashboard: graph-first agent actions

## Metadata

```yaml
id: T-8
stacked_on: T-7
node:
  branch: rn/agent-orchestration/T-8-dashboard-graph-actions
```

## Brief (local)

- Add node-level actions in the graph/Details panel:
  - assign/unassign agent
  - start/stop/restart session
  - attach instructions (e.g. tmux attach command) when applicable
  - quick access to worktree ergonomics where helpful (handoff to `rn checkout`), without clutter

## Acceptance Criteria

- A user can start/stop/restart/attach an agent session without leaving the graph view.
- Attach UX is clear and ergonomic (“copy this command”, “open in terminal”, etc.), without forcing a separate “agents table” workflow.

## Updates

- Added graph Details panel actions:
  - assign/unassign agent (with a lightweight dropdown)
  - start/stop/restart session (runner-owned; detached tmux when available)
  - copy attach/logs commands + `rn checkout --node …` handoff
- Added node session control API endpoints:
  - `POST /v1/nodes/{node_id}/session/start`
  - `POST /v1/nodes/{node_id}/session/stop`
  - `POST /v1/nodes/{node_id}/session/restart`
- `POST /v1/nodes/{node_id}/agent` now emits `node.agent_set` so the WebSocket stream can trigger a refresh.
- Epic graph returns all agents (not just assigned) so the UI can reassign without a separate agents list view.
- Keep the Details panel “Agent” section status-first and action-first; avoid inert property enumerations (e.g., full `cwd`/worktree paths) unless tucked behind a copy/expand affordance.
- Reduce visual busyness: prefer spacing + subtle separators over nested bordered panels for the Agent/Session/Worktree blocks.
- Add graph-first quick actions (hover/context) for the common loop (Start/Attach/Stop/Restart) without forcing a Details panel drilldown.
- Add an epic-level “fleet” affordance (copy `rn run --epic … --fleet-size …` + status summary) so multi-agent startup doesn’t require per-task clicking.
