---
rn:
  node:
    branch: rn/agent-orchestration/T-8-dashboard-graph-actions
  parent: T-7
---

# T-8 Dashboard: graph-first agent actions

## Plan

- Add task-level actions in the graph/Details panel:
  - start/stop/restart agent
  - attach instructions (e.g. tmux attach command) when applicable
  - quick access to worktree ergonomics where helpful (handoff to `rn shell`), without clutter
- Add epic-level “Run” settings so multi-agent startup does not require per-task clicking.

## Acceptance Criteria

- A user can start/stop/restart/attach an agent for a task without leaving the graph view.
- Attach UX is clear and ergonomic (“copy this command”, “open in terminal”, etc.), without forcing a separate “agents table” workflow.

## Updates

- Add graph Details panel actions:
  - start/stop/restart agent (tmux detached when available)
  - copy attach/logs commands + `rn shell --task-id …` handoff
- Keep the Details panel “Agent” section status-first and action-first; avoid inert property enumerations (e.g., full `cwd`/worktree paths) unless tucked behind a copy/expand affordance.
- Reduce visual busyness: prefer spacing + subtle separators over nested bordered panels.
- Add graph-first quick actions (hover/context) for the common loop (Start/Attach/Stop/Restart) without forcing a Details panel drilldown.
- Add an epic-level “Run” surface (settings + copy command + status summary) so multi-agent startup doesn’t require per-task clicking.

### Final designs

#### A) Task Details → Agent section (status-first, action-first)

The Agent section should read like a control surface, not a properties table:

- **Header line**: `Agent a-<task_id>` + compact status label.
- **Primary actions** (contextual):
  - Not started / stopped: `Start`
  - Running (tmux): `Attach`, `Stop`, `Restart`
  - Running (no tmux): `Stop`, `Restart`, `Copy logs`
  - Failed: `Restart`, `Copy logs`
- **Copy actions** (always available when meaningful):
  - `Copy attach command` (tmux only)
  - `Copy logs command`
  - `Copy rn shell command`

No default display of long filesystem paths; paths exist behind copy affordances.

#### B) Graph-first quick actions (hover/context)

On node hover (and/or on selection), show 2–3 icon actions derived from state:

- Start (if not running)
- Attach (if running + tmux)
- Stop (if running)
- Restart (if failed or running)

Overflow actions (copy commands) live in a small context menu, not inline.

#### C) Epic-level “Run” panel (graph header)

A compact epic-level control for fleet startup and configuration:

- **Fleet sizing**
  - `Auto-size to eligible tasks` (starts agents for all eligible tasks)
  - or `Fixed fleet size` (number input); CLI `--fleet-size` overrides this
- **Default harness configuration**
  - text input for harness command template (e.g. `codex`, `claude`, etc.)
  - `Detach (tmux) by default` toggle
- **Actions**
  - `Copy rn run …` (primary for now; UI doesn’t need to start processes directly)
  - optional `Run now` (only in local mode if we later decide to add it)
- **Status summary**
  - `running / eligible / blocked / failed` counts

This panel is the primary “graph-level configuration” surface (not a generic on/off toggle).

#### D) Task-keyed lifecycle surfaces (avoid node-keyed APIs)

Even before the Node→Task merge, user-facing surfaces should be task-keyed:

- API: `POST /v1/tasks/{task_id}/agent/start|stop|restart`
- UI routes/commands should prefer `--task` arguments and task ids.
