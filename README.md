# Redesmyn

Redesmyn is a local-first cockpit for orchestrating multi-agent work on a git repo, using a task graph (stacked branches) and a real-time UI.

It’s an experiment in what “project management” looks like when:

- work is actually merged via git branch sequencing, and
- a fleet of agents (not just humans) is doing the work.

## Why a graph (and not just tickets)

Ticket systems typically treat parent/child as **constitution** (“this work is part of that work”).
In Redesmyn, edges primarily represent **sequencing** (“this branch should land before that branch”).

That distinction matters because:

- representing fulfillment along the *sequencing* dimension calls for different views than lists/tables,
- and the graph directly maps to merge behavior (what can land when, and onto what).

The graph is about **merge sequencing**, not “who can work on what”.
Downstream tasks can be worked on simultaneously as long as contracts are clear and shared context stays coherent.

## Agents (worker layer + overseer layer)

Redesmyn maintains a layer of *workers* (task-scoped agents) whose context is derived from:

- the task itself,
- its position in the graph (dependencies + dependents),
- and the epic’s higher-level intent.

This gives both low- and high-granularity context, and reduces the amount of context the human has to manually manage.
The intent is that the user mostly interacts with a much smaller **overseer/meta** layer that reviews, communicates, and coordinates workers on their behalf.

The “agent harness” layer is intended to be extensible: users should be able to run whichever agent program they prefer, while Redesmyn standardizes the orchestration surface (status, capabilities, commands).

## Architecture (today and where it’s going)

The current architecture is a work in progress and somewhat baroque for local usage, but it’s moving toward remote/distributed orchestration:

- **Control plane (server)**: state, APIs/UI, persistence, projections.
- **Daemon / repo executor (host-local)**: repo/worktrees, git actions, telemetry, process/session lifecycle.

Long-term, the goal is for the control plane to operate without assuming repo filesystem access, and to route git mutations to an explicit executor.

## Integrations

Tickets are still important (and they’re everywhere), so we integrate with them.
The goal is a unified interface for external events agents can act on: GitHub, Linear, CloudWatch, etc.

That orchestration is intentionally not fully generic: it stays close to (but not myopically eclipsed by) the code and the repo graph where work actually lands.

## UI philosophy

Most GUIs are lists, tables, and widget panels.
Our hunch is this won’t be enough for understanding and steering an agent fleet: there’s too much information to distill uniformly.

This project is an attempt to explore graph-native, realtime representations that make agent activity legible to humans.

## Status / limitations

There are many rough edges right now:

- incomplete harness interfaces (turn detection, capability signaling, etc.)
- partial distributed-mode story (still being refined)
- sharp corners around git state, projections, and concurrency

If you hit something confusing, assume it’s not you.

Thanks for reading.

## Development quickstart

**Prereqs**

- Python 3.11+
- Node (see `.nvmrc`; Vite currently expects Node `>=22.12.0`)
- `uv` (install via Homebrew: `brew install uv`)
- Optional: `hk` (git hooks manager; install via Homebrew: `brew install hk`)

**Dev (recommended): one command**

- Run dashboard HMR + daemon reload on a single origin: `rn dev`
  - If `rn` isn’t installed yet: `uv run rn dev`
  - Dashboard: `http://127.0.0.1:9234/` (Vite)
  - API: `http://127.0.0.1:9234/v1/*` (proxied to the daemon on `:9235`)
- Install pre-commit formatting hooks (one-time): `just hooks` (or `hk install`)

**Install + init**

- Install: `just install` (runs `uv sync` + `uv tool install --editable . --force`)
- Make `rn` available without `uv run` (pick one):
  - Recommended (global install): `uv tool install --editable .` (one-time)
  - Or per-shell: `source .venv/bin/activate`
- Initialize repo state: `rn init`

**Dashboard (alternate setup)**

- `cd dashboard && npm install`
- Run dev server (separate origin): `npm run dev` (proxies `/v1/*` to the daemon)
- Or build + serve from daemon: `npm run build` then open `http://127.0.0.1:9234/`

## Epics

- `epics/README.md`
- `epics/redesmyn/README.md` (bootstrapping epic; we dogfood Redesmyn to build Redesmyn)
- `epics/revise-architecture/README.md` (control plane/daemon split)
- `epics/agent-orchestration/README.md` (own agent lifecycle + harness integration)
- `epics/messages-commands/README.md` (messages, commands, and agent control loop)
- `epics/graph-viz/README.md` (graph UI improvements)
