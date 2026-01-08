# V0 Launch Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **V0 Launch** epic: intent, sequencing, and key decisions.

## Metadata

```yaml
slug: v0-launch
name: V0 Launch
root_branch: main
linear:
  project_id: null
```

## 1) Vision

Deliver a polished v0 install + startup experience with a clean, unambiguous architecture:

- **Control plane (server)**: API/UI + persistence. Must run without repo filesystem access.
- **Daemon (host-local executor)**: repo access, worktrees, agent lifecycle, git projections, telemetry.

“Local mode” is removed: there is one execution model, and it always routes repo-local work through the daemon.

## 2) Current state (why this epic exists)

Today, the system is in a hybrid state that creates ambiguity:

- The **server** can directly execute repo-local behavior in “local” mode (agent lifecycle + monitoring loops).
- The **daemon** already implements repo-executor behavior for merge/restack plans and emits repo telemetry.
- Dev entrypoints (`just run --local`, `just dev`) overlap in confusing ways, sometimes spawning multiple
  processes with partially duplicated responsibilities.

This epic resolves the overlap by cleanly separating concerns and removing the mode switch.

## 3) Principles / product invariants (v0)

- No “two ways” to do the same thing (avoid mode switches that fork semantics).
- No duplicated responsibility between server and daemon (server does not run git, observers, or agents).
- One-command happy path to “dashboard open + daemon connected + repo attached”.
- Dev convenience stays in `just` (no `rn dev` product command).

## 4) Non-goals (v0)

- Multi-control-plane-instance presence (daemon presence is currently in-process).
- Robust daemon background service management across OSes (v0 can be best-effort).
- Full “remote runner” orchestration beyond local-first (cloud runner is a later epic).

## 5) Key decisions

### 5.1 Remove `runner_mode=local`

The server must not execute repo-local work. This removes the current hybrid behavior where the server can:

- start/monitor agents directly, and
- run repo observer loops in-process.

### 5.2 Remove user-facing `rn observer` and `rn dev`

- “Observer” is a daemon capability (debug-only tooling may exist, but not as a required product command).
- Dev HMR/reload is `just dev` (or equivalent), not `rn dev`.

### 5.3 Task-doc branch metadata is not canonical

Task docs may include branch hints, but v0 treats the DB + git state as the source of truth for branch/worktree names.

## 6) Deliverables (definition of “done”)

- A user can:
  - install Redesmyn, and
  - run `rn up` against a repo, and
  - reach the dashboard with the daemon connected/attached,
  without requiring Node on their machine.
- The control plane process can run without repo filesystem access and still provide UI/API.
- All repo-local execution (agents, git/worktrees, projections, telemetry) is daemon-owned.

## 7) Plan (task graph, with parallel tracks)

This work can be parallelized into tracks, then merged in a controlled order:

- **Track A — CLI surface + repo selection**
  - T-1 (CLI contract/removals) → T-2 (repo selector)
- **Track B — Daemon execution ownership**
  - T-4 (daemon agent lifecycle) → T-5 (daemon monitoring/observation) → T-3 (remove local mode)
- **Track C — Install UX (dashboard packaging)**
  - T-8 (packaged dashboard assets)
- **Track D — Startup UX + dev glue**
  - T-6 (`rn daemon up/down/status`) → T-7 (`rn up/down`) → T-9 (`justfile` coherence)

Where a task depends on multiple tracks, we express the merge sequencing via `must_land_after` in the task metadata.

## 8) Notes / constraints

- v0 assumes a single control-plane process (daemon presence is an in-memory registry today).
- The daemon should remain “one per host” even if it attaches to multiple repos over time.
- Keep user-facing commands boring and explicit; anything experimental belongs under a debug namespace or scripts.
