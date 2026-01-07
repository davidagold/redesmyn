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

## 2) Principles / product invariants (v0)

- No “two ways” to do the same thing (avoid mode switches that fork semantics).
- No duplicated responsibility between server and daemon (server does not run git, observers, or agents).
- One-command happy path to “dashboard open + daemon connected + repo attached”.
- Dev convenience stays in `just` (no `rn dev` product command).

## 3) Non-goals (v0)

- Multi-control-plane-instance presence (daemon presence is currently in-process).
- Robust daemon background service management across OSes (v0 can be best-effort).
- Full “remote runner” orchestration beyond local-first (cloud runner is a later epic).

## 4) Key decisions

### 4.1 Remove `runner_mode=local`

The server must not execute repo-local work. This removes the current hybrid behavior where the server can:

- start/monitor agents directly, and
- run repo observer loops in-process.

### 4.2 Remove user-facing `rn observer` and `rn dev`

- “Observer” is a daemon capability (debug-only tooling may exist, but not as a required product command).
- Dev HMR/reload is `just dev` (or equivalent), not `rn dev`.

## 5) Sequencing (tasks)

The tasks below are ordered to converge on a coherent UX while keeping the system runnable at each step.

- T-1 CLI contract + removals
- T-2 Global repo selector (`-C/--repo`)
- T-3 Control plane becomes pure (no local mode)
- T-4 Daemon agent lifecycle + remote runner backend
- T-5 Daemon owns observation + agent monitoring
- T-6 `rn daemon up/down/status` + attach/registry ergonomics
- T-7 `rn up/down` (single-command startup/shutdown)
- T-8 Package dashboard assets for installs
- T-9 Simplify `justfile` to dev-only glue

