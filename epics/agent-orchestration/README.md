# Agent Orchestration Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Agent Orchestration** epic: intent, v0 spec, sequencing, and key decisions. Keep it current.

## Metadata

```yaml
slug: agent-orchestration
name: Agent Orchestration
root_branch: main
linear:
  project_id: null
```

## 1) Vision

Enable real multi-agent dogfooding by having Redesmyn **start, manage, and observe** agent harness processes while keeping the UI **graph-first** and real-time:

- Start/stop/restart per-node agent sessions from the graph UI and CLI.
- Allow humans to **attach/detach** into those sessions (so “agent orchestration” doesn’t mean losing the harness UX users already like).
- Persist liveness + activity (commits, worktree state) and stream updates to the UI via WebSocket.
- Support multiple harnesses with a thin adapter layer:
  - Claude Code
  - Codex
  - Amp
  - OpenCode
  - Cursor

## 2) Key decisions

### 2.1 Own orchestration (not BYO orchestrator)

Redesmyn owns agent lifecycle (startup/cleanup), session identity, assignment, liveness, messaging, and synchronization primitives. We still integrate with harnesses, but we do **not** rely on external orchestration systems as the source of truth.

### 2.1.1 “Adapters” are mostly profiles + validation + docs (v0)

For many harnesses, the “adapter” should be largely data-driven:

- **Profile**: how to launch, attach, and what capabilities to assume (hooks/no hooks, CLI/GUI, etc.).
- **Validation**: preflight + smoke checks (`rn agent doctor <harness>`) so users get actionable errors instead of silent degraded behavior.
- **Documentation**: clear UX guidance and explicit limitations per harness.

Hooks (when available) are an enrichment path, not a hard dependency.

### 2.2 Split control plane vs runner (even if co-located in v0)

- **Control plane**: state + commands + events + API/UI.
- **Runner**: host-local worktrees + spawning harness processes + telemetry.

v0 can run both in a single local daemon, but the boundary must exist in the code and data model so we don’t bake in “server == host”. This matters for cloud deployment where the agent host and server diverge.

### 2.3 Detach/attach via tmux when possible

Prefer tmux-backed sessions for harness processes to enable:

- `rn agent run ... --detach`
- `rn agent attach ...`

Provide a fallback mode when tmux isn’t available (foreground process + logs).

### 2.3.1 Git enforcement strategy: PATH shim + cooperative skill guidance

To keep invariants enforceable even when harnesses don’t support hooks:

- **Enforced path (preferred)**: inject a `PATH` shim so `git` resolves to a wrapper that calls `rn git ...` (blocks/invariants apply).
- **Cooperative path**: supply a bootstrap prelude (and/or a skill, e.g. via `agentskills.io`) that instructs the agent to use `rn`/API surfaces (useful even when `git` can’t be reliably shimmed).

### 2.4 Graph-first UX, lists as exceptions

The graph is the control surface. “Agents list” is an exception view for unassigned/offline/global settings, not the primary workflow.

### 2.5 WebSocket event stream

Use WebSocket as the primary real-time transport so we can later support interactive app-level chat and richer bidirectional flows.

## 3) v0 scope

### 3.1 Required capabilities (MVP)

- Agent sessions:
  - start/stop/restart, status, last-seen
  - harness selection
  - attach/detach when supported
- Activity + telemetry:
  - detect commits and ref movements for node branches
  - basic worktree health (exists, clean/dirty, current branch)
- UI:
  - agent presence + activity integrated into graph nodes
  - live updates without manual refresh

Messaging and commands are intentionally factored into a separate epic: `epics/messages-commands/README.md`.

### 3.2 Harness adapters (parallelizable)

After the runner/session model exists, harness adapters can be implemented in parallel (they should not require a strict sequence). Hooks, when available, are an enrichment path rather than a hard dependency.

### 3.3 Explicit non-goals (v0)

- Multi-user orchestration with shared state across developers.
- “Perfect” deep harness integration for every vendor (hooks are opportunistic).
- A general plugin ecosystem for third-party orchestration systems.

## 4) End-to-end acceptance criteria (dogfooding)

From the dashboard graph:

1. Select a node, start a harness session (e.g. Codex) for that node.
2. Attach into the session, make commits, and see the node update live (status + activity pulse).
3. Switch between nodes/worktrees ergonomically (`rn checkout`-style workflow) without breaking worktree invariants.
