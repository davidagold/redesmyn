# Harness Interface v0 Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Harness Interface v0** epic: intent, scope, current state, and plan. Keep it current.

## Terminology note

You were looking for **successor** (or “replacement epic”) rather than “supersessor”.

## Glossary (v0)

- **Agent**: the interactive program being driven (Codex, Claude Code, a generic CLI).
- **Agent session**: a single run instance of the agent for a task (attach/log pointers + runtime status).
- **Transport**: tmux/pty/process IO (send text/interrupt, collect output/logs).
- **Interpreter**: consumes output and produces semantic state (ready/turn complete) + capability declaration.
- **Driver**: long-running loop that owns an interpreter for a session, persists/broadcasts semantic status, and issues safe high-level operations via transport.

Rationale: the UI and workflows are session-first; the DB `Agent` identity object is legacy/internal; and “harness” reads as a test/process harness, so v0 reserves “transport/runtime” for tmux/pty concerns and uses “Agent” for program semantics.

## Metadata

```yaml
slug: harness-interface-v0
name: Harness interface v0
root_branch: main
supersedes:
  - epic: agent-orchestration
    scope: harness-related unfinished work (profiles/adapters/doctor)
linear:
  project_id: null
```

## 1) Why this epic exists

The original `agent-orchestration` epic treated “harness integration” as a set of adapters and profiles layered on top of a runner model. Since then:

- We’ve sharpened the control-plane/daemon split (see `epics/revise-architecture/README.md` and T-9).
- We’ve implemented a working local-first agent loop (tmux-backed, prelude support, sandbox knobs, etc.).
- The remaining harness work is no longer “a few adapters”; it wants a coherent, typed **interface** that we can rely on across:
  - local execution (server == runner host)
  - remote execution (server routes to daemon executor)
  - UI workflows (configure/run/attach/logs)

So, this epic is the successor for the unfinished harness-centric tasks from `agent-orchestration`, with an updated conception: **define the harness interface first**, then implement adapters as data + validation on top.

## 2) Current state (what exists today)

This section describes how harness integration currently works in `main`.

### 2.0 Executive summary

We already have a working “harness” system, but the contract is implicit and spread across the server, CLI, UI, and local runner implementation:

- The **server** (and CLI) orchestrate agent session lifecycle and store `AgentSession` state in the DB (task-anchored; no persistent DB `Agent` identity).
- The **executor** is effectively “local host” today (`LocalRunnerBackend`); the remote backend exists but is not yet a real implementation.
- A “launch configuration” is stored in `launch_configurations` and snapshotted onto sessions, but we don’t yet treat this as a stable, versioned **interface**.

### 2.1 Data model (DB)

Hubs of harness-related state:

- `LaunchConfiguration` (`launch_configurations` table; ORM: `redesmyn/db/models.py`):
  - `id` (string PK)
  - `kind` (string)
  - `source` (`builtin`/`user`)
  - `display_name`
  - `definition` (JSON)
- `AgentSession` (`agent_sessions` table; ORM: `redesmyn/db/models.py`):
  - sessions are **task-anchored** (`task_id`) and are the canonical runtime record for “agent state”
  - there is no persistent logical DB `Agent` identity (labels are derived from `task_id`, e.g. `a-<task_id>`)
  - sessions snapshot `launch_configuration_id`, `resolved_launch_configuration`, `attach`, `cwd_path`, and `prelude_rendered`
  - sessions also carry `started_at` / `ended_at` timestamps (derived from lifecycle)

Related schema/API types live in `redesmyn/schemas/core.py`.

### 2.1.1 Repo-scoped defaults for harness & sandbox

The UI and CLI are primarily driven by repo-scoped “orchestration defaults” (loaded from `.redesmyn/config.toml`):

- `harness.command`: shell command used to start the harness (e.g. `codex`, `claude`, etc.)
- `harness.detach`: whether we run in a tmux session by default
- `harness.prelude`: template text sent after start (optional)
- `harness.send_prelude`: whether we send the prelude
- `harness.submit_prelude`: whether we press Enter after sending
- `sandbox.type`: currently `none` or `worktree`
- `sandbox.network`: currently `allow` or `deny`

Definition + normalization are in `redesmyn/orchestration_config.py`.

### 2.2 Runtime implementation (local-first)

The harness process model is currently:

- **tmux-first**: agent sessions are started in a stable tmux session per task (`rn-a-<task_id>`).
- A harness command is treated as a shell-like string, parsed to `argv`, and launched in the task worktree.
- A git “shim” can be injected onto `PATH` so `git` resolves to a wrapper that runs `rn git …` (best-effort enforcement).
- A prelude message can be rendered (with placeholders) and sent into the tmux session after startup.
- Sandbox policies exist (notably “worktree sandbox”) and can be enabled via orchestration defaults.

Key implementation files:

- `redesmyn/agent_runtime.py`:
  - parses harness command → `argv`
  - ensures worktree + tmux session
  - records attach + resolved profile + rendered prelude
  - best-effort git shim injection
  - sandbox configuration
- `redesmyn/runner_backend.py`:
  - `LocalRunnerBackend` calls `agent_runtime.*`
  - `RemoteRunnerBackend` exists as a placeholder (returns 501)

### 2.3 User surfaces

- CLI:
  - harness defaults and prelude behavior are configured via `.redesmyn/config.toml` (`rn config set harness.*`)
  - starting/restarting agent sessions accepts a harness command + optional one-time prelude override
- API:
  - `/v1/tasks/{task_id}/agent/start|restart|stop`
  - `/v1/harness-profiles` list/upsert
  - orchestration defaults endpoints expose harness + prelude config
- Dashboard:
  - “Configure” panel includes harness command + detach mode + prelude toggles
  - task cards show agent status + attach affordances
  - task details drawer includes the last-run harness command (and, when stopped/errored, allows overriding it for restart)

## 3) What `agent-orchestration` planned (harness-related)

The harness plan in `epics/agent-orchestration/README.md` and its tasks was roughly:

1) Define the harness integration model + runner boundary (profiles, attach, capabilities, doctor).
2) Implement a generic runner/session model (tmux, logs, attach semantics).
3) Add **data-driven launch configurations** + `rn agent doctor` validation tooling.
4) Add per-harness “adapters” that are mostly profiles + docs (Codex, Claude Code, Cursor, Amp, OpenCode).
5) Add sandboxing support and surface it in the UI.

Some of these ideas landed (tmux-first orchestration defaults, prelude plumbing, basic launch configuration persistence), but the “interface” is still implicit and interleaved with local execution details.

### 3.1 The specific unfinished harness work (as expressed in tasks)

The harness-centric tasks that are incomplete (or only partially realized in code) include:

- Profiles + doctor:
  - `epics/agent-orchestration/tasks/T-15/README.md` (profiles, capabilities, doctor, degraded-mode reporting)
  - `epics/agent-orchestration/tasks/T-16/README.md` (repo defaults: harness + prelude)
- Per-harness “adapters” (intended to be mostly data + docs):
  - `epics/agent-orchestration/tasks/T-10/README.md` (Codex)
  - `epics/agent-orchestration/tasks/T-11/README.md` (Claude Code)
  - `epics/agent-orchestration/tasks/T-12/README.md` (Cursor)
  - `epics/agent-orchestration/tasks/T-13/README.md` (Amp)
  - `epics/agent-orchestration/tasks/T-14/README.md` (OpenCode)
- Sandboxing:
  - `epics/agent-orchestration/tasks/T-18/README.md` (sandbox provider + UX)

This epic does not invalidate the intent of those tasks, but it changes their sequencing and responsibility boundaries: we want to lock down the interface and capability semantics first, then make profiles/doctor/sandbox “plug into” that interface cleanly.

## 4) What’s missing / what we’re changing

### 4.1 What’s missing (gaps)

- A first-class, typed **harness interface**: a stable contract for:
  - launch
  - attach/inspect
  - send prelude / send text
  - log streaming / log paths
  - capability declaration + degraded mode reporting
- A single place to define harness “capabilities” in a way that both UI and server logic can consume.
- A coherent plan for remote execution:
  - server issues “start/stop/restart/attach/logs” commands to the executor host
  - the daemon executes and reports structured results

### 4.2 Design shift

Instead of “N harness adapters”, we want:

- **One** harness interface contract (v0),
- **Many** profiles (data) + validations (doctor),
- Optional harness-specific enrichments only when they add real value.

## 5) v0 scope (proposal)

This section captures the v0 plan as revised by the goals in this thread.

### Must-have

- Define a first-class **Harness interface**: semantic status + capability contract that is independent of the terminal/session transport.
- Implement a **GenericHarness** that supports any command (best-effort), with only the capabilities we can provide without harness-specific assumptions.
- Implement harness-specific adapters for:
  - **Codex**
  - **Claude Code**
- Add harness identification:
  - automatically infer harness kind from the user command (best-effort)
  - allow users to override the inferred harness explicitly
- Enable conflict auto-assist for merge/restack:
  - on conflict, automatically send the remediation message to the agent when supported
  - require **both**: (a) repo conflict resolved and (b) agent has completed its turn before resuming

### Nice-to-have

- A harness “support matrix” surface in the UI derived from capabilities.
- `rn harness doctor <kind>` (or `rn agent doctor`) driven by the interface/capabilities.

### Non-goals (v0)

- Deep harness-specific message/command integrations beyond what we can standardize.
- Multi-user scheduling/quotas for shared fleets.
- Supporting every harness under the sun with first-class semantics (GenericHarness exists specifically so unknown harnesses still work, just without advanced features).

## 7) Task map (v0)

- `epics/harness-interface-v0/tasks/T-8/README.md`: Remove DB `Agent` construct; align terminology so “Agent” is the program.
- `epics/harness-interface-v0/tasks/T-1/README.md`: Agent interface + capability model + GenericAgent (baseline).
- `epics/harness-interface-v0/tasks/T-7/README.md`: AgentDriver (output → semantic status) plumbing shared by all agent implementations.
- `epics/harness-interface-v0/tasks/T-2/README.md`: Agent kind identification (infer from command) + user override (UI/CLI/data model).
- `epics/harness-interface-v0/tasks/T-3/README.md`: Codex agent interface implementation (turn/idle detection + capabilities).
- `epics/harness-interface-v0/tasks/T-10/README.md`: Structured exec mode + semantic event stream plumbing (enables deterministic turns/messages).
- `epics/harness-interface-v0/tasks/T-9/README.md`: UI: task card agent message preview (1-line, truncated).
- `epics/harness-interface-v0/tasks/T-4/README.md`: Claude Code agent interface implementation (turn/idle detection + capabilities).
- `epics/harness-interface-v0/tasks/T-5/README.md`: Conflict auto-assist + gated auto-resume (requires repo clean + agent turn complete).
- `epics/harness-interface-v0/tasks/T-6/README.md`: Agent doctor + capabilities surface (UI + CLI).

## 6) Relationship to `tests-v0`

This epic should add (or enable adding) integration tests that validate:

- harness start/stop/restart state transitions
- prelude rendering + delivery invariants
- attach metadata correctness
- degraded mode reporting
- remote runner command routing (when implemented)
