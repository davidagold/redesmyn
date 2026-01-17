---
rn:
  node:
    branch: rn/revise-architecture/T-8-agent-config-session
  linear:
    issue_id: c0c8a260-e51c-479e-aa65-320effe4a6aa
    identifier: RED-37
  parent: T-1
---

# T-8 Split agent config from agent sessions (AgentConfig + AgentSession)

## Background / Motivation

Today, Redesmyn uses a single `Agent` record to represent:

- a *task-keyed* logical identity (the thing a task “has”)
- the *most recent run* (tmux session lifecycle, attach info, status, start/end)
- the *resolved launch configuration* (argv/env/working dir)

As we began to run, stop, and re-run agents in practice, this dual meaning created
conceptual and data-model friction:

### Observed pain points

- **Ghost identities**: when a start attempt fails early (missing harness command,
  missing tmux, etc.), we can end up with a DB row referenced by the task/node but
  with no evidence of a “real run” (no tmux session, no `started_at`, etc.). The
  UI then shows something like “not started” while still showing an agent id.
- **Mutable config makes history ambiguous**: if we treat `Agent` as the stable
  identity and later change its configuration, historical “runs” that point at
  that agent no longer reflect what actually happened at the time.
- **No place to put per-run artifacts**: we want to store (and potentially
  surface) session-scoped data like:
  - the *rendered prelude* that was sent
  - attach/session metadata
  - exit codes / end reasons
  - per-run harness argv/env and working directory
  - timing + telemetry snapshots

The current single-row approach can store only “latest”, and conflates identity,
config, and run artifacts in a way that becomes harder to reason about as the
system becomes more daemon-first and cloud-deployable.

## Desired outcome

Split the concept into two explicit primitives:

1) `AgentConfig` (or “HarnessLaunchConfig”)
   - Reusable and/or task-specific configuration used *to start* an agent session.
   - Intended to be editable over time without corrupting run history.
   - Likely sources:
     - repo defaults (`config.toml`)
     - launch configuration definitions (argv/env/working_dir)
     - (optional) per-task overrides

2) `AgentSession` (or `AgentRun`)
   - Append-only (or at least immutable-ish) record of a concrete run attempt:
     - `task_id` (or `node_id`, until Node is merged into Task)
     - `agent_config_id` (nullable if we inline a config snapshot)
     - status (`running`/`blocked`/`stopped`/`error`)
     - start/end timestamps and exit code/reason
     - attach info (tmux session name/socket/log path)
     - resolved launch configuration snapshot (argv/env/working_dir)
     - rendered prelude text (what we sent)

The UI should talk primarily in terms of **sessions**:

- “No session yet” (never run)
- “Running session” (attach available)
- “Stopped session” (last run details available)

If we retain a “logical agent” concept at all, it should be deliberately named
and scoped (e.g. `AgentConfig` or “Agent preset”), not conflated with a session.

## Design notes / Options

### Option A: Task → AgentSession directly (no Agent identity)

- `tasks.latest_agent_session_id` points at the last session.
- Sessions are the primary unit of reality and history.
- Config is either referenced (`agent_config_id`) or embedded as a snapshot.

Pros: simplest ontology; avoids “what is an Agent?” entirely.
Cons: you may later want stable identity for liveness/locking/leases.

### Option B: Task → Agent (identity) → AgentSession (history)

- `Agent` is a stable identity (task-keyed).
- `AgentSession` rows reference `agent_id` and store per-run artifacts.
- `Agent.current_session_id` points at the active session (if any).

Pros: explicit “agent slot” for leases/assignment; convenient for “current”.
Cons: easy to accidentally reintroduce config mutability ambiguity if `Agent`
stores anything that affects historical interpretation.

This task should decide which option matches Redesmyn’s intended model.

## Implementation sketch (if we choose to do it in v1)

- Add new tables (`agent_sessions` and optionally `agent_configs`).
- Update daemon runner lifecycle to create a new `AgentSession` on start, and
  update it on stop/error/exit.
- Update attach/logs to be session-scoped.
- Update API + dashboard:
  - task card shows latest session status indicator
  - details panel shows “Latest session” with harness used, attach, logs, prelude
  - optionally: show run history list (collapsed) later
- Migration:
  - treat the existing `agents` rows as “latest session” rows; backfill into
    `agent_sessions` and wire tasks to their latest session.
  - keep `agents` temporarily as a compatibility view or rename accordingly.

## Acceptance Criteria

- A clear, written decision exists: Option A vs Option B, with rationale.
- Data-model proposal includes the exact fields for `AgentSession` (and
  `AgentConfig`, if used), and how they relate to Task/Node during migration.
- Dashboard semantics are updated in the design: “no session yet” vs “latest
  session stopped/error/running”, with no “ghost identity” state.
- A migration plan is outlined (even if implementation is scheduled later),
  including how existing rows map to the new model.

## Decision

We choose **Option B: Task/Node → Agent (identity) → AgentSession (history)**.

Rationale:

- We want a stable **slot/identity** that a task “has” for future locking/leases,
  even if there is no current session.
- It is the smallest conceptual change from the current `nodes.agent_id` wiring,
  while still allowing **session history** and **immutable-ish per-run artifacts**.
- It composes cleanly with T-6 (Node → Task): sessions already carry both
  `node_id` (v0) and `task_id` (future primary).

## Proposed data model (v0)

### `agents` (identity)

An `Agent` is a task-keyed identity (“slot”), not a run.

Fields:

- `id` (int, PK)
- `display_name` (str) — currently `a-<task_id>` in v0
- `current_session_id` (int, FK → `agent_sessions.id`, nullable) — active session pointer
- `created_at` (datetime)

### `agent_configs` (launch config)

An `AgentConfig` is the editable launch configuration used to start sessions.
It is **not** the historical record of what happened in a past run.

Fields:

- `id` (int, PK)
- `agent_id` (int, FK → `agents.id`, unique)
- `launch_configuration_id` (str, FK → `launch_configurations.id`, nullable)
- `definition` (JSON) — snapshot of `LaunchConfigurationDefinition` (argv/env/working_dir)
- `created_at` (datetime)
- `updated_at` (datetime)

### `agent_sessions` (run attempts)

An `AgentSession` is an append-only (mutable only for status/end) record of a
concrete run attempt.

Fields:

- `id` (int, PK)
- `agent_id` (int, FK → `agents.id`)
- `agent_config_id` (int, FK → `agent_configs.id`, nullable)
- `task_id` (int, FK → `tasks.id`) — explicit attribution (even before T-6)
- `node_id` (int, FK → `nodes.id`) — v0 attachment (until T-6)
- `status` (enum: `running|blocked|stopped|error`)
- `host_id` (int, FK → `hosts.id`, nullable)
- `cwd_path` (str, nullable)
- `pid` (int, nullable)
- `attach` (JSON, non-null) — `AttachInfo` (tmux/external/none) with session-scoped log path
- `resolved_launch_configuration` (JSON, nullable) — immutable snapshot of `LaunchConfigurationDefinition`
- `exit_code` (int, nullable)
- `started_at` (datetime, nullable)
- `ended_at` (datetime, nullable)
- `prelude_rendered` (text, nullable) — what we attempted/sent
- `created_at` (datetime)

### Relationships

- `nodes.agent_id` keeps pointing at `agents.id` (identity).
- The UI primarily reasons about **latest session per node/agent**:
  - “No session yet”: no `AgentSession` exists (even if an `Agent` identity exists).
  - “Running session”: latest session status `running|blocked` and tmux attach available.
  - “Stopped/error session”: latest session status `stopped|error`.

## Migration plan (from current `agents` rows)

We treat each existing `agents` row as the **latest session projection** and backfill:

1) Create `agent_configs` for agents that have a `resolved_launch_configuration`:
   - `definition = agents.resolved_launch_configuration`
   - `launch_configuration_id = agents.launch_configuration_id`

2) Create a single `agent_sessions` row per agent that has any evidence of a run:
   - use existing `agents.started_at|ended_at|exit_code|attach|cwd_path|pid|status`
   - set `agent_config_id` when a config exists
   - set `node_id` by locating the node referencing that agent (if any)
   - set `task_id` via that node’s `primary_task_id` if present (or left null in v0 if unavailable)

3) Set `agents.current_session_id` for agents whose latest session is active.

We keep the legacy `agents.*` “latest” columns temporarily for compatibility and
to allow incremental rollout, but new runtime writes should be session-scoped.
