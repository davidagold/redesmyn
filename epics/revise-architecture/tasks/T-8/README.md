# T-8 Split agent config from agent sessions (AgentConfig + AgentSession)

## Metadata

```yaml
id: T-8
stacked_on: T-1
node:
  branch: rn/revise-architecture/T-8-agent-config-session
```

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
     - harness profile definitions (argv/env/working_dir)
     - (optional) per-task overrides

2) `AgentSession` (or `AgentRun`)
   - Append-only (or at least immutable-ish) record of a concrete run attempt:
     - `task_id` (or `node_id`, until Node is merged into Task)
     - `agent_config_id` (nullable if we inline a config snapshot)
     - status (`running`/`blocked`/`stopped`/`error`)
     - start/end timestamps and exit code/reason
     - attach info (tmux session name/socket/log path)
     - resolved profile snapshot (argv/env/working_dir)
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

