# T-3 Daemon + CLI: spawn/attach/stop per-task agents (tmux-first)

## Metadata

```yaml
id: T-3
stacked_on: T-2
node:
  branch: rn/agent-orchestration/T-3-runner-tmux-cli
```

## Brief (local)

- Implement daemon-owned process lifecycle for **per-task agents**:
  - spawn a harness process in the task’s worktree with consistent env injection
  - capture logs/output
  - stop/restart gracefully
- Prefer tmux-backed detached processes to support `rn agent attach`.
- Add ergonomic CLI commands for humans:
  - `rn agent start/attach/stop/restart/logs`
  - `rn run` (fleet: start many agents at once)
- Establish a baseline “generic” harness runner suitable for early dogfooding:
  - run the harness in the node worktree
  - inject a `PATH` shim so `git` resolves to `rn git ...` (best-effort invariant enforcement)
  - emit clear warnings when shimming is unavailable or likely ineffective
  - optionally print/bootstrap cooperative guidance (e.g., skill-based or textual prelude)

## Acceptance Criteria

- `rn run --epic <slug|id> --fleet-size <n>` starts **N per-task agents** without per-task clicking.
- `rn agent start/attach/stop/restart/logs --task <task_id>` is ergonomic for humans.
- `rn agent attach` works when tmux is available; a fallback path exists when not.
- The daemon updates agent state (status, last-seen, attach/log metadata) in the control plane.
- The harness environment resolves `git` to the shim (or Redesmyn reports a degraded mode explicitly).
- Agents reference a persisted `harness_profile_id` and record the resolved profile/attach metadata actually used.

## Updates

- Daemon `git` shim strips itself from `PATH` before delegating to `rn git` to avoid recursion.
- Daemon can “adopt” pre-existing worktree paths when the branch matches (useful when the node worktree already exists).
- Ad-hoc runs persist a `HarnessProfile` keyed by a stable hash of the resolved definition.
- Merge “session” into “agent” for v0: a task’s agent is the harness instance; do not model/require a separate AgentSession identity for orchestration UX.

### Final designs

#### A) Ontology + invariants

- **One agent per task** (`a-<task_id>`), representing the harness process intended to work on that task’s branch/worktree.
- “Session” is not user-facing and not required in the data model for this epic; agent lifecycle owns:
  - starting/stopping/restarting
  - attach instructions (tmux) and log paths
  - process status + timestamps
- **Stable tmux session name** per task agent (reused across restarts): `rn-a-<task_id>`.

#### B) Fleet workflow (`rn run`)

Command:

- `rn run [--epic <slug|id>] [--fleet-size <n>] [--harness \"<command>\"] [--detach/--no-detach] [--dry-run] [--restart]`
- `--epic` is optional only when inferable (single epic), otherwise required.
- `--fleet-size` overrides the default (see `T-16`); omit to use the configured default or `auto`.

Modes:

- **Fleet mode**: `--fleet-size <n>` (or default) starts agents for the top N eligible tasks (ordered).
- **Explicit mode** (mutually exclusive with fleet size): `rn run --task <task_id>...` (start agents only for that set).

Guardrails:

- `--dry-run` prints the plan with no mutations (no agent creation, no process spawn).
- Default behavior is idempotent and safe:
  - skip tasks whose agents are already running
  - skip tasks with `state ∈ {blocked, done}`
- `--restart` stops and restarts agents for selected tasks (still respects eligibility rules unless explicitly overridden).

Eligibility (default):

- Task must be in the epic.
- Task must have a branch/node backing (`Task.node_id != NULL`) so a worktree can be resolved/created.
- Task `state ∈ {todo, in_progress}` (include `in_progress` by default).
- Exclude `blocked` and `done` by default (future flags can override).

Ordering (fleet mode):

- Topological / parent-first by the stack topology (closest-to-trunk first), stable tie-break by task id.

Provisioning:

- Ensure an agent row exists for each selected task:
  - name: `a-<task_id>`
  - identity is pinned to the task (agents are not “reassigned” across tasks).
- Start the agent in the task worktree with the configured harness command.

Output:

- Always prints a concise plan section (“eligible → selected → skipped (why)”).
- Then prints a compact table of started agents with copy-paste commands:
  - `rn agent attach --task <task_id>`
  - `rn agent logs --task <task_id>`
  - `rn shell --task-id <task_id>`

#### C) Task-keyed lifecycle surfaces (avoid node-keyed APIs)

Even before the Node→Task merge, user-facing surfaces should be task-keyed:

- CLI: `rn agent start|stop|restart|attach|logs --task <task_id>`
- API: `POST /v1/tasks/{task_id}/agent/start|stop|restart`

Internally, resolve `task_id → node_id` via `tasks.node_id` until the architecture refactor merges nodes into tasks.
