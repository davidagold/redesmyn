# Repo Executor Protocol (v1, migration)

This document describes the **control plane ↔ repo executor** contract for repo-local git/worktree actions.
In v1 the repo executor is usually a **host-local daemon** connected over WebSocket, but the same contract applies
to any future “repo worker” implementation.

This spec exists to keep **local** (in-process) and **remote** (daemon) execution semantics aligned:

- The server never assumes repo filesystem access.
- Plan computation is treated as part of repo execution.
- The server coordinates routing + persistence; executors do git and stream back events/projections.

## Roles and responsibilities

### Control plane (server)

- Validates API inputs and enforces routing policy (repo-scoped canonical vs instance-scoped).
- Persists merge runs (`merge_runs` rows) and the event log (`events` rows).
- Broadcasts events to the UI WebSocket stream (`/v1/ws`).
- Delivers commands to connected executors over the daemon WebSocket (`/v1/daemon/ws`).

### Repo executor (daemon / host-local)

- Owns repo filesystem access and runs `git_mechanics_v0` plan building + execution.
- Emits progress and projection events back to the server.
- Must reject canonical mutation commands when it is not eligible (defense in depth).

## Identity and targeting

### Logical repo identity

- `(workspace_id, repo_id)` identifies a **logical repo** (server-side navigation unit, desired state, event attribution).

### Repo instance identity

- `(workspace_id, repo_id, host_key)` identifies a concrete **repo instance** (a specific host checkout/executor).

### Command targeting modes

1) **Instance-scoped commands**
   - Explicitly target a `host_key`.
   - Semantics: “do this on that repo instance”.

2) **Repo-scoped canonical commands**
   - Do not explicitly target a host in the API layer.
   - The server resolves the current **primary executor** (`repo_executor_leases`) and routes the command to that `host_key`.
   - The server must not send repo-scoped canonical mutation intents to non-primary executors.

## Capability negotiation

Daemon capability flags are carried in the daemon `hello` message (`capabilities: dict[str, Any]`) and exposed via
the server’s runtime presence registry.

### `repo_plan_v1` (bool)

When `true`, the daemon supports plan computation commands:

- `repo.merge_run.plan`
- `repo.restack.plan`

When `false`/missing, the server:

- cannot support remote `dry_run`,
- and may persist placeholder plan snapshots for remote runs until the daemon emits a real plan via events.

## Merge/restack planning and execution

Planning and execution are separate concerns:

- Planning: compute `MergeRunPlanData` for a requested operation.
- Execution: run the plan and stream progress events.

The control plane uses planning for:

- `dry_run=true` responses (return steps/base branch without executing),
- preflight validation (e.g. “running agents” checks),
- persisting an accurate `MergeRun.plan` snapshot before execution starts.

### Plan schema: `MergeRunPlanData`

Plan snapshots must conform to `redesmyn.db.models.MergeRunPlanData`.
This includes (non-exhaustive):

- `operation`: `"merge"` or `"restack"`
- `base_branch`
- `base_worktree`
- `scope`: `"spine"` or `"descendants"`
- `restack_mode`: `"strict"` or `"merge_then_restack"`
- `spine_task_ids`, `affected_task_ids`
- `steps[]` with `MergeRunPlanStepData` items

## Command protocol (server → daemon)

All commands are delivered as `ServerCommand` over `/v1/daemon/ws` and also persisted in `daemon_commands`.

### 1) Plan-only commands (v1)

These commands compute the full plan snapshot but **do not execute**.

#### `repo.merge_run.plan`

Payload (server → daemon) keys:

- `run_id: str`
- `task_id: int`
- `operation: "merge"` (optional; included for clarity)
- `scope: "spine" | "descendants"`
- `restack_mode: "strict" | "merge_then_restack"`
- `force: bool`

#### `repo.restack.plan`

Payload keys:

- `run_id: str`
- `task_id: int`
- `operation: "restack"` (optional)
- `scope: "spine" | "descendants"`

#### Plan command response (daemon → server)

Plan results are returned via `command_ack`:

- `DaemonCommandAck.state`:
  - `succeeded` on success
  - `failed` on error (include `data.detail` for a human-readable message)
- `DaemonCommandAck.data` on success:
  - `plan: MergeRunPlanData` (preferred key)
    - may also be accepted as `plan_snapshot` during migration
  - `running_agents: bool` (whether plan would affect running agents/sessions)

### 2) Execution commands (v1)

#### `repo.merge_run.start`

Payload keys (server → daemon):

- `run_id: str`
- `task_id: int`
- `operation: "merge"`
- `scope: "spine" | "descendants"`
- `restack_mode: "strict" | "merge_then_restack"`
- `allow_running: bool`
- `force: bool`
- `canonical: bool` (true iff server resolved primary target)

#### `repo.merge_run.resume`

Payload keys:

- `run_id: str`
- `allow_running: bool`
- `canonical: bool`

## Event protocol (daemon → server)

Daemon-emitted events are delivered as `DaemonEvent` messages over `/v1/daemon/ws` and persisted in the server `events` table.
The server also uses these events to update projections and run state.

### Required attribution

Daemon events must include `workspace_id`, `repo_id`, and should include `host_key` (added by the server at ingest).

### `merge.run` events

`event_type="merge.run"` is used to drive merge-run status and UI updates.

Recommended fields in `data`:

- `run_id: str`
- `status: "running" | "blocked" | "resumable" | "succeeded" | "failed" | "canceled"`
- `operation: "merge" | "restack"`
- `task_id: int` (task currently being acted on / last task for “merge” spine)
- `epic_id: int`
- `requested_task_id: int`

#### Plan snapshot attribution (fixes placeholder plans)

On the first “running/started” event for a run, the daemon should include:

- `plan: MergeRunPlanData` (preferred)
  - or `plan_snapshot` during migration

The server will persist this into `merge_runs.plan` so the UI can show steps/base branch promptly.

#### Block/resume attribution

When blocked or resumable, include:

- `blocked_step_index: int`
- `blocked_step_kind: str`
- `blocked_branch_name: str`
- optionally `error: str`

### `task.merge` events

`event_type="task.merge"` is used for step-level progress.
Payloads are produced by `git_mechanics_v0` via callbacks and should include (non-exhaustive):

- `run_id`
- `task_id`
- `kind` (`"merge_ff"` or `"rebase"`)
- `phase` (`"started" | "succeeded" | "failed" | ...`)
- `branch_name`
- optionally `error`

## Server-side persistence expectations (migration notes)

- The server persists `merge_runs.host_key` and `merge_runs.canonical` on run creation.
- The server may create a placeholder `merge_runs.plan` in remote mode when it cannot obtain a real plan.
- When `merge.run` events include `plan`/`plan_snapshot`, the server updates the persisted plan snapshot.

## Backwards/forwards compatibility

- Daemons should only advertise `repo_plan_v1=true` once both plan command types are implemented.
- The server should treat missing/false `repo_plan_v1` as “plan not supported” and fall back to:
  - no remote dry-run, and
  - placeholder plan snapshots until a `merge.run` event carries the real plan.
