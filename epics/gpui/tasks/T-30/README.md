---
epic: gpui
branch:
  suggested: rn/gpui/T-30-merge-restack-executor
rn:
  node:
    branch: rn/gpui/T-30-merge-restack-executor
  parent: T-29
---

# T-30 Merge/restack executor (resumable, step updates, command integration) (Domain 3)

## Problem

Executing merge/restack safely requires:

- worktree-aware git mutations,
- a clear and observable lifecycle (no silent actions),
- resumability when blocked (conflicts, running agents, etc.),
- and consistent reporting of progress to the control plane.

If execution is not modeled explicitly, the UI/CLI cannot reliably guide the user through conflicts and resume.

## Goal

Implement daemon-side execution of merge/restack plans (from T-29) that:

- executes step-by-step with safe boundaries,
- emits command updates and domain events for each step,
- supports “blocked” and “resumable” states,
- and can resume/cancel deterministically.

## Requirements

### 1) Execution engine

Implement an execution engine that:

- takes a plan + command_id/run_id,
- executes steps sequentially,
- records current step index,
- and emits:
  - command lifecycle updates (`running`, progress, `blocked`, `succeeded`, etc.)
  - domain events for UI (per-step events)

### 2) Resumability model

Define what “blocked” means and what state must be persisted to resume:

- blocked step index/kind
- worktree path
- error classification (conflict vs running agents vs git op in progress)

Resume behavior:

- resume continues from the blocked step if preconditions are now satisfied.
- cancel stops at the next safe boundary and marks command canceled.

### 3) Safety rails

Before each step:

- verify repo attachment + repo instance exclusivity lock (T-24),
- if `plan.requires_repo_primary=true`, verify primary lease (T-25),
- verify no in-progress git operations,
- verify worktree health and branch correctness.

For plans with `requires_repo_primary=false`:

- do not fail solely because lease/primary metadata is missing,
- rely on attachment + repo-instance lock + worktree/git preconditions as the safety gates.

### 4) Integration with control plane command engine

Execution must integrate with Domain 2 command lifecycle:

- command dispatch from control plane triggers execution.
- daemon reports updates on the daemon stream (T-11), which the control plane persists and publishes.

### 5) Completion semantics for merge commands

Merge execution must produce explicit task completion semantics:

- On successful `/merge`, mark affected task(s) as `done` in durable control-plane state.
  - At minimum, mark the requested target task `done`.
  - If the execution model integrates additional task branches in the same run, mark those tasks
    `done` as well.
- Persist completion metadata for each task update:
  - `completed_at`
  - `completion_source=merge_command`
  - command/run linkage needed for auditability.
- `/restack` does not mark tasks `done`.
- Non-terminal merge states (`blocked`, `failed`, `canceled`) must not mark tasks `done`.
- Completion updates must be idempotent and produce observable updates/events for UI + CLI.

### 6) Testability

Provide end-to-end daemon tests using temp repos that validate:

- successful execution (no conflicts),
- blocked state simulation (e.g., inject a conflict or fake “running agents” blocker),
- resume and cancel behavior.
- successful merge marks expected task(s) `done` with completion metadata,
- and non-success outcomes do not mutate task completion state.

No sleeps for correctness; drive execution deterministically.

## Acceptance criteria

- Merge/restack commands execute stepwise with visible progress updates.
- Blocked/resumable flows work and are observable to clients.
- Execution is safe (repo instance lock + worktree + in-progress checks; primary lease when required) and failure modes are actionable.
- Plans that do not require repo-scope primary execute without lease coupling.
- Successful merge commands durably mark expected task(s) `done`; restack and non-success outcomes do not.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on planning (T-29), worktrees (T-27), repo attachment/exclusivity (T-24), and daemon/control-plane command protocol (T-11/T-19).
- Lease/primary (T-25) applies only when `plan.requires_repo_primary=true`.

## Reference implementation (today; execution/resume orientation only)

- Execution engine (Python today):
  - `redesmyn/git_mechanics_v0.py` (`execute_merge_cascade_plan`, `execute_restack_plan`, per-step updates via `MergeRunStepUpdate`).
  - `redesmyn/merge_runs.py` (persisting merge run state; emitting events consumed by UI).
  - `redesmyn/merge_conflict_assist.py` (conflict assist supervisor; interacts with merge run state).
- API surfaces today (Python):
  - `redesmyn/api.py`:
    - `POST /v1/tasks/{task_id}/merge`
    - `POST /v1/tasks/{task_id}/restack`
    - `POST /v1/merge-runs/{run_id}/resume`
    - `POST /v1/merge-runs/{run_id}/cancel`
- Tests (Python today):
  - `tests/test_git_mechanics_execution.py` (blocked/conflict behavior; resume semantics; restack correctness).
  - `tests/test_api_integration.py` (merge/restack endpoints create merge run records; running agent blocking contract).
