---
epic: gpui
branch:
  suggested: rn/gpui/T-27-worktree-service
rn:
  node:
    branch: rn/gpui/T-27-worktree-service
  parent: T-26
---

# T-27 Worktree management service (Domain 3)

## Problem

Redesmyn’s concurrency model depends on worktrees:

- each task maps to a branch + worktree,
- git operations must be safe across multiple worktrees,
- and the daemon must be the single writer that creates/mutates worktrees (enforced by repo instance exclusivity; see T-24).

If worktree behavior is implicit and scattered, we risk:

- corrupted worktrees,
- inconsistent branch/worktree mapping,
- and hard-to-debug “wrong cwd” issues for agents.

## Goal

Implement a daemon-side worktree management service that:

- owns the worktree layout policy,
- creates/repairs worktrees deterministically,
- validates invariants (branch matches worktree, etc.),
- and exposes a small API used by higher-level operations (agents, merge/restack).

## Requirements

### 1) Worktree layout policy

Define a deterministic layout under a daemon-managed root, e.g.:

- `${worktree_root}/{repo_id}/{epic_ref}/{task_ref}/`

Rules:

- paths are stable across runs,
- safe to compute from identifiers (no titles in paths unless sanitized),
- collisions are impossible given unique IDs.

### 2) Core operations

Implement operations:

- `ensure_task_worktree(task_id, branch_name) -> WorktreePath`
- `verify_worktree_health(...) -> WorktreeHealth`
- `remove_worktree(...)` (best-effort; guarded)

Enforce:

- worktree is on expected branch (not detached),
- branch exists and points to expected ref when required,
- no destructive actions without explicit command intent.

#### 2.1) Task session start contract (worktree-first startup)

Task session startup must consume the worktree service directly:

- `StartTaskSession` must call `ensure_task_worktree(...)` before launching an agent runtime.
- The resolved task worktree path is passed through the agent-driver start API as required
  `working_directory`.
- Driver implementations must fail fast with a structured error if `working_directory` is missing,
  inaccessible, or not a git worktree path.
- Implement this for Codex driver first; keep the same driver interface contract for Claude Code.

#### 2.2) Branch materialization (stack correctness)

When creating a task worktree, the daemon is responsible for ensuring the task’s **git branch backing**
exists and is based on the correct “stack” base ref.

Required behavior (parity with Python `ensure_task_worktree`):

- If the task branch already exists: add/attach the worktree to that branch (no implicit rebases).
- If the task branch is missing: create it from `base_ref`, then add the worktree.
- **Never** create a new task branch from repo `HEAD` as a fallback.

`base_ref` selection rules:

- Start with `base_ref = epic.root_branch`.
- Let `cursor = parent_task` (walk the parent chain upward).
- While `cursor` exists and has a `branch_name`:
  - If `cursor.branch_name` is already contained in `epic.root_branch` (i.e. `git is-ancestor cursor.branch epic.root_branch`):
    - The cursor task is effectively merged; continue walking upward.
  - Else:
    - If `cursor` is marked `done`, error: “parent is done but root branch does not contain its branch; fast-forward the base branch first.”
    - Otherwise, set `base_ref = cursor.branch_name` and stop.

This rule ensures new child branches are based on the nearest unmerged ancestor branch (or the epic root)
so stacked branches remain coherent even before merge/restack (T-29/T-30) is implemented.

### 3) Repo instance exclusivity

Worktree-creating/removing operations are mutating:

- must require repo attachment + repo instance exclusivity (attach lock; T-24).
- do not require a repo-scope primary lease (T-25); they are strictly local to a repo instance.

### 4) Telemetry hooks

Provide structured “worktree health” data for the observation loop (T-28) and for UI display.

### 5) Testability

- Use temp git repos + worktree fixtures in tests.
- Tests cover create/verify/repair behavior and common failure modes.

## Acceptance criteria

- The daemon can deterministically create and validate task worktrees.
- Worktree invariants are enforced with actionable errors.
- Higher-level code can rely on a small, typed API (no ad-hoc worktree shelling out).
- Starting a task session guarantees the runtime starts in that task's worktree (`working_directory` is explicit, not implicit).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on git backend abstraction (T-26) and repo attachment/exclusivity (T-24) for mutating ops.
- Used later by agent runtime (Domain 4) and merge/restack execution (T-30).
- Provides startup preconditions for task-session initialization and agent-driver cwd routing.

## Reference implementation (today; worktree orientation only)

- Worktree creation/repair (Python today):
  - `redesmyn/agent_runtime.py` (`ensure_task_worktree`, `checkout_task_worktree`, worktree/branch semantics used by agents).
  - `redesmyn/repo.py` (`git_worktree_add` and related helpers).
  - `redesmyn/cli.py` (`rn sync --from local` sets `branch_name` and repairs worktree branch mismatch in some cases).
- Current worktree layout (Python today):
  - `.redesmyn/worktrees/…` (layout appears in tests; exact policy lives in the Python worktree helpers).
- Tests (Python today):
  - `tests/test_agent_worktree_autobranch.py` (auto-branch + worktree creation semantics).
  - `tests/test_cli_integration.py` (worktree layout and branch mismatch repair scenarios).
