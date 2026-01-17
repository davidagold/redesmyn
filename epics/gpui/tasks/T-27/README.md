---
epic: gpui
branch:
  suggested: rn/gpui/T-27-worktree-service
rn:
  parent: T-26
---

# T-27 Worktree management service (Domain 3)

## Problem

Redesmyn’s concurrency model depends on worktrees:

- each task maps to a branch + worktree,
- git operations must be safe across multiple worktrees,
- and the daemon must be the single writer that creates/mutates worktrees (subject to lease).

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

### 3) Lease enforcement

Worktree-creating/removing operations are mutating:

- must require primary lease (T-25).

### 4) Telemetry hooks

Provide structured “worktree health” data for the observation loop (T-28) and for UI display.

### 5) Testability

- Use temp git repos + worktree fixtures in tests.
- Tests cover create/verify/repair behavior and common failure modes.

## Acceptance criteria

- The daemon can deterministically create and validate task worktrees.
- Worktree invariants are enforced with actionable errors.
- Higher-level code can rely on a small, typed API (no ad-hoc worktree shelling out).

## Dependencies / sequencing

- Depends on git backend abstraction (T-26) and lease enforcement (T-25) for mutating ops.
- Used later by agent runtime (Domain 4) and merge/restack execution (T-30).

