---
epic: gpui
branch:
  suggested: rn/gpui/T-29-merge-restack-planner
rn:
  parent: T-26
---

# T-29 Merge/restack planner (deterministic plans, no execution yet) (Domain 3)

## Problem

Merge/restack operations are central and complex:

- must be safe (worktree-aware, conflict-aware),
- must be explainable (UI shows plan, blockers, next steps),
- and must support resumable/blocking states.

We want to separate concerns:

- planning is deterministic and testable,
- execution can be resumed and emits step updates.

## Goal

Implement a deterministic merge/restack planning module in the daemon that:

- takes a requested task and scope (spine/descendants),
- computes the ordered sequence of git steps,
- and returns a typed plan suitable for UI display and for execution by T-30.

This ticket does not execute git mutations; it only plans.

## Requirements

### 1) Plan types

Define typed plan structures:

- `MergePlan` / `RestackPlan`
- list of steps with:
  - step index
  - kind (`rebase`, `merge_ff`, etc.)
  - target task/branch/worktree
  - upstream/base refs
  - metadata needed for UI (what is being merged where)

Plans must be stable/deterministic given the same repo state.

### 2) Inputs and invariants

Planner inputs include:

- epic/task graph topology (from control plane state or a cached view),
- current repo refs (from git backend),
- worktree availability/health (from worktree service),
- “merge ready” gating info where relevant.

Planner must validate:

- branch/worktree existence,
- no in-progress git operations in target worktrees,
- and that required invariants hold before returning a plan.

Return actionable planning errors when not possible.

### 3) Scope semantics

Define and document:

- `scope=spine` vs `scope=descendants`
- ordering rules (why a step is in the plan)

### 4) Blocker representation

Planner must surface blockers without executing:

- “worktree missing”
- “dirty worktree”
- “git operation already in progress”
- “not primary executor”

These should be representable as structured errors suitable for UI callouts.

### 5) Testability

Planner tests use temp repos and small task graphs.

- No sleeps.
- Assert plan structure and determinism.

## Acceptance criteria

- Given a simple stack, planner produces a clear, deterministic plan.
- Errors are structured and actionable.
- Plan types are stable and suitable for UI display and for execution by T-30.

## Dependencies / sequencing

- Depends on git backend (T-26), worktree service (T-27), and lease enforcement (T-25).
- Execution is implemented in T-30.

