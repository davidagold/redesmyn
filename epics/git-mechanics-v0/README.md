---
slug: git-mechanics-v0
name: Git Mechanics v0
root_branch: main
linear:
  project_id: null
---

# Git Mechanics v0 Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Git Mechanics v0** epic: intent, sequencing, and key decisions. Keep it current.

## 1) Vision

Make Redesmyn’s “merge/sync” experience safe-enough and ergonomic for stacked task branches managed via git worktrees.

Key goals:

- Provide a **stack-preserving** “merge task” workflow that can cascade through ancestors + downstream branches while operating **inside each worktree** (avoid ref updates failing because branches are checked out elsewhere).
- Tighten safety rails: check for dirty worktrees, in-progress rebases/merges, and batch confirmations for potentially dangerous operations (e.g. running agents).
- Improve dashboard affordances for “ready to merge” and merging from the graph UI.
- Surface when branches are “left behind” (no longer stacked on the most recent parent/base timeline), without requiring perfect detection.

## 2) Key decisions

### 2.1 Cascade operates per-worktree (not via `--update-refs`)

Git cannot move a branch ref that is checked out in another worktree. Therefore, cascade operations must run git commands **in the worktree that has that branch checked out**.

### 2.2 Preserve the stack invariant

When cascading, each branch is rebased onto its **immediate parent branch** (or the epic base for roots), not directly onto the epic base. This preserves stacked semantics across the DAG, including side branches.

### 2.3 Batch confirmations for dangerous fanout

If an operation affects multiple branches with running tasks (agents), prompt once with a summary and require a single confirmation (CLI `--yes` / UI confirm modal).

## 3) Scope (v0)

- CLI: stack-preserving `rn merge --cascade` (and `--dry-run`, `--yes`) using per-worktree operations.
- Server: endpoint(s) that can execute merge/cascade and stream progress via the existing websocket event channel.
- Dashboard: graph-level actions for “Ready to merge” and “Merge/Merge stack”.
- Out-of-sync detection + UI surfacing for branches that no longer track the latest parent/base timeline.

## 4) Non-goals (this epic)

- Perfect synchronization guarantees while agents are actively writing (v0 assumes the user ensures agents are quiescent; we still enforce dirty-worktree checks).
- Fully automatic conflict resolution.
- Comprehensive git-history provenance mapping across rebases (heuristics are acceptable for v0 “out-of-sync” indicators).

## 5) Task map

- `epics/git-mechanics-v0/tasks/T-1/README.md`: Stack-preserving cascade merge (CLI + API + dashboard actions).
- `epics/git-mechanics-v0/tasks/T-2/README.md`: Detect + surface “out-of-sync / left-behind” branches in the UI.
- `epics/git-mechanics-v0/tasks/T-3/README.md`: Abort merge runs (cancel) + future rollback design.
- `epics/git-mechanics-v0/tasks/T-4/README.md`: Skip rebasing already-merged/done spine tasks when merging.
- `epics/git-mechanics-v0/tasks/T-5/README.md`: Squash merge mode (one commit per spine task; leave work branches intact).
