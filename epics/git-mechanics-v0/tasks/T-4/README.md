---
rn:
  node:
    branch: rn/git-mechanics-v0/T-4-skip-done-spine
---

# T-4 Skip rebasing already-merged/done spine tasks when merging

## Problem

During merge planning, we currently treat “merge a task” as “rebase the entire spine (root → leaf)”, even when some ancestor tasks on the spine are already **merged/marked `done`**.

This causes avoidable problems:

- Unnecessary rebases and potential conflicts (especially format-only churn).
- Merge runs failing because a `done` task may no longer have an active worktree (or we don’t want to touch it).
- Confusing UX: a merged task appears to be “actively participating” in a merge run.

We should treat `done` spine tasks as effectively “collapsed into” the epic base branch for purposes of merge planning.

## Desired semantics

Given a merge request for task **X**:

### 1) Compute spine + classify nodes

- Compute the spine node list `root → X` (inclusive).
- Split into:
  - **`merged_spine`**: contiguous prefix of spine nodes whose primary task is `done`.
  - **`active_spine`**: remaining spine nodes (including X) whose primary task is *not* `done`.

Optional safety check (recommended):

- If a spine task is `done`, but its branch tip is not contained in the epic base branch history, treat this as **inconsistent state** and fail with a clear error + remediation. (Alternative: treat it as not-done and include it in `active_spine`.)

### 2) Build plan steps

- Do **not** create `rebase` steps for `merged_spine` nodes.
- For the first node in `active_spine`:
  - Rebase onto the epic base branch (not onto its immediate parent, since that parent is “collapsed”).
- For subsequent nodes in `active_spine`:
  - Rebase onto their immediate parent node branch (existing stack-preserving behavior).

Fast-forward steps (as in T-1) remain unchanged: a merge run still merges the spine into the epic base branch; this task only changes which spine nodes participate in the rebase phase.

### 3) Safety rails and gating

- Merge-ready gating applies only to `active_spine` nodes (not `merged_spine`).
- Dirty-worktree and in-progress git-operation checks apply only to worktrees the plan will actually touch.
- Running-agent batching should consider only the tasks/worktrees that will be affected by the requested merge strategy.

## UI expectations

No dedicated UI changes required, but plan previews (CLI + dashboard) should make it obvious that `done` ancestors are skipped (i.e. the first rebase target becomes the epic base branch).

## Acceptance Criteria

- Merge plan for a task with `done` ancestors does not include rebase steps for those ancestors.
- The first non-done spine node rebases onto the epic base branch.
- Merge-ready gating and safety checks do not consider `done` spine nodes.
- Behavior is consistent across:
  - Merge (no descendants)
  - Merge and Restack (strict)
  - Merge then Restack (relaxed)
