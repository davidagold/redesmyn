---
rn: {}
---

# T-1 Rebase: update worktree-bound branch refs

## Background

During `rn merge`, we use `git rebase --update-refs` to rebase a task branch onto the epic base branch, then fast-forward the base branch. This generally works, but there is an important caveat in multi-worktree setups:

- When a branch is **checked out in another worktree**, Git treats its ref as “in use” and it may not be movable by `--update-refs`.
- As a result, some ancestor task branches can remain pointing at pre-rebase commits and appear “off the main line” even though their changes are incorporated into the merged leaf.

This creates ambiguity and paper-cuts:

- The UI branch graph can look “split”, even though the merged leaf fast-forwarded cleanly.
- Any logic that relies on branch ref ancestry can produce confusing results.

## Plan

Design and implement an approach that ensures task branches that are checked out in worktrees are updated consistently during rebase workflows.

The solution must be explicit and safe:

- No mysterious destructive actions.
- Refuse to proceed (with actionable guidance) if preconditions aren’t met.
- Treat uncommitted work as a first-class concern: never discard it silently.

## Candidate approaches

1. **Stricter preconditions**
   - Require that no branches in the affected stack are checked out in other worktrees (or require agents to be stopped).
   - Pros: simple.
   - Cons: interrupts dogfooding; defeats the “run many agents concurrently” workflow.

2. **Cascade rebase down the stack**
   - Instead of rebasing only the leaf with `--update-refs`, rebase the path from root → leaf in order (each branch onto its parent), using each branch’s own worktree when available.
   - For branches without an existing worktree, create a temporary worktree, rebase, then remove it.
   - Pros: keeps checked-out refs movable because the rebase runs in the worktree that owns them; preserves a coherent stack.
   - Cons: more steps; requires careful preflight checks across multiple worktrees.

3. **Post-rebase reconciliation**
   - After rebasing the leaf, compute new commits corresponding to ancestor branch tips and update those checked-out worktrees (e.g. via `git reset --hard <new_tip>`).
   - Pros: could be faster.
   - Cons: requires a robust mapping from old commits → new commits; easy to get wrong.

## Acceptance Criteria

- After `rn merge` (or a dedicated “rebase stack” command), all branches on the merged path point at the rebased commits, even if some were checked out in worktrees.
- The workflow is safe under concurrent dogfooding:
  - cleanly errors if any involved worktree is dirty
  - can be run repeatedly (idempotent)
- Documentation clearly describes the behavior and constraints.
