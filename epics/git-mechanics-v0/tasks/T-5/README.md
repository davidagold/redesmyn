---
rn:
  node:
    branch: rn/git-mechanics-v0/T-5-squash-merge-mode
  parent: null
---

# T-5 Squash merge mode (one commit per spine task)

## Problem

Today, merge runs fast-forward the epic base branch to include the spine branch tips. This preserves the exact commit history from each work branch on the base branch.

Some users want a cleaner base-branch history:

- Work branches often contain WIP / fixup / experiment commits.
- A stacked spine can produce many small commits on the base branch.
- Reviewing the base branch history becomes noisy compared to “one commit per task”.

## Goal

Add an **optional squash mode** for merge runs:

- When enabled, merging produces **one new commit per (active) spine task branch** on the epic base branch.
- The original work branches remain **unchanged** (no rebase/force-push/rewriting).
- The rest of the merge run semantics should remain aligned with v0 expectations (worktree-aware, safety checks, progress streaming, etc.).

This is a *mode*, not a replacement: the default stays the current fast-forward behavior.

## Desired semantics

### 1) API and CLI surface

- Extend merge requests to include a strategy (name TBD):
  - `merge_mode: "ff" | "squash"` (default `"ff"`)
  - Alternatively: `squash: bool = False`
- CLI: `rn merge --task <id> --cascade --squash` (or `--merge-mode squash`).
- Dashboard: a toggle in the “Configure” area that applies to merge actions in the current session (persisting later is optional).

### 2) What gets squashed

When merging task **T**:

- Compute the same **merge spine** and **active spine** as the existing merge planner (respecting `TaskState.Done` as “merged” proxy).
- For each active spine task **S** (root → leaf):
  - Produce a squash commit on the epic base branch that represents **only the diff introduced by S relative to its spine parent**.
  - Skip tasks without `branch_name` (cannot be merged) and do not mark them “done” implicitly.

This implies *ordering matters*: apply squashes in **spine order** so each patch applies to a base branch whose tree matches the parent’s tree.

### 3) Leaving work branches intact

Squash mode must not rewrite branch refs. In particular:

- Do not `rebase` or `reset` the work branches being merged.
- Avoid creating merge commits into the work branches.
- If a task needs follow-up work after squash, it should happen on a new branch (out of scope for v0, but the mode should not make that impossible).

## Git operation options (investigation required)

We need a deterministic way to create “one commit per spine branch diff” while operating from the base branch worktree.

### Option A (preferred): patch-apply + commit (per task)

For a spine task branch `S` with parent ref `P`:

- Compute a patch representing **exactly** `P..S`:
  - `git diff --binary P..S` (or `P...S` if that’s more correct for our stacking model; decide explicitly).
- Apply to the base worktree:
  - `git apply --index --3way` (3-way can help, but requires careful failure handling).
- Commit with a task-derived message:
  - e.g. `T-12: <title>` + possibly a footer referencing the original branch name / tip SHA.

Pros: keeps work branches untouched, preserves “one commit per branch” intent, works without checking out each branch.

Cons: conflict UX differs from rebase/merge; we need to define failure + recovery semantics carefully.

### Option B: `git merge --squash` (likely insufficient for per-branch commits)

`git merge --squash S` computes changes from the merge-base between base and `S`. For stacked spines, that likely yields a **cumulative** diff (base → leaf), which doesn’t match “one commit per spine task”.

This may still be useful if we decide squash mode should be “one commit for the entire active spine”, but that is a different feature.

## Data model / invariants (important)

Today we treat “merged” as:

- `TaskState.Done` in the DB, and
- a runtime safety check that `branch_name` is an ancestor of the epic base branch (see restack/merge consistency checks).

With squash merges, the branch tip will **not** be in the base branch history, so we must define a new invariant.

Options to explore:

- Add merge provenance fields on `Task` (or `MergeRunTask`):
  - `merged_via: "ff" | "squash"`
  - `merged_base_sha: str` (the base-branch commit created for this task)
  - `merged_from_sha: str` (the work-branch tip at merge time)
- Replace “ancestor check” for done tasks with:
  - a patch-id based check, or
  - a “tree equivalence / no-diff” check between the task’s expected content and base at merge time.

This task should record the chosen invariant and update any code paths that assume ancestry (merge planning, restack planning, out-of-sync detection, etc.).

## Acceptance criteria

- Merge requests can select `"squash"` mode (API + CLI; UI toggle can be follow-up if needed).
- Merging a simple 3-task spine produces **three commits** on the epic base branch, one per task, with clear commit messages.
- Work branch refs are unchanged after the merge run.
- The system has a clear and enforceable “done/merged” invariant that remains correct under squash mode (documented and implemented).
- Existing ff-only behavior remains the default and unchanged unless squash mode is explicitly enabled.
