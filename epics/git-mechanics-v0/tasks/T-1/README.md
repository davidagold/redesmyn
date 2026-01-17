---
rn:
  node:
    branch: rn/git-mechanics-v0/T-1-cascade-merge
  parent: null
---

# T-1 Stack-preserving cascade merge (CLI + API + UI)

## Brief (local)

Implement a stack-preserving “merge” workflow that works with git worktrees and can safely (as safely as possible in v0) cascade updates through the stack.

Core requirements:

- **Operate in each worktree** for any branch being rebased/merged (do not rely on `git rebase --update-refs`).
- Preserve stack semantics:
  - Rebase each node branch onto its **immediate parent branch** (or epic base branch for roots).
  - When merging a task, include unmerged ancestors on the spine; after rebasing, fast-forward the epic base branch as appropriate.
  - Include side branches in the affected set so the overall stack remains coherent.
- Safety rails:
  - Hard fail if any involved worktree is dirty.
  - Hard fail if any involved worktree is mid rebase/merge.
  - Detect running agents in the affected set; require a **single batched confirmation** (CLI flag like `--yes` / UI modal).
  - Provide a `--dry-run` / plan preview (CLI + UI) showing what will happen and which branches/worktrees are involved.
- Reuse a unified backend procedure for “achieve desired git state” and websocket event emission (so single-task and bulk behaviors stay consistent).

Dashboard UX requirements (graph-first):

- Expose “Ready to merge” and “Merge / Merge stack” directly from the graph UI (not buried in the details panel).
- Use a hover/selection affordance (e.g. ellipsis menu) that avoids visual noise when not interacting.
- “Ready to merge” should be an action (menu item), not a hidden switch.
- “Merge” actions should be disabled until ready (with guidance/tooltips).

## Acceptance Criteria

- CLI supports `rn merge --task <id> --cascade` with `--dry-run` and `--yes`.
- Cascade preserves stack semantics across both the spine and descendants/side branches.
- Cascade enforces safety checks for dirty worktrees and in-progress rebases/merges.
- Running-agent confirmation is batched (one prompt/modal per request).
- Dashboard exposes “Ready to merge” and “Merge/Merge stack” from the graph node UI.
- Progress is reflected in the UI as steps execute (via websocket events).
