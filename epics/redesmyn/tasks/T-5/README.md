---
rn:
  node:
    branch: rn/redesmyn/task-5-worktree-path
  parent: T-3
---

# T-5 Worktrees: record `nodes.worktree_path`

## Plan

- Ensure that when a worktree is created/attached for a node branch, we persist its location in
  `nodes.worktree_path` so the daemon/UI can reason about local state.
- Provide a first-class CLI flow for users/agents to obtain a worktree for a node branch (instead of
  relying on plain `git worktree add`), e.g. `rn shell --print --task-id <task_id>`.
- Provide an ergonomic “go there now” workflow for humans, e.g. `rn shell -e <epic> -t <T-…>`.
- Track the related modeling question: do we want separate `tasks` and `nodes` tables long-term, or
  is the relationship effectively 1:1 for our v0 dogfooding workflows?

## Acceptance Criteria

- After an `rn` command creates or attaches a worktree for a node branch, `nodes.worktree_path` is
  non-null and points at the correct worktree.
- The operation is safe/idempotent: re-running does not create duplicate worktrees or duplicate DB
  state.
- No new UI is required in this task; this is a correctness + observability foundation.

## Notes / Contracts

- Keep `rn git …` as a faithful proxy; the higher-level worktree command should compose on top of it.
- If we decide to keep `tasks` and `nodes` distinct, document the separation-of-concerns in
  `epics/redesmyn/README.md` (and/or `AGENTS.md`) so future agents don’t “simplify” it accidentally.
