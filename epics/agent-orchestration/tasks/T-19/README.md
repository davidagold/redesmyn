---
rn:
  node:
    branch: rn/agent-orchestration/T-19-merge
  parent: T-18
---

# T-19 Merge workflow: “ready to merge” + `rn merge`

## Brief (local)

Support a simple, explicit “merge a stacked task branch” workflow:

- Add a “ready to merge” state on tasks (manual toggle in the dashboard).
- Add `rn merge --task <db_id>`:
  - rebase the task branch onto the epic root branch (using `--update-refs`)
  - fast-forward the epic root branch to the task branch head (`--ff-only`)
- `rn merge` must refuse to run unless the task is ready, unless `--force` is provided.

## Acceptance Criteria

- Users can mark a task ready in the dashboard and run `rn merge --task <id>`.
- `rn merge` fails safely and explains how to proceed when preconditions aren’t met.
- The workflow supports stacked task graphs (rebasing and ref-updates for downstream refs).
