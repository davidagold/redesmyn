---
id: T-20
stacked_on: T-19
node:
  branch: rn/agent-orchestration/T-20-merge-complete
---

# T-20 Merge UX: complete styling + auto-mark merged ancestors

## Brief (local)

Polish the merge workflow and make completion clearer:

- Style completed tasks in the graph in a tasteful way so it’s obvious they’re done.
- After `rn merge`, mark the merged task as complete, and also mark any other tasks whose
  branches are in the merged branch’s history as complete.

## Acceptance Criteria

- Completed tasks are visually distinct in the graph without making the UI feel busy.
- `rn merge` updates task completion state in the DB after a successful fast-forward.
