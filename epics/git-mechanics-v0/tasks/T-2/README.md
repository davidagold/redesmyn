---
rn:
  node:
    branch: rn/git-mechanics-v0/T-2-out-of-sync-indicators
  parent: null
---

# T-2 Surface out-of-sync / left-behind branches in the UI

## Brief (local)

Add lightweight detection and UI surfacing for branches that are no longer aligned with the most recent expected timeline (e.g. parent/base advanced due to merges/rebases and the child branch hasn’t been rebased yet).

Constraints:

- v0 detection does not need to be perfect; heuristics are acceptable.
- Prefer cheap checks (e.g. `git merge-base --is-ancestor <parent> <child>` or similar) and avoid expensive history analysis in hot paths.

UI behaviors:

- When a branch is out-of-sync, show a subtle indicator on the node card (and/or in a list/filter) with actionable guidance (“rebase needed”).
- The UI should help the user understand *which* branches are impacted (e.g. “this branch is behind parent X”).
- Avoid adding busy borders; use small, tasteful iconography consistent with existing graph status cues.

## Acceptance Criteria

- The system can identify a set of “out-of-sync” nodes for an epic (best-effort).
- The graph UI surfaces the state unobtrusively and provides a clear next action.
- Detection does not materially degrade dashboard performance.
