---
id: T-3
stacked_on: T-2
node:
  branch: rn/linear-integration/T-3-sync-from-linear
linear:
  issue_id: 948137cf-96c4-4765-8cd2-5337dfda019c
  identifier: RED-13
---

# T-3 `rn sync --from linear`: label-filtered import + ID allocation + parent selection

## Brief (local)

- Implement pull sync from Linear into a single epic:
  - only issues in the configured Linear project with the epic slug label
  - create/update local task docs and DB tasks
  - allocate stable local ids (`T-###`) for imported issues and persist the association
  - infer topology constraints from blockers:
    - 0 blockers → no parent
    - 1 blocker → set `stacked_on`
    - >1 blockers → interactive parent selection (including “No parent”); store remaining blockers in `must_land_after`

## Acceptance Criteria

- `rn sync --from linear` updates the epic’s task docs under `epics/<slug>/tasks/` and updates DB rows.
- Imported tasks receive stable `T-###` ids, and the mapping to `linear.issue_id` is preserved.
- Multi-blocker issues prompt for a parent choice and complete without error.
- The task doc “Brief (local)” is preserved; only the managed metadata + synced section change.

## Notes / Design

- Use the label == epic slug filter to allow multiple epics to share one Linear project.
- Conflict handling is out of scope; overwrites are acceptable in v0.
