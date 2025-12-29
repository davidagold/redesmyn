# T-4 `rn sync --to linear`: create/update issues + overwrite semantics

## Metadata

```yaml
id: T-4
stacked_on: T-3
node:
  branch: rn/linear-integration/T-4-sync-to-linear
```

## Brief (local)

- Implement push sync from local task docs/DB to Linear:
  - create Linear issues for tasks without `linear.issue_id`
  - update linked issues with naive overwrite (title/description/state/dependencies)
  - apply the epic slug label to created/updated issues
- Push dependency mapping:
  - `stacked_on` and `must_land_after` become “blocked by” edges (best-effort)

## Acceptance Criteria

- `rn sync --to linear` creates issues for local-only tasks and writes back `linear.issue_id` + `linear.identifier` to the task doc metadata.
- Re-running `rn sync --to linear` is idempotent: linked issues are updated, not duplicated.
- Task state is pushed using the coarse mapping (todo/in_progress/blocked/done).
- Dependency edges are pushed as “blocked by” edges.

## Notes / Design

- v0 overwrite model intentionally punts on conflicts; record known conflict situations in the epic control doc.
- Branch naming should remain stable; do not auto-rename branches on title edits.

