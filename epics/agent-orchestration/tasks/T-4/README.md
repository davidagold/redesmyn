# T-4 Repo observer: commits + worktree telemetry → events

## Metadata

```yaml
id: T-4
stacked_on: T-3
node:
  branch: rn/agent-orchestration/T-4-repo-observer-telemetry
```

## Brief (local)

- Implement a host-local observer that emits normalized events for:
  - node branch ref movements and new commits
  - worktree health (path exists, clean/dirty, branch mismatch)
- Persist events for Timeline and stream them live to the UI.
- Keep it robust when git changes occur outside `rn` (best-effort detection).

## Acceptance Criteria

- A new commit on a node branch produces an event that can drive a tasteful “activity pulse” in the graph UI.
- Worktree changes (dirty/clean, missing path) are observable and attributable to a node.

## Updates

- Added `rn observer run` (polling) to emit `git.commit` + `worktree.health` events into `events`.
- Observer adopts an existing worktree path into `nodes.worktree_path` when the branch matches.
- Treat observation as “one process per repo” (daemon capability): remove the need for `--epic`-scoped observation in normal usage.
- Prefer `rn dev` / `rn daemon run` to auto-start observation for local dogfooding so users don’t have to learn a separate observer lifecycle.
