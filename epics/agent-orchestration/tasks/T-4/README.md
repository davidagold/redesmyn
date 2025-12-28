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
- The observer is a **client-host daemon capability**: start it from `rn daemon run` (embedded or subprocess), not as a separate “thing users must remember”.

### Final designs

#### A) One observer per repo (no epic scoping)

- Remove `--epic` from the observer CLI/API surface.
- Run one observer per repo, responsible for:
  - ref movement + commit detection for node branches
  - worktree health detection (exists/dirty/mismatch)
- Attribute events to tasks/agents via branch → node mapping:
  - `git.commit` includes `node_id` (and thus task) and `agent_id` when a task agent is running/known.

#### B) Daemon-owned lifecycle

- `rn daemon run` starts the observer automatically (default on for dogfooding).
  - Provide `--no-observer` (or env flag) for debugging.
- Keep `rn observer run` as a debug command only (foreground run, custom interval), but not a required part of normal UX.

#### C) Events after “session → agent” merge

- Observer events should no longer reference “session id”; attribute to `agent_id` (and `task_id` via node mapping) instead.
