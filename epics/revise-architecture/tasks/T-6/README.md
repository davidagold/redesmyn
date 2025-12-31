# T-6 Merge Node into Task (single graph primitive)

## Metadata

```yaml
id: T-6
stacked_on: T-1
node:
  branch: rn/revise-architecture/T-6-merge-node-into-task
```

## Brief (local)

Consolidate “graph node” concepts by merging `Node` into `Task` so the graph has a single primitive:

- Move branch/topology/worktree/assignment metadata into `Task` (or `Task`-scoped companion tables where needed).
- Ensure git-derived “projection” fields used by the UI move with the graph primitive (e.g. `stackInSync` should not remain on a removed `Node` API).
- Migrate foreign keys and payloads that currently point at `nodes.*`:
  - sessions should target tasks
  - events referencing nodes should reference tasks
- Remove the `Node` API surface; the UI operates on tasks and their relationships.
- Preserve existing data through a DB migration; provide an upgrade path for local dev DBs.
- Invest in Alembic infrastructure (needed for this migration and future ones):
  - `just` targets for revision/upgrade/downgrade/history/current
  - a predictable migrations directory and a clear “dev DB” vs “test DB” story

## Acceptance Criteria

- There is no longer a separate “node” concept in public API/UI.
- The data model has a single graph primitive (tasks) with topology/branch metadata.
- Agent sessions and telemetry can be attributed to tasks without indirection.

## Updates

### 2025-12-31

- Recent work added `stackInSync` to the epic graph node payload to surface “left-behind/out-of-sync” child branches.
  Once Node is merged into Task, this field (and its derivation/telemetry) must be moved so the UI/API does not retain a node concept.
