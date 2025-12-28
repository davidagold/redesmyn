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
- Migrate foreign keys and payloads that currently point at `nodes.*`:
  - sessions should target tasks
  - events referencing nodes should reference tasks
- Remove the `Node` API surface; the UI operates on tasks and their relationships.
- Preserve existing data through a DB migration; provide an upgrade path for local dev DBs.

## Acceptance Criteria

- There is no longer a separate “node” concept in public API/UI.
- The data model has a single graph primitive (tasks) with topology/branch metadata.
- Agent sessions and telemetry can be attributed to tasks without indirection.
