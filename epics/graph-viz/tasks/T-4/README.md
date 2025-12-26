# T-4 Commit strings (API + data model)

## Metadata

```yaml
id: T-4
stacked_on: T-1
node:
  branch: rn/graph-viz/task-4-commit-strings-api
```

## Brief (local)

- Extend the daemon API so the UI can treat “commit strings” as first-class:
  - Node work ranges (`parent..branch`) with counts + identifiers.
  - Edge-level commit sequence metadata for parent → child edges.
- Keep this minimal but future-proof: we should be able to progressively enrich the commit model.

