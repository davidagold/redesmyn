# T-4 Commit strings (API + data model)

## Metadata

```yaml
id: T-4
stacked_on: T-1
node:
  branch: rn/graph-viz/task-4-commit-strings-api
```

## Brief (local)

- Make “commit strings” first-class data, not an afterthought. The UI needs enough structured information to render and inspect commit sequences along nodes/edges.
- Extend the daemon API so the UI can treat “commit strings” as first-class:
  - Node work ranges (`parent..branch`) with counts + identifiers.
  - Edge-level commit sequence metadata for parent → child edges.
- Keep this minimal but future-proof: we should be able to progressively enrich the commit model.

## Acceptance Criteria

- The dashboard can obtain (for an epic) commit-string metadata sufficient to:
  - Display a commit count per node work range.
  - Display a commit count (or equivalent summary) per parent → child edge.
- The data model is designed for progressive disclosure:
  - v0 graph fetch returns summary fields (counts + head/base SHAs).
  - A follow-up fetch can return detailed commit lists only when needed (e.g., on edge selection), so large graphs remain fast.
- Git computation is correct relative to the epic’s branch graph:
  - Work range is `parent..branch` (multi-commit allowed).
  - Missing branches/refs degrade gracefully with explicit “unknown/unavailable” signals.

## Notes / Contracts

- Prefer stable identifiers in API responses (commit SHA strings) and avoid duplicating heavy commit payloads everywhere.
- Keep the v0 surface area minimal, but pick shapes that won’t force a breaking change when we add:
  - commit metadata (author, timestamp, subject)
  - rebase lineage / historical views
