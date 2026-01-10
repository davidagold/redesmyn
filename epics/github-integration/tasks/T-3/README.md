# T-3 PR create/open + PR auto-detection + push semantics

## Metadata

```yaml
id: T-3
stacked_on: T-1
must_land_after:
  - T-2
node:
  branch: rn/github-integration/T-3-github-pr-actions
```

## Brief (local)

- Implement PR operations for task branches:
  - create PR (push first if needed),
  - open PR,
  - auto-detect existing PR.
- Persist PR identity on the task and expose it through API/CLI.
- Add a global “auto force-push” toggle for stack rewrites.

## Acceptance Criteria

- `rn github pr create T-123` (or equivalent) results in:
  - branch pushed to remote,
  - PR created against the task’s effective upstream branch,
  - PR identity persisted on the task.
- `rn github pr open T-123` opens the PR in a browser (or prints the URL in non-interactive contexts).
- Best-effort PR auto-detection links an existing PR for a task branch when present.
- PR title defaults to task title.
- PR body includes a Linear issue link when the task is linked to Linear.
- Base branch selection:
  - uses the parent branch when unmerged,
  - skips merged ancestors (uses effective upstream, consistent with “sync projection” logic).
- Provide a global toggle (GitHub menu / CLI config) controlling whether stack rewrites trigger automatic `git push --force-with-lease`.

## Notes / Design

- v0 may use GitHub REST API; GraphQL can be added later for richer PR state.
- Prefer “force-with-lease” and make the default conservative (off).
