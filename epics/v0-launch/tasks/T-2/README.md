# T-2 Repo selection everywhere: `-C/--repo` plumbing

## Metadata

```yaml
id: T-2
stacked_on: T-1
node:
  branch: rn/v0-launch/T-2-repo-selector
```

## Brief (local)

Enable running `rn` from outside the repo by introducing a single repo selector and plumbing it through the CLI:

- Add a global `-C/--repo` option (exact flag name TBD) that selects the target repo root.
- Ensure all repo-scoped commands use the selected repo (server start, daemon attach, status, task operations).
- Keep behavior unchanged when run inside a git repo (repo defaults from `cwd`).

## Acceptance Criteria

- From any directory, `rn -C /path/to/repo status` works.
- `rn server run -C /path/to/repo …` and `rn daemon run -C /path/to/repo …` target the same repo identity and DB path.
- Errors clearly distinguish:
  - “not a git repo”
  - “repo not initialized”
  - “daemon not connected / not attached”

