# T-2 Repo selection everywhere: `-C/--repo` plumbing

## Metadata

```yaml
id: T-2
stacked_on: T-1
node:
  branch: rn/v0-launch/T-2-repo-selector
```

## Plan

Once the server/daemon split is strict, many commands must be runnable from outside the target repo (especially
`rn daemon up`, `rn up`, and “status” style commands). The revise-architecture control doc calls this out explicitly:
repo selection should not be “whatever directory your shell is in”.

### Work

1) Define the repo selector UX

- Add a global repo selector option (preferred shape: `rn -C /path/to/repo …`, matching git’s `-C` mental model).
- Decide whether the selector is:
  - a path to any directory inside a git worktree, or
  - a strict “repo root path only”.
  For v0, “any directory inside the repo” is usually friendlier.

2) Plumb it through all repo-scoped commands

- Ensure every command that currently does `get_repo_context()` can instead target the selected repo.
- Make `rn server …` and `rn daemon …` accept the same selector so they talk about the same repo identity and DB.
- Ensure `rn up/down` uses this selector as its primary input.

3) Make errors unambiguous

- “Not a git repo” (selector points somewhere invalid)
- “Repo not initialized” (needs `rn init` or auto-init semantics—decide later, but message must be explicit)
- “Daemon not connected/attached” (needs `rn daemon up` / attach)

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
