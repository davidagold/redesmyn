# T-7 CLI integration tests

## Metadata

```yaml
id: T-7
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-7-cli-integration
```

## Goal

Validate that key CLI commands work end-to-end against real repo state and the local DB.

## Behaviors to validate

- `rn sync --from local` imports epic/task docs into the DB deterministically.
- `rn shell`:
  - prints worktree path correctly with `--print`
  - refuses nesting by default (unless `--nested`)
  - respects `--no-create`

## Suggested approach

- Use `subprocess` calls to invoke the CLI (`uv run rn ...`) in temp repos/worktrees where appropriate.
- Avoid tests that require interactive prompts (or provide non-interactive flags).

## Acceptance Criteria

- CLI integration tests pass in CI-like environments (non-interactive, deterministic).

