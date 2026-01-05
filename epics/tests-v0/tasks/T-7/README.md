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
- `rn sync --from local --no-create-branches`:
  - still syncs **task topology** (e.g., `stacked_on` → `parent_task_id`)
  - does **not** assign/create branches
- `rn merge` / `rn restack`:
  - prints a plan and prompts for confirmation by default
  - supports a non-interactive `-y` path

## Suggested approach

- Use `subprocess` calls to invoke the CLI (`uv run rn ...`) in temp repos/worktrees where appropriate.
- Avoid tests that require interactive prompts (or provide non-interactive flags).

## Proposed tests (names + intent)

- `test_sync_from_local_imports_tasks_and_sets_parent_links_without_creating_branches()`
  - Run `rn sync --from local --no-create-branches` and assert `parent_task_id` is populated for stacked tasks while `branch_name` remains null.
- `test_shell_print_outputs_worktree_path_and_exits_zero()`
- `test_shell_refuses_nesting_by_default()`
- `test_merge_prompts_for_confirmation_and_supports_yes_flag()`
  - Exercise `rn merge --dry-run` with and without `-y` in a non-interactive context.
- `test_restack_prompts_for_confirmation_and_supports_yes_flag()`

## Acceptance Criteria

- CLI integration tests pass in CI-like environments (non-interactive, deterministic).
