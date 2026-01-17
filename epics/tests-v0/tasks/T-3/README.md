---
rn:
  node:
    branch: rn/tests-v0/T-3-git-mechanics
  parent: T-1
  after: []
---

# T-3 Git mechanics (unit + integration)

## Goal

Prove correctness of core git mechanics at both:

- integration level (real git repo + real branches/commits), and
- selective unit level for mechanics that benefit from tight feedback.

## Behaviors to validate

At minimum:

1. **Effective base** selection:
   - when a task’s parents are “merged” (ancestor of epic base), new branches/worktrees base off the epic base, not the merged parent’s branch tip
2. **Restack semantics**:
   - restack rebases the target branch and all downstream branches (scope dependent)
   - merge run status and events reflect restack progress and failures
3. **Merge semantics**:
   - merge fast-forwards base through the target (and optionally cascades)
   - merge_then_restack ordering is respected when configured
4. **Resume semantics**:
   - after a conflict/stall, resume revalidates and continues safely

## Preferred entrypoints (don’t re-implement logic in tests)

Drive tests through the same public planning/execution entrypoints used by the CLI/server:

- `redesmyn/git_mechanics_v0.py`:
  - `build_merge_cascade_plan(...)`
  - `execute_merge_cascade_plan(...)`
  - `build_restack_plan(...)`
  - `execute_restack_plan(...)`
- Where useful, also validate the human-facing plan formatting:
  - `format_merge_cascade_plan(plan)` / `format_restack_plan(plan)` (if present/used)

## Proposed tests (names + intent)

These names are suggestions; adjust as needed, but keep them behavior-first.

### Planning

- `test_merge_plan_uses_epic_base_when_parent_is_merged()`
  - Set up: parent task is marked `Done` and its branch tip is an ancestor of epic base (`main`).
  - Assert: `build_merge_cascade_plan()` chooses `upstream_ref == epic.root_branch` for the child, not the merged parent’s branch.
- `test_merge_plan_requires_base_worktree_present()`
  - Assert: if no worktree has the epic base checked out, planning fails with actionable guidance.
- `test_merge_plan_blocks_when_running_agents_and_not_forced()`
  - Assert: running-agent detection (spine/affected tasks) triggers the expected error shape unless `force=True` (or `allow_running` is provided at execution, depending on layer).
- `test_restack_plan_descendants_only_includes_existing_worktrees()`
  - Assert: descendants without worktrees are skipped, but active descendants without worktrees produce an error (if that is current behavior).

### Execution + resume

- `test_execute_merge_conflict_marks_run_blocked_and_can_resume()`
  - Create an intentional conflict mid-plan.
  - Assert: merge run is blocked, and after manual resolution, resume continues and completes.
- `test_execute_merge_then_restack_orders_steps_correctly()`
  - Assert: when configured, the step order matches “rebase spine → merge spine → rebase remaining descendants”.
- `test_resume_revalidates_clean_worktrees_before_continuing()`
  - Assert: resume fails with a clear error if any affected worktree has dirt/in-progress operations.

## Suggested approach

- Use the scenario repo fixture(s) from T-1 to construct small stacks:
  - base branch + 2–4 task branches
  - inject conflicts by editing the same file differently
- Drive the actual mechanics through the same public functions used by the server/CLI (avoid reimplementing logic in tests).

## Acceptance Criteria

- Tests validate both “happy path” and at least one conflict/resume case.
- Tests are readable: scenario variants do the heavy lifting; test bodies focus on assertions.
