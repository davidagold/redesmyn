# T-3 Git mechanics (unit + integration)

## Metadata

```yaml
id: T-3
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-3-git-mechanics
```

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

## Suggested approach

- Use the scenario repo fixture(s) from T-1 to construct small stacks:
  - base branch + 2–4 task branches
  - inject conflicts by editing the same file differently
- Drive the actual mechanics through the same public functions used by the server/CLI (avoid reimplementing logic in tests).

## Acceptance Criteria

- Tests validate both “happy path” and at least one conflict/resume case.
- Tests are readable: scenario variants do the heavy lifting; test bodies focus on assertions.

