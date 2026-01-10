# T-5 Nice-to-have: PR checks/review/close affordances

## Metadata

```yaml
id: T-5
stacked_on: null
must_land_after:
  - T-4
node:
  branch: rn/github-integration/T-5-pr-status-details
```

## Brief (local)

- Extend PR integration with richer state:
  - checks summary (pass/fail/pending),
  - review state,
  - close PR affordance (not merge).

## Acceptance Criteria

- Dashboard can display a compact, non-noisy PR status summary on the task card and/or in the details panel.
- Actions remain constrained: v0 does not attempt “merge PR”.
