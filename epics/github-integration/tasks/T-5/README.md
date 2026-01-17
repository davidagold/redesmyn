---
rn:
  node:
    branch: rn/github-integration/T-5-pr-status-details
  parent: null
  after:
  - T-4
---

# T-5 Nice-to-have: PR checks/review/close affordances

## Brief (local)

- Extend PR integration with richer state:
  - checks summary (pass/fail/pending),
  - review state,
  - close PR affordance (not merge).

## Acceptance Criteria

- Dashboard can display a compact, non-noisy PR status summary on the task card and/or in the details panel.
- Actions remain constrained: v0 does not attempt “merge PR”.
