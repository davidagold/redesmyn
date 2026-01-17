---
rn:
  node:
    branch: rn/github-integration/T-4-dashboard-github-ui
  parent: T-3
  after: []
---

# T-4 Dashboard UX: icon-only integration indicators + task-card PR badge

## Brief (local)

- Add GitHub to the dashboard’s integration affordances.
- Update the existing Linear affordance to match the new pattern (icon-only).
- Surface per-task PR state above the task card.

## Acceptance Criteria

- Top-right integration cluster:
  - Linear indicator is icon-only (no “Linear” text, no chevron).
  - Add a GitHub icon-only button that opens a menu with:
    - connect/disconnect/status
    - repo association shortcut (epic-level)
    - “auto force-push” toggle
- Task cards show a GitHub badge next to the Linear badge:
  - icon + PR number,
  - border color communicates PR state (open/draft/merged/closed; v0 can start with open vs merged vs closed).
- Clicking the GitHub badge opens the PR.

## Notes / Design

- Prefer small, non-noisy UI: integration affordances should not compete with run/stop controls.
- Avoid chevrons for icon-only menus; treat the icon itself as the button.
