---
rn:
  node:
    branch: rn/linear-integration/T-5-dashboard-linear-sync
  linear:
    issue_id: 2be4144e-5201-461e-8982-6e0107ab57ed
    identifier: RED-15
  parent: T-2
---

# T-5 Dashboard: unified “Linear” sync menu button (epic + task)

## Brief (local)

- Add a unified “Linear” sync menu button that can be shown at:
  - epic level (graph header/subheader)
  - task level (details panel for the selected task)
- The button should embed status (connected/disconnected) without adding busy UI.
- Provide one-click actions:
  - Connect / Disconnect
  - Sync from Linear / Sync to Linear
  - Open in Linear (project / issue)

## Acceptance Criteria

- Epic view shows a single, tasteful sync control with status embedded.
- Task details panel shows the same control (contextual to the task).
- Actions call through to the backend/CLI wiring with clear success/error feedback.

## Notes / Design

- Follow dashboard UI conventions: avoid extra borders and inert property lists; keep the menu short and action-oriented.
