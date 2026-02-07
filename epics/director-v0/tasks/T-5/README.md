---
rn:
  node:
    branch: rn/director-v0/T-5-director-session-ui-v0
  parent: T-4
---

# T-5 Director UI v0 (pinned session + controller overlay)

## Plan

- Use the epic-pinned director session as the primary UI (no separate heavy workbench in v0).
- Add automatic-direction controls:
  - composer disabled while automatic direction is active
  - explicit pause/resume automatic direction toggle; composer re-enabled when paused
- Add integrated active-director visual treatment:
  - subtle, structural styling (integrated with layout/chrome)
  - avoid badge-only or duplicate status labels
- Add a small controller overlay in graph view (e.g. corner dock) that shows:
  - queued wake/message count
  - controller state (idle/sending/waiting-ack/error)
  - last wake reason/time
- Keep overlay scope tight:
  - do not duplicate task-card progress/status content already visible in graph

## Acceptance Criteria

- A user can clearly tell when the director session is actively auto-directing the epic.
- Users cannot accidentally send manual messages while auto-direction is active.
- Controller queue/wake state is visible without introducing duplicate task-status surfaces.
