---
rn:
  node:
    branch: rn/messages-commands/T-4-dashboard
  parent: T-3
---

# T-4 Dashboard: per-node thread + command issuance (graph-first)

## Plan

- Add a per-node message thread to the selection Details panel.
- Add command issuance UI from the same context (and show command state).
- Integrate live updates via WebSocket events without requiring manual refresh.

## Acceptance Criteria

- Users can send a message/command from a node selection and see responses/state changes live.
- UI remains graph-first and avoids “inert property dumps”.
