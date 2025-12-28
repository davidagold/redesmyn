# T-7 Dashboard: agent presence + activity integrated into the graph

## Metadata

```yaml
id: T-7
stacked_on: T-6
node:
  branch: rn/agent-orchestration/T-7-dashboard-graph-presence
```

## Brief (local)

- Render agent/session presence directly on graph nodes:
  - assigned/unassigned
  - running/idle/blocked/error
  - recent activity indicators (commit/worktree pulse)
- Consume WebSocket updates and reflect them in the graph UI in a tasteful way.

## Acceptance Criteria

- Nodes visually communicate “is someone working on this, and are they healthy?” at-a-glance.
- Activity indicators do not add noisy borders or busy UI; use subtle emphasis consistent with existing styling.

## Updates

- Graph nodes render agent + session presence (status dot + session state) and subtle activity pulses from streamed events.
- `EpicGraphResponse` now includes active `sessions` so the dashboard can render session state without extra requests.
- Node cards should be task-first (no “Node <id>” in the UI): remove node ids from the card UI as nodes are not user-facing.
- Adjust node card layout: top-align text content, avoid concatenated ID/title repeats, and prefer spacing over extra borders.
- Use a single status affordance: a colored status circle in the upper-right with a tooltip (status label + uptime).
- Surface the assigned agent as a “resource tag” aligned to the bottom of the card (small border radius; distinct from regular tags).
