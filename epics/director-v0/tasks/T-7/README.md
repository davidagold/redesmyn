---
rn:
  node:
    branch: rn/director-v0/T-7-direction-overlay-queues
  parent: T-4
---

# T-7 Direction overlay + queue projections

## Implementation Boundary

- Implement overlay and queue projections in Rust GPUI/control-plane surfaces only.
- Do not add or rely on legacy Python/webview implementations for this task.

## Plan

- Implement a read-only direction overlay in graph/session context.
- Add event wake queue surface:
  - pending/unacknowledged count
  - controller state (idle/sending/waiting-ack/error)
  - expandable recent event history preview
- Add merge queue preview surface:
  - ordered candidates and current state summary
  - blocked reason preview where applicable
- Keep overlay behavior simple in v0:
  - no auto-collapse
  - no mutating controls in overlay
- Define projection contracts and wiring:
  - event queue projection from controller/wake subsystem
  - merge queue projection from merge queue model
  - timestamp/order guarantees for deterministic rendering

## Acceptance Criteria

- Overlay exposes wake queue and merge queue information without duplicating task-card status UI.
- Queue surfaces are read-only and remain legible under normal graph interactions.
- Projection updates are deterministic and reflect durable control-plane state.
- Implementation targets Rust runtime paths (desktop + control plane), not legacy paths.
