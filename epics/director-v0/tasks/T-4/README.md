---
rn:
  node:
    branch: rn/director-v0/T-4-controller-wake-protocol
  parent: T-1
---

# T-4 Controller <-> director wake protocol + backlog delivery

## Implementation Boundary

- Implement wake protocol and backlog delivery in Rust control-plane + desktop integration surfaces.
- Do not add or rely on legacy Python/webview implementations for this task.

## Plan

- Define significant-event detection for wakeups:
  - which event kinds require waking the director
  - coalescing/deduping behavior while director is busy
- Define wake payload contract:
  - include all available unacknowledged events at wake time
  - deterministic ordering
  - chunking behavior when payload exceeds message budget
- Define director acknowledgement contract:
  - explicit ack cursor returned by director
  - retry/replay behavior when ack is missing or partial
- Define controller safety behavior:
  - no polling requirement in v0
  - bounded queue growth / backlog handling strategy
  - clear wake reasons and observability for debugging

## Acceptance Criteria

- Director can be driven entirely by controller wake messages in v0 (no direct event polling required).
- No events are silently dropped between wake and ack, including large-backlog scenarios.
- Replayed wake deliveries are safe and do not cause unintended duplicate orchestration actions.
- Implementation targets Rust runtime paths (control plane + desktop integration), not legacy paths.
