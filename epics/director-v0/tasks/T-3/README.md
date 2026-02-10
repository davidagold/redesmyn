---
rn:
  node:
    branch: rn/director-v0/T-3-gating-policy
  parent: T-1
---

# T-3 Gate policy + caching (as commands)

## Status

- **Deferred from `director-v0` delivery scope.**
- **Do not implement/start this task as part of `director-v0`.**
- Keep this task as a placeholder/spec stub for a follow-up epic or `director-v1`.

## v0 Scope Note

- `T-3` is intentionally out-of-scope for Director v0.
- Director v0 should ship with gates excluded; this task exists only to preserve the future contract.
- If execution planning tooling proposes `T-3` during v0, treat that as a planning mismatch and skip it.

## Implementation Boundary

- Implement gate command semantics, cache keys, and durable outputs in Rust command/control-plane paths.
- Do not add or rely on legacy Python/webview implementations for this task.

## Plan (Deferred)

- Capture requirements and interfaces only; do not implement in `director-v0`.
- Keep future shape constrained to command/event contracts:
  - gate policy schema and cache key contracts,
  - durable command outputs and review-consumable results,
  - integration points with director wake protocol.

## Acceptance Criteria (Deferred)

- `director-v0` can ship without this task.
- Deferred scope is documented clearly enough that follow-up implementation can start without re-discovery.
- Implementation targets Rust runtime paths (command engine + control plane), not legacy paths.
