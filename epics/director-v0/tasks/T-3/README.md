---
rn:
  node:
    branch: rn/director-v0/T-3-gating-policy
  parent: T-1
---

# T-3 Gate policy + caching (as commands)

## Plan

- Define “gates” as first-class commands:
  - gate policy schema (which gates; on which repo/stack; required vs advisory)
  - executor selection (local vs remote)
  - durable status and results (events + artifacts)
- Define a caching strategy keyed by:
  - candidate ref (usually commit SHA)
  - base ref used (merge-base or trunk)
  - policy version/config
- Define when gates are applied:
  - manual trigger vs automatic
  - “parsimonious gating” strategy (avoid re-running expensive gates unnecessarily)

## Acceptance Criteria

- Gate execution is representable without new bespoke “gate runner” subsystems (reuse command engine + events).
- Cached gates can be reused safely when the key inputs are unchanged.
- The conductor can understand which gates ran, where, and why a gate is considered valid.
