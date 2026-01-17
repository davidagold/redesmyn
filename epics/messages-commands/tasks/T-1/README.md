---
rn:
  node:
    branch: rn/messages-commands/T-1-design
  parent: null
---

# T-1 Design: messaging + commands model + UX

## Brief (local)

- Do a dedicated design pass for:
  - message threading model (by session, by node, by epic; history + retention)
  - command types/payloads (typed Pydantic models + enums) and lifecycle
  - delivery modes (hooks vs cooperative polling vs manual bridging)
  - graph-first UI integration (Details panel, activity indicators, command issuance)
- Define how messages/commands appear on the event stream (namespaces + payload shapes), keeping the contract versionable and additive.

## Acceptance Criteria

- The control doc is updated with the agreed v0 data model and UX.
- Clear “degraded mode” behavior is specified for harnesses without hooks.
