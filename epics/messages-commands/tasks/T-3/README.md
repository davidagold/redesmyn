---
id: T-3
stacked_on: T-2
node:
  branch: rn/messages-commands/T-3-runner-delivery
---

# T-3 Runner + harness integration: delivery modes (hooks/cooperative/manual)

## Brief (local)

- Implement delivery modes per the design:
  - hook-based enrichment (when available)
  - cooperative polling (`rn`/skill-driven) for harnesses without hooks
  - clear manual bridging affordances when attached (last-resort)
- Ensure agent sessions can acknowledge/advance commands consistently across harnesses.

## Acceptance Criteria

- At least one harness can receive messages/commands end-to-end without manual steps.
- Degraded modes are explicit and ergonomic, not silent failure.
