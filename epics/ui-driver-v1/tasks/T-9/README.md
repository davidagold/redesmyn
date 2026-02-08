---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-9-multi-channel-script-orchestration
rn:
  parent: T-3
  after:
    - T-1
---

# T-9 Multi-channel script orchestration (`ui_driver` + `control_plane`)

## Problem

End-to-end automation often needs both UI intents and domain-level control-plane calls; today this composition is awkward and blurs boundary ownership.

## Goal

Extend script runner to orchestrate multiple explicit channels while preserving ownership boundaries.

## Scope

1. Add `channel` routing in script steps.
2. Support control-plane request execution alongside UI-driver requests.
3. Provide shared wait/assert/report semantics across channels.
4. Keep logs/results explicit about channel used per step.

## Dependencies / sequencing

- Depends on T-3.
- After T-1 (ownership matrix).

## Acceptance criteria

- Single script can safely mix UI-driver and control-plane steps.
- Channel ownership is explicit and validated.

## Validation

- End-to-end script tests covering cross-channel workflows.
