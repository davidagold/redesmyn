---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-8-generic-target-actions
rn:
  parent: T-7
---

# T-8 Generic target actions (`activate`, `set_toggle`, `set_text`, `send_keys`, `scroll`)

## Problem

Without generic target actions, automation requires bespoke per-feature protocol additions and cannot scale to evolving UI.

## Goal

Add generic target-addressed action primitives over the target registry.

## Scope

1. Add target action requests/responses to protocol.
2. Implement target resolution + action execution.
3. Support at minimum:
   - activate/click-like intent,
   - set toggle/select value,
   - set/append text,
   - send keys,
   - scroll container/viewport.
4. Ensure actionable errors for unsupported actions/invalid target states.

## Dependencies / sequencing

- Depends on T-7.

## Acceptance criteria

- Scripts can manipulate settings and scrollable surfaces via target IDs.
- No coordinate-based fallback is required.

## Validation

- Integration tests covering each generic action type.
