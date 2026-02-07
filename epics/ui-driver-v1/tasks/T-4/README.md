---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-4-shared-semantic-operations
rn:
  parent: T-1
---

# T-4 Shared semantic operation layer (user + UI-driver parity)

## Problem

User keyboard/mouse flows and UI-driver flows currently diverge in some behaviors because they often mutate state through different paths.

## Goal

Introduce a semantic operation layer for automatable UI behaviors and route both user and UI-driver handlers through it.

## Scope

1. Define operation module(s) for representative surfaces:
   - graph selection/clear/expand,
   - task filters open/close/navigation,
   - session settings menu open/keyboard apply.
2. Refactor event handlers and UI-driver handlers to call shared operations.
3. Keep focus and availability behavior explicit and source-aware.

## Dependencies / sequencing

- Depends on T-1.

## Acceptance criteria

- Shared operations are the canonical entrypoints for covered behaviors.
- UI-driver handlers stop duplicating bespoke state mutation logic for those behaviors.

## Validation

- Unit tests around operation semantics.
- Regression checks for known focus/context edge cases.
