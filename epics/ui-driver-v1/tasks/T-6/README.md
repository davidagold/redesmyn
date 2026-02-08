---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-6-instance-scoped-targets
rn:
  parent: T-4
---

# T-6 Instance-scoped target model for session/composer actions

## Problem

Global or ambiguous target assumptions cause cross-instance bugs (e.g., task-card settings actions affecting left-pane session).

## Goal

Introduce explicit target scope for UI-driver requests and operation calls where multiple instances exist.

## Scope

1. Define target scope enums/identifiers (e.g., left session, task-card session).
2. Add target parameter to relevant UI-driver requests.
3. Route handlers to resolved target instance only.
4. Enforce errors for ambiguous/missing targets where required.

## Dependencies / sequencing

- Depends on T-4.

## Acceptance criteria

- Session/composer-affecting actions are instance-scoped.
- Cross-instance leakage bugs are structurally prevented.

## Validation

- Unit/integration tests proving isolation across multiple session views.
