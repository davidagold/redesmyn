---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-5-parity-conformance-tests
rn:
  parent: T-4
  after:
    - T-2
---

# T-5 Parity conformance tests (user path vs UI-driver path)

## Problem

Parity regressions are hard to detect without explicit tests that compare user-triggered behavior with UI-driver-triggered behavior.

## Goal

Add conformance tests that run equivalent operations through both paths and compare semantic outcomes.

## Scope

1. Build test harness helpers for dual-path execution.
2. Add parity suites for:
   - graph selection and filter availability,
   - settings menu open/navigation/close,
   - focus restoration after transient overlays.
3. Compare via `UiSnapshot` and focused, stable invariants.

## Dependencies / sequencing

- Depends on T-4.
- After T-2 (script runner) for reusable automation plumbing.

## Acceptance criteria

- Known parity regressions are captured by tests.
- New operation-layer changes must pass parity suite.

## Validation

- Automated parity test run in CI.
