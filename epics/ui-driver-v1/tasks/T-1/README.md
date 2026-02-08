---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-1-contract-and-boundary
rn:
  parent: null
---

# T-1 UI Driver v1 contract + ownership matrix + migration plan

## Problem

UI-driver capabilities and control-plane capabilities overlap in places, and migration from fixed smokers to script-driven automation needs a single canonical contract to avoid drift.

## Goal

Define the v1 contract before implementation:

- method ownership matrix (UI-driver vs control-plane),
- script schema and execution semantics,
- compatibility/deprecation strategy for legacy `rn ui-driver` commands,
- and acceptance criteria for parity and targetability.

## Scope

1. Publish method ownership matrix and enforceable rules.
2. Define script line schema (`channel`, `request`, optional `timeout`, optional `assertions`).
3. Define step result schema and error handling semantics.
4. Define migration plan replacing fixed smoke commands.
5. Define required parity invariants for operation-layer work.

## Non-goals

- Implementing runtime changes.

## Dependencies / sequencing

- Foundation task for all other tasks in this epic.

## Acceptance criteria

- Canonical v1 contract doc exists and is approved.
- Every existing UI-driver method is classified as: keep, move, or deprecate.
- Migration policy is explicit (no silent command behavior changes).

## Validation

- Contract review with maintainers.
- Task-level checklists updated to reference this contract.
