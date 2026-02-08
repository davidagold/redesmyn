---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-7-ui-snapshot-target-registry
rn:
  parent: T-6
---

# T-7 `UiSnapshot` target registry v1 (discoverable automation catalog)

## Problem

Dynamic automation needs target discovery (stable IDs/roles/state/actions), but `UiSnapshot` currently exposes only coarse surface state.

## Goal

Extend `UiSnapshot` with a stable target registry suitable for dynamic script generation and assertion.

## Scope

1. Add target catalog structures to protocol.
2. Emit stable target IDs and metadata from desktop view state.
3. Include key state fields required for reliable automation decisions.
4. Document stability guarantees for IDs.

## Dependencies / sequencing

- Depends on T-6.

## Acceptance criteria

- Scripts can enumerate actionable targets without coordinate coupling.
- Target IDs are stable across renders for the same logical controls.

## Validation

- Snapshot tests for target IDs/metadata stability.
