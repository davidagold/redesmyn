---
epic: ui-driver-v1
branch:
  suggested: rn/ui-driver-v1/T-2-script-runner-core
rn:
  parent: T-1
---

# T-2 Script runner core and removal of fixed smoke subcommands

## Problem

`rn ui-driver` currently ships hardcoded scenario functions (`smoke`, `graph-smoke`, etc.), which slows iteration and forces code changes for test flow changes.

## Goal

Replace fixed scenarios with a script executor that reads LDJSON/NDJSON and executes steps dynamically.

## Scope

1. Add `rn ui-driver run` (or equivalent canonical command) for script execution.
2. Parse script lines into typed requests.
3. Execute per-step with deterministic timeout/error semantics.
4. Emit machine-readable step results.
5. Remove fixed smoke scenario subcommands and related code paths.

## Non-goals

- Full target registry/generic target actions (later tasks).

## Dependencies / sequencing

- Depends on T-1.

## Acceptance criteria

- Fixed smoke subcommands are removed.
- Script runner can execute equivalent basic flows (`open_epic`, waits, screenshot).
- Per-step failures report request, method, and actionable error.

## Validation

- Unit tests for parser and execution behavior.
- End-to-end run against desktop UI-driver socket.
