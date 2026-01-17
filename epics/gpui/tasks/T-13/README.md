---
epic: gpui
branch:
  suggested: rn/gpui/T-13-protocol-tooling
rn:
  parent: T-10
  after:
    - T-8
---

# T-13 Protocol tooling + wiretap (Domain 1)

## Problem

We are choosing Protobuf for performance, but we still need excellent observability during development.

Without first-class tooling, a binary protocol becomes a debugging liability and slows iteration.

## Goal

Provide protocol debugging tools that make Protobuf feel as inspectable as JSON:

- decode/encode frames,
- optionally “tap” live connections,
- and produce actionable output without flooding logs.

## Requirements

### 1) `rn` protocol tooling

Extend the Rust CLI skeleton (T-8) with subcommands like:

- `rn protocol decode` (bytes/frames → JSON)
- `rn protocol encode` (JSON → bytes/frames) for testing
- optional: `rn protocol tap --uds <path>` (connect and log message summaries)

### 2) Formatting policy

- Default output is concise (message type + ids + scope + sizes).
- Provide `--verbose` to show payloads (with size limits / truncation).

### 3) Compatibility

- Tooling must understand:
  - the canonical envelope (T-9),
  - the generated protobuf schema (T-10),
  - and JSON diagnostic format where applicable.

## Acceptance criteria

- A developer can take a captured frame log and decode it into readable JSON.
- A developer can generate test frames to exercise parsers.
- The tool is safe by default (no accidental giant dumps).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-10/README.md` (schema/codegen).
- Depends on `epics/gpui/tasks/T-8/README.md` (Rust CLI skeleton) for integration.

