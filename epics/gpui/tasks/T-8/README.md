---
epic: gpui
branch:
  suggested: rn/gpui/T-8-rn-rust-skeleton
rn:
  parent: T-1
---

# T-8 Rust `rn` skeleton (fast CLI harness) (Domain 0)

## Problem

`rn` is first-class and must be **very fast** in the Rust port.

If we wait to build the CLI until late, we risk:

- building APIs that are UI-only and awkward for CLI,
- discovering ergonomics/perf problems late,
- and missing opportunities to use `rn` as a developer tool during the port (wiretap, DB inspection, protocol decode).

## Goal

Create a Rust `rn` binary skeleton that:

- starts fast,
- has a coherent command structure,
- establishes exit-code and error formatting conventions,
- can grow into the primary interface for agents and humans.

## Requirements

### 1) Name and coexistence (split codebase)

During the port, the Python CLI already exists as `rn`.

Resolve this by:

- naming the Rust binary `rn-rs` initially, or
- providing a clear switch mechanism (documented) so contributors can choose which `rn` is on PATH.

This ticket should pick one approach and document it clearly.

### 2) CLI framework + conventions

- Use `clap` (or equivalent) with subcommands.
- Define:
  - structured output conventions (human vs json),
  - exit code mapping (aligned with T-3).

### 3) Early developer-facing subcommands

Include a few early “developer utility” commands that reinforce the architecture:

- `rn-rs doctor` (environment/config sanity)
- `rn-rs protocol decode` (if available from T-7)
- `rn-rs version`

### 4) Performance hygiene

- Keep dependencies minimal.
- Add a tiny startup benchmark harness (even if crude) so we can track regressions.

## Acceptance criteria

- `rn-rs --help` is fast and responsive.
- Command structure is established and documented.
- Coexistence with Python `rn` is clear and non-confusing for contributors.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-1/README.md` (workspace + crate skeletons).
- Should coordinate with `epics/gpui/tasks/T-3/README.md` (exit codes + error formatting).
- Optionally depends on `epics/gpui/tasks/T-7/README.md` if we include protocol decode tooling in the initial CLI skeleton.

## Reference implementation (today; `rn` CLI)

- CLI entrypoints (Python today):
  - `redesmyn/__main__.py` (module entrypoint).
  - `redesmyn/cli.py` (Typer app; includes `rn sync`, `rn debug dev`, merge run commands, etc.).
- CLI integration tests (Python today):
  - `tests/test_cli_integration.py`
