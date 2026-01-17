---
epic: gpui
branch:
  suggested: rn/gpui/T-3-error-conventions
rn:
  parent: T-1
---

# T-3 Error/result conventions (Domain 0)

## Problem

Parallel development in Rust quickly devolves into inconsistent error strategies:

- ad-hoc `anyhow` everywhere (hard to match on, unclear semantics),
- mixed error wrapping policies,
- inconsistent exit codes and user-facing messages.

For Redesmyn, errors must be:

- actionable (especially for CLI and graph UX),
- structured (so UI can render them meaningfully),
- consistent across daemon/control-plane boundaries.

## Goal

Define and implement a consistent, maintainable error strategy for the Rust workspace.

## Requirements

### 1) Library vs binary policy

- Library crates (`redesmyn_*`): use typed error enums (`thiserror`), return `Result<T, E>`.
- Binaries (`rn`, server, daemon, desktop): may use `anyhow` for top-level error reporting, but must map errors to:
  - stable exit codes (CLI),
  - structured error responses (APIs/protocol),
  - user-visible, non-noisy UI messages.

### 2) Error taxonomy

Establish a small set of shared, stable error categories (in a core crate) that can cross boundaries, e.g.:

- `InvalidRequest`
- `NotFound`
- `Conflict`
- `Unauthorized`
- `Unavailable`
- `Internal`

These categories should be represented in:

- protocol error envelopes (daemon/control-plane),
- control-plane API responses,
- CLI output formatting.

### 3) Panic policy

- No panics in normal control flow.
- Panics only for truly impossible invariants; prefer `debug_assert!` when the invariant is “should never happen but not security sensitive”.

### 4) Documentation

Add a short doc describing:

- how to add new error types,
- how to map to exit codes / API status codes,
- how to produce actionable user-facing messages.

## Acceptance criteria

- At least one end-to-end example exists:
  - a typed error raised in a library crate,
  - mapped to a structured protocol/API error,
  - rendered as a user-facing message and an exit code in a binary.
- Error strategy is consistent and easy for parallel implementers to follow.

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-1/README.md` (workspace + crate skeletons).
- Should coordinate with `epics/gpui/tasks/T-7/README.md` (protocol error envelope) and `epics/gpui/tasks/T-8/README.md` (CLI exit codes/output).

## Reference implementation (today; error handling patterns)

- API errors (Python today):
  - `redesmyn/api.py` (FastAPI endpoints returning `HTTPException` / `JSONResponse` with `detail`).
  - `redesmyn/schemas/core.py` (request/response types; error `detail` conventions).
  - `redesmyn/task_agent_messaging.py` (`TaskAgentMessageError` with `status_code`; conflict detail conventions).
- CLI errors (Python today):
  - `redesmyn/cli.py` (Typer commands; uses `typer.Exit(...)` codes).
  - `tests/test_cli_integration.py` (asserts CLI behavior/exit codes).
