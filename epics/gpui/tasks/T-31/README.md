---
epic: gpui
branch:
  suggested: rn/gpui/T-31-daemon-integration-tests
rn:
  parent: T-23
---

# T-31 Daemon integration test harness (real repo fixtures) (Domain 3)

## Problem

We need confidence that the daemon:

- executes repo-local commands correctly,
- emits telemetry correctly,
- and respects lease/attachment semantics.

We also have an explicit requirement: AI must be able to test workflows deterministically.

Control-plane tests (T-22) cover mock daemon mode, but we also need a **real repo** daemon harness for true integration coverage.

## Goal

Create a daemon integration test harness that:

- spins up a daemon instance in-process (or as a subprocess) with a temp state dir,
- registers and attaches a temp git repo,
- exercises key operations (attach, observe tick, worktree ensure, plan/execute merge),
- and asserts emitted messages/events deterministically.

## Requirements

### 1) Fixture helper library

Provide shared test helpers to:

- create temp git repos and branches,
- create simple “task graphs” (as control-plane fixtures or minimal stubs),
- and simulate control-plane stream endpoints for the daemon to connect to.

### 2) Deterministic “tick” control

Observation and execution must be controllable without sleeps:

- run one observation tick explicitly,
- advance timers via test runtime controls where needed.

### 3) Coverage targets (initial)

At minimum, cover:

- repo registry registration + attach/detach
- lease enforcement rejection
- observation emits expected events
- merge/restack plan generation
- merge execution success path

### 4) Failure diagnostics

On failure, tests should emit:

- decoded protocol frames (via tooling, T-13),
- and/or structured logs with stable IDs.

## Acceptance criteria

- Daemon integration tests exist and can run in CI on macOS/Linux.
- Tests are deterministic and do not depend on timing sleeps.
- The harness is reusable for future daemon features (agent runtime, artifacts, etc.).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Builds on T-23/T-24/T-26/T-28, and later on T-29/T-30.
- Complements the control-plane harness in T-22.

## Reference implementation (today; integration testing orientation only)

- Existing repo/worktree fixtures (Python today):
  - `tests/scenarios/scenario.py` (creates a temp repo and worktrees; provides in-process app + daemon registry).
  - `tests/scenarios/seeds/git.py` (real conflict scenarios and stacks used by merge/restack tests).
- Existing “close to daemon” integration coverage today:
  - `tests/test_daemon_ws_runtime_integration.py` (daemon WS registry + routing behavior).
  - `tests/test_git_mechanics_planning.py` / `tests/test_git_mechanics_execution.py` (real git + worktree planning/execution).
  - `tests/test_agent_worktree_autobranch.py` (worktree creation semantics used by agents).
