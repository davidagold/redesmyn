---
epic: gpui
branch:
  suggested: rn/gpui/T-22-control-plane-test-harness
rn:
  parent: T-20
---

# T-22 Control plane integration test harness (mock daemon + real repo modes) (Domain 2)

## Problem

AI-first testability requires tests that are:

- easy to run,
- deterministic,
- and capable of validating both model state and UI state.

We also need a practical strategy for speed:

- most tests should run without touching a real git repo,
- a smaller set should validate true integration with repo-local execution.

## Goal

Build a control-plane-centric integration test harness that supports two modes:

1. **Mock daemon mode**: deterministic daemon simulator that emits telemetry/events and handles commands from fixtures.
2. **Real repo mode**: temporary repo fixture with a real daemon/executor (later domains will implement the daemon side).

This ticket focuses on the control-plane harness, APIs, and fixtures so later domains can plug in real implementations.

## Requirements

### 1) Mock daemon harness

Provide a harness component that can:

- register a fake daemon connection (in-proc transport),
- receive command dispatches,
- emit deterministic command updates and telemetry/events,
- and simulate failure/blocking scenarios.

### 2) Fixture-driven tests

Define a fixture pattern so tests can express:

- initial graph state,
- sequence of actions/commands,
- expected command lifecycle transitions,
- expected events and final projection state.

### 3) “Wait until” usage (no sleeps)

All tests must use wait primitives (T-15) rather than `sleep` for correctness.

### 4) UDS coverage

At least one integration test should exercise the UDS server path end-to-end:

- start control plane,
- connect a client over UDS,
- run a scenario through requests + subscriptions + waits.

## Acceptance criteria

- There is a reliable harness to write AI-friendly integration tests against the control plane today.
- Tests can trigger actions and assert outcomes deterministically.
- Mock daemon mode is sufficient to test “no silent actions” semantics (command updates visible immediately).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on:
  - client API server (T-20),
  - command engine (T-19),
  - event hub (T-18),
  - and the AI testability contract (T-15).

## Reference implementation (today; testing orientation only)

- Scenario-based integration harness (Python today):
  - `tests/scenarios/scenario.py` (temp git repo + SQLite DB + in-process ASGI app + in-process daemon connection registry).
  - `tests/scenarios/seeds/git.py` and `tests/scenarios/variants.py` (common seeded graphs/repos for tests).
  - `tests/helpers/ws.py` (in-process WebSocket helpers).
- Current integration tests to mirror/learn from:
  - `tests/test_daemon_ws_runtime_integration.py` (daemon WS + command routing + merge.run event ingestion).
  - `tests/test_epic_graph.py` (graph response expectations).
  - `tests/test_git_mechanics_planning.py` / `tests/test_git_mechanics_execution.py` (plan/execution semantics in the current system).
- E2E harness (Python today):
  - `tests/e2e/` (Playwright-based tests)
  - `tests/README.md` (test conventions)
