---
epic: gpui
branch:
  suggested: rn/gpui/T-42-agent-runtime-integration-tests
rn:
  parent: T-41
---

# T-42 End-to-end agent session integration tests (mock agents, determinism, persistence) (Domain 4)

## Problem

Agent runtime is a high-change, high-risk subsystem during the port.

We also have a product requirement:

- the new app must be easy for AI to test (deterministic actions + assertable outcomes).

If we rely on real Codex/Claude installations for testing, tests will be flaky and non-portable.

## Goal

Build a deterministic, portable integration test suite for agent sessions that:

- does not require real external agents,
- exercises control plane ↔ daemon boundaries,
- validates durable session event persistence (T-40),
- and asserts the “send message” semantics (T-41).

## Requirements

### 1) Mock agent fixtures (portable)

Provide deterministic “mock agent” executables used in tests that emulate:

- Codex JSONL output (`thread.started`, `turn.started`, assistant message, `turn.completed`)
- Claude stream-json output (`system:init`, `assistant`, `result`)

Constraints:

- must run on macOS + Linux,
- must be hermetic and fast,
- must allow deterministic timing (avoid sleeps where possible).

### 2) End-to-end flows to cover

At minimum:

- Start structured Codex session → send message → observe assistant message → stop.
- Resume-by-id structured turn when external session id exists.
- Conflict handling:
  - turn in progress (structured) → 409 with stable code
  - session conflict → 409 with stable code
- Interrupt semantics:
  - interrupt then resume-by-id (where supported)

Also cover a minimal Shell/tmux flow using a fake tmux facade if real tmux is unavailable.

### 3) Assertions (model + events)

Tests must assert:

- command lifecycle transitions (no silent actions),
- durable session events were persisted (no deltas),
- query surfaces can retrieve conversation history with pagination.

### 4) Tooling hooks

When failures occur:

- capture enough logs/artifacts to debug quickly (without gigantic dumps).

## Acceptance criteria

- A CI-friendly test suite runs deterministically without Codex/Claude installed.
- The suite covers the key compatibility requirements:
  - resume-by-id
  - conflict semantics
  - interrupt semantics
  - durable session history

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on control plane agent command semantics (T-41) and session persistence (T-40).
- Depends on daemon session supervisor + runners (T-35..T-38).

## Reference implementation (today; for behavior orientation only)

- Integration harness (Python today):
  - `tests/scenarios/scenario.py`
  - `tests/test_api_integration.py`
- Agent semantics (Python today):
  - `redesmyn/task_agent_messaging.py`
  - `redesmyn/agent_interface/codex.py`
  - `redesmyn/agent_interface/claude_code.py`
