---
epic: gpui
branch:
  suggested: rn/gpui/T-42-agent-runtime-integration-tests
rn:
  node:
    branch: rn/gpui/T-42-agent-runtime-integration-tests
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

### 2) Drive tests through the same Client API surfaces the UI uses

To ensure “chat just works” after this lands, tests must be written in terms of the public
client↔control-plane protocol surfaces, not by calling control-plane internals directly.

At minimum, cover:

- in-proc client API (`ControlPlaneHandle::connect_in_proc_client`) using `redesmyn_client_api::Client`,
- session events subscription stream (T-59) for observing assistant/turn events, and
- the same request shapes the GPUI session viewer relies on:
  - `CreateChatSession` / `CloseChatSession`,
  - `SendSessionMessage` (including conflict actions),
  - `GetSessionEvents` for DB verification.

Note: a UDS transport coverage test is also valuable, but can live in T-22; do not block T-42 on it.

### 3) End-to-end flows to cover

At minimum:

- User-managed **chat session** flow (what the GPUI “Chat” pane uses):
  - create chat session (repo-scoped) → subscribe to session events → send message →
    observe `TurnStarted` / assistant output / `TurnCompleted` via subscription → close chat session.
- Task-scoped **agent session** flow (what task sessions use):
  - start structured Codex task session → send message → observe assistant message (and turn events) → stop.
- Resume-by-id structured turn when external session id exists.
- Conflict handling:
  - turn in progress (structured) → 409 with stable code
  - session conflict → 409 with stable code
- Interrupt semantics:
  - interrupt then resume-by-id (where supported)

Also cover a minimal Shell/tmux flow using a fake tmux facade if real tmux is unavailable.

### 4) Assertions (model + events)

Tests must assert:

- command lifecycle transitions (no silent actions),
- durable session events were persisted (no deltas),
- query surfaces can retrieve conversation history with pagination.

Important: validate the daemon→control-plane event path:

- mock runners emit `DaemonMessage::SessionEventBatch` frames over the daemon stream protocol,
- the control plane persists them (T-40) and surfaces them via session event subscriptions (T-59).

Important (anti-fake): the “send message → assistant reply” path must be triggered by the
client API call that the UI uses (e.g. `SendSessionMessage`), not by directly pushing events into
the control plane from the test harness.

### 5) Tooling hooks

When failures occur:

- capture enough logs/artifacts to debug quickly (without gigantic dumps).

## Acceptance criteria

- A CI-friendly test suite runs deterministically without Codex/Claude installed.
- The suite covers the key compatibility requirements:
  - resume-by-id
  - conflict semantics
  - interrupt semantics
  - durable session history
  - and the end-to-end chat-session flow (create/subscribe/send/observe/close) via client API.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on control plane agent command semantics (T-41) and session persistence (T-40).
- Depends on the client API surface + session subscription plumbing (T-12/T-59) and daemon session supervisor + runners (T-35..T-38).

## Reference implementation (today; for behavior orientation only)

- Integration harness (Python today):
  - `tests/scenarios/scenario.py`
  - `tests/test_api_integration.py`
- Agent semantics (Python today):
  - `redesmyn/task_agent_messaging.py`
  - `redesmyn/agent_interface/codex.py`
  - `redesmyn/agent_interface/claude_code.py`
