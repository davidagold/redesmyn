---
epic: gpui
branch:
  suggested: rn/gpui/T-38-daemon-claude-code-runner
rn:
  parent: T-35
  after:
    - T-34
    - T-32
---

# T-38 Daemon Claude Code runner (structured) (exec-based) (Domain 4)

## Problem

Claude Code structured sessions are important for Redesmyn and must behave consistently with Codex structured sessions:

- stable resume handles,
- durable chat/session history,
- interrupt semantics,
- and predictable conflict behavior at the control plane layer.

## Goal

Implement the daemon-side Claude Code runner built on:

- exec-session supervisor (T-35),
- Claude parser (T-34),
- and resume-by-id utilities (T-32).

This runner does not require tmux.

## Requirements

### 1) Structured invocation requirements

For structured mode, enforce that the harness argv results in:

- `claude --print --output-format stream-json ...` (or equivalent forms).

If not structured, refuse with a user-actionable error.

### 2) Resume handle capture + persistence

- Capture and persist `session_id` from Claude stream-json events.
- Surface capability `can_resume_by_id` only when a stable session id exists.

### 3) Resume-by-id turns

Use the canonical resume-by-id builder semantics (T-32):

- insert `--resume <session_id>` after `--print`/`-p`,
- drop `--continue`,
- replace existing `--resume`.

### 4) Interrupt semantics

Implement interrupt best-effort (SIGINT), with escalation on timeout, and ensure state transitions are observable.

### 5) Testability

Do not require Claude installed:

- use a deterministic fixture process that emits Claude-like stream-json.

## Acceptance criteria

- A deterministic integration test can start a “mock Claude” session and observe:
  - session id capture,
  - turn boundaries,
  - assistant messages,
  - and interrupt behavior.
- Resume-by-id turns match the Python semantics (T-32 tests + runner-level integration).

## Dependencies / sequencing

- Depends on exec-session supervisor (T-35).
- Depends on Claude parser (T-34).
- Uses resume-by-id builder / kind detection (T-32).

## Reference implementation (today; for behavior orientation only)

- Claude parsing (Python today):
  - `redesmyn/agent_interface/claude_code.py`
  - `tests/test_claude_code_agent_interface.py`
- Resume-by-id turns (Python today):
  - `redesmyn/agent_turn_transport.py`
  - `tests/test_agent_turn_transport.py`

