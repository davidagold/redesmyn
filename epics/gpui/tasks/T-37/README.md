---
epic: gpui
branch:
  suggested: rn/gpui/T-37-daemon-codex-runner
rn:
  parent: T-35
  after:
    - T-33
    - T-32
---

# T-37 Daemon Codex runner (structured) (exec-based; output-last-message capture) (Domain 4)

## Problem

Codex structured sessions are our top priority.

We need a daemon-side runner that:

- spawns Codex in the correct worktree context,
- captures structured semantics for the native session viewer,
- records stable resume handles (thread_id),
- and remains robust if streaming is imperfect (e.g., capture final message).

## Goal

Implement the daemon-side Codex runner built on the exec-session supervisor (T-35), using the Codex parser (T-33).

The runner must support:

- starting a new structured session, and
- running resume-by-id turns (when a thread id exists),

without requiring tmux.

## Requirements

### 1) Structured invocation requirements

For structured mode, enforce that the harness argv results in:

- `codex exec --json ...`

If the harness is not structured, the runner must refuse with a user-actionable error (the control plane uses this to guide config).

### 2) Resume handle capture + persistence

- Persist external session ref as soon as it is observed (`thread_id`, and `turn_id` when available).
- Surface capability `can_resume_by_id` only when a stable thread id is known.

### 3) Output-last-message capture (robustness)

Preserve the “capture final message” strategy used today:

- when invoking `codex exec ...`, inject `--output-last-message <path>` if not present,
- treat the captured content as a durable assistant message emission (bounded size; artifact if huge),
- ensure we do not duplicate messages when both streaming and file capture are present (dedupe strategy).

### 4) Interrupt semantics

- Implement interrupt as best-effort (SIGINT), with escalation on timeout.
- Ensure the control plane can observe in-flight → interrupted/canceled transitions (no silent actions).

### 5) Testability

The runner must be testable without real Codex installed:

- accept a “mock codex executable” path in test mode, or
- spawn a deterministic fixture process that emits Codex-like JSONL.

## Acceptance criteria

- A deterministic integration test can start a “mock Codex” session and observe:
  - thread id capture,
  - turn boundary events,
  - assistant message events,
  - and output-last-message capture (when enabled).
- Resume-by-id turn execution uses the canonical builder semantics (T-32).
- Interrupt semantics work and are observable.

## Dependencies / sequencing

- Depends on exec-session supervisor (T-35).
- Depends on Codex parser (T-33).
- Uses resume-by-id builder / kind detection (T-32).

## Reference implementation (today; for behavior orientation only)

- Codex parsing (Python today):
  - `redesmyn/agent_interface/codex.py`
  - `tests/test_codex_agent_interface.py`
- Output-last-message capture (Python today):
  - `redesmyn/agent_runtime.py` (`_codex_exec_argv_with_output_last_message`, `agent_session_codex_last_message_path`).
  - `redesmyn/agent_driver.py` (`maybe_capture_codex_last_message_file`).
- Resume-by-id turns (Python today):
  - `redesmyn/agent_turn_transport.py`
  - `tests/test_agent_turn_transport.py`

