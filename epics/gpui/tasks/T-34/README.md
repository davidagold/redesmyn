---
epic: gpui
branch:
  suggested: rn/gpui/T-34-claude-code-parser
rn:
  node:
    branch: rn/gpui/T-34-claude-code-parser
  parent: T-14
---

# T-34 Claude Code structured parser → session events (pure state machine) (Domain 4)

## Problem

Claude Code (structured) is a key agent type for Redesmyn.

We need a deterministic parser for Claude’s `--print --output-format stream-json` mode so we can:

- extract durable chat history events,
- detect turn boundaries reliably,
- and persist stable resume handles (session_id).

## Goal

Implement a **pure, deterministic** Claude Code output parser that:

- consumes streaming text incrementally,
- extracts structured events (one JSON object per line, with multiline-object fallback),
- emits semantic session events (per T-14),
- and maintains:
  - turn state,
  - capabilities,
  - external session reference (`session_id`).

Parser-only: no daemon process management in this ticket.

## Requirements

### 1) Structured stream-json path (preferred)

Preserve the current semantics for events:

- `{"type":"system","subtype":"init", ...}` initializes ready state and enables stream semantics.
- `{"type":"user", ...}` / `{"type":"assistant", ...}` drive busy state and imply “turn started”.
- `{"type":"result", ...}` drives ready state and implies “turn completed”.

Capture `session_id` when present and expose it as the resume handle (capability `can_resume_by_id`).

### 2) Robustness to chunking

Handle:

- JSON objects split across reads (chunked lines),
- multiline JSON objects (not newline-delimited),
- and non-JSON noise before the first structured event.

### 3) Output: typed semantic surface

As with T-33, define a small typed surface returned by the parser containing:

- emitted semantic events (ultimately mapped to `SessionEvent`),
- current turn state + detail,
- capabilities and external session ref updates.

### 4) Tests (port behavior precisely)

Add unit tests equivalent to:

- `tests/test_claude_code_agent_interface.py`

## Acceptance criteria

- Parser reproduces the Python behavior for:
  - capability enabling,
  - session_id capture,
  - chunked + multiline object handling,
  - and turn/message event emission.
- Parser is reusable by the daemon runtime and test harnesses.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on the structured session event contract (T-14) for the event vocabulary and size constraints.
- Complements utilities from T-32 (kind/mode inference + resume-by-id).

## Reference implementation (today; for behavior orientation only)

- Claude parser (Python today):
  - `redesmyn/agent_interface/claude_code.py` (`ClaudeCodeAgent`).
- Tests (Python today):
  - `tests/test_claude_code_agent_interface.py`

