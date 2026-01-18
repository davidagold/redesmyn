---
epic: gpui
branch:
  suggested: rn/gpui/T-33-codex-parser
rn:
  parent: T-14
---

# T-33 Codex structured parser → session events (pure state machine) (Domain 4)

## Problem

Codex (structured) is the highest-priority agent runtime for this epic.

To build a native session viewer (chat + structured timeline) we need a robust, performant way to convert Codex output into stable, typed session events.

If parsing is ad-hoc (regexes scattered in the daemon), we will:

- miss events,
- regress resume handles,
- and create fragile “it works in dev” behavior.

## Goal

Implement a **pure, deterministic** Codex output parser that:

- consumes streaming text incrementally,
- emits semantic events suitable for the `SessionEvent` contract (T-14),
- and maintains internal state for:
  - turn state,
  - capabilities,
  - external session reference (thread_id/turn_id).

This ticket is parser-only (no process spawning, no daemon wiring).

## Requirements

### 1) Structured JSONL path (preferred)

Support Codex “exec --json” output where one JSON object is emitted per line.

Semantics to preserve:

- `thread.started` captures thread id (resume handle).
- `turn.started` / `turn.completed` drive turn state and “turn boundary” semantics.
- `item.started` / `item.completed` are treated as activity signals (even when explicit turn boundaries are absent).
- Assistant messages can appear as:
  - `assistant.message` style events, and/or
  - `item.completed` with `item.type` in `{reasoning, agent_message, assistant_message}`.

### 2) Heuristic fallback path (interactive)

When structured events are absent, implement the conservative heuristic behavior used today:

- detect Codex prompt tails (`>\s*$`, including variants),
- treat prompt transitions as “ready”/“completed” when plausible,
- degrade to `Unknown` after a bounded busy timeout (so we never wedge forever).

This path is important for Shell/tmux compatibility and for robustness when Codex structured output is misconfigured.

### 3) Output: typed semantic surface

Define a small typed surface returned by the parser (exact type shape is up to implementation), containing:

- emitted semantic events (ultimately mapped to `SessionEvent`),
- current turn state + detail,
- capabilities (e.g., can_resume_by_id, can_detect_turn_complete, can_stream_semantic_events),
- external session reference updates (thread_id/turn_id).

### 4) Performance

- Avoid unbounded buffers; keep bounded tail buffers for prompt heuristics.
- Avoid quadratic behavior on long outputs.

### 5) Tests (port behavior precisely)

Add unit tests equivalent to:

- `tests/test_codex_agent_interface.py`

## Acceptance criteria

- The parser roundtrips the Python behavior for:
  - structured events (thread/turn),
  - assistant message extraction,
  - item.* activity,
  - prompt heuristics,
  - and busy timeout degradation.
- Parser is reusable by daemon runtime and by any future tooling that needs to interpret Codex output.

## Dependencies / sequencing

- Depends on the structured session event contract (T-14) for the event vocabulary and size constraints.
- Complements utilities from T-32 (kind/mode inference + resume-by-id).

## Reference implementation (today; for behavior orientation only)

- Codex parser (Python today):
  - `redesmyn/agent_interface/codex.py` (`CodexAgent`).
- Tests (Python today):
  - `tests/test_codex_agent_interface.py`

