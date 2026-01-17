---
epic: harness-interface-v0
branch:
  suggested: rn/harness-interface-v0/T-10-structured-exec-events
rn:
  linear:
    issue_id: b2435ea5-1887-4e55-8e23-5bb3a12565ad
    identifier: RED-18
  parent: T-3
---

# T-10 Structured exec mode + semantic event stream plumbing (enables T-5/T-9)

## Problem

T-3/T-4 implement agent-specific parsing and capability semantics (Codex / Claude Code), but Redesmyn currently starts agents primarily in an interactive, tmux-backed TUI mode.
That mode is useful for “drop in and drive the agent”, but it does **not** reliably produce machine-readable structured events (turn boundaries, assistant messages, etc.).

As a result:

- “turn complete” semantics are mostly heuristics → not reliable enough to gate automation.
- we don’t have a stable “assistant message” signal → T-9 becomes either flaky or hacky.
- T-5’s gated auto-resume becomes unsafe (cannot deterministically know “agent turn complete”).

The v0 intent of the interface is that **when an agent supports structured signals**, Redesmyn can opt into a runtime mode that makes those signals available and can plumb them end-to-end (driver → DB → websocket → UI).

## Goal

Add the missing “structured mode” runtime and the missing “semantic event stream” plumbing so that:

- Codex/Claude can emit rich structured events (turn boundaries + messages) in a way the driver can consume.
- AgentDriver persists and publishes these events (not only the derived `agent_semantic_status` snapshot).
- Downstream tasks (T-5 conflict gating, T-9 message preview) can be implemented on top of deterministic structured signals instead of tmux prompt heuristics.

## Requirements

### 1) Launch / runtime must support structured exec modes

Redesmyn must be able to start an agent session in a mode that emits structured output:

- Codex: a JSONL/structured stream (e.g. `codex exec --json ...`) with thread/turn/message events.
- Claude Code: a JSON/stream-json output mode (per T-4 findings) with session id + turn/message events.

Implementation guidance:

- Introduce an explicit “runtime mode” / “interface mode” concept that is separate from:
  - agent kind (Codex/Claude/Shell),
  - transport/run mode (tmux detached vs foreground vs external).
- For v0, it is acceptable if “structured exec mode” is only supported for Codex/Claude kinds.
- The session’s structured output **must** be routed into the session log path so AgentDriver can tail it (same tailing mechanism for tmux/external).

Primary modules likely to change:

- `redesmyn/agent_runtime.py` (start/restart path must support structured exec argv construction and output routing)
- `redesmyn/db/models.py` (persist the chosen mode on `AgentSession`, if needed)
- `redesmyn/schemas/core.py` + API endpoints if the mode is surfaced to UI

### 2) AgentDriver must process and persist semantic events (not just snapshots)

Today, `AgentBackend.consume_output(...)` returns `list[AgentEvent]`, but AgentDriver largely ignores the returned list and only persists:

- `agent_capabilities`
- `agent_semantic_status`
- `external_session_ref`

This task must add a first-class event pipeline:

- AgentDriver consumes output → gets `AgentEvent`s → persists them in DB → publishes over websocket.
- Events should be idempotent / deduplicated enough to avoid DB thrash (e.g. “persist only on change” for derived snapshots, and “append-only” for semantic events).

Primary modules likely to change:

- `redesmyn/agent_driver.py` (consume `AgentEvent`s; persist + publish)
- `redesmyn/api.py` / websocket event hub plumbing (ensure events reach UI)

### 3) Extend the event model to cover turns + assistant messages (minimum set)

To support T-5 and T-9 with stable semantics, the v0 event model must include at least:

- `turn_started` / `turn_completed` (explicit boundary events; includes external ids when available)
- `assistant_message` (text content suitable for preview; includes external ids when available)

Design constraints:

- Keep payloads typed/validated (Pydantic discriminated unions, per repo conventions).
- Keep the DB footprint modest: v0 does not need full transcripts, but must be able to derive:
  - “latest assistant message preview” (for T-9),
  - “latest turn complete for remediation turn” (for T-5 gating).

Primary modules likely to change:

- `redesmyn/agent_interface/v0.py` (extend `AgentEvent` union)
- `redesmyn/agent_interface/codex.py` (parse structured events and emit message/turn events)
- `redesmyn/agent_interface/claude_code.py` (same)

### 4) Persist a minimal preview state derived from structured events

To unblock T-9 and provide a stable Timeline foundation later:

- Persist the latest assistant-message preview on `AgentSession` (truncated, bounded size).
- Update this preview only from stable semantic signals (structured message events), not from raw tmux screen scraping.

Primary modules likely to change:

- `redesmyn/db/models.py` (preview field; JSON column preferred)
- `redesmyn/agent_driver.py` (derive + persist preview)

### 5) Tests: validate structured streams end-to-end

Add tests that validate the full loop without needing to actually run Codex/Claude binaries:

- Given a fixture JSONL/JSON “structured stream”, the backend parser emits the expected `AgentEvent`s.
- AgentDriver processes these events and persists:
  - turn boundary state,
  - external session ref updates,
  - message preview updates.
- Websocket publish shape is stable enough for UI consumers.

## Acceptance criteria

- A Codex session started in “structured exec mode” reliably produces turn boundary events and assistant message events (no prompt heuristics required).
- AgentDriver persists and publishes semantic events; downstream tasks can implement gating/preview on top of them.
- The system has an explicit and visible degraded path:
  - interactive tmux sessions can still run, but are treated as “heuristic-only” and do not claim structured capabilities unless structured events are observed.
