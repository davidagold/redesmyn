---
id: T-11
epic: harness-interface-v0
stacked_on: T-10
branch:
  suggested: rn/harness-interface-v0/T-11-structured-resume-transport
linear:
  issue_id: null
  identifier: null
---

# T-11 Structured continuation transport: resume-by-id turns (enables T-5)

## Problem

T-10 enables **structured** agent sessions by plumbing machine-readable event streams (turn boundaries + assistant messages) into the AgentDriver, DB, websocket event stream, and UI.

However, after T-10 we still lack a first-class way to **drive a structured agent session** with follow-up input without “typing into tmux”:

- We can observe structured events (and derive status/preview), but we cannot reliably *send the next message* in a structured-safe way.
- In particular, T-5 conflict assist needs to deliver a remediation prompt and then wait for an explicit turn boundary, but:
  - tmux send-keys is inherently interactive/heuristic,
  - it contaminates structured output if the agent is in JSON streaming mode, and
  - it does not leverage the `external_session_ref` resume handles we already persist.

In short: we have “structured read”, but not “structured write”.

## Goal

Implement a structured continuation transport that can run **a new structured turn** for an existing agent session using a persisted resume handle:

- **Codex**: resume a thread (and optionally a turn cursor) and run a new turn.
- **Claude Code**: resume a session and run a new turn.

This must work without tmux keystroke injection. The continuation should be executed as a fresh process invocation (a “turn runner”) that:

1. receives the remediation prompt in a structured-safe way,
2. streams structured output into the existing log path,
3. allows the AgentDriver to produce `agent.turn_started` / `agent.turn_completed` / `agent.assistant_message` events, and
4. yields a deterministic boundary for “the remediation turn completed”.

## Relationship to T-10 and T-5

- **T-10** provides:
  - a persisted `external_session_ref` for structured sessions (Codex thread id / Claude session id),
  - semantic events persisted as DB `Event`s and streamed to the UI.
- **T-5** needs:
  - a safe, structured-only way to deliver a remediation prompt,
  - a reliable way to correlate “the turn that processed that remediation prompt” to its completion event.

This task closes the gap so T-5 can be implemented without tmux prompt heuristics or tmux send-keys.

## Design constraints (v0)

- Prefer a “one process per structured turn” model:
  - it naturally enforces “no concurrent turn” and makes completion observable (process exit + turn-completed event).
- Avoid CLI/UI surface area changes unless required; this is an internal capability that other tasks can consume.
- Keep typing strict: request/response payloads should be typed and validated (Pydantic).

## Requirements

### 1) Define a continuation contract

Introduce a single internal API for “run one structured turn” for an existing session, expressed in terms of:

- `agent_session_id` (or `{task_id, agent_session_id}`),
- a prompt payload (string),
- the persisted `external_session_ref`,
- and any required execution metadata (cwd, env, launch configuration definition).

This should be the only thing T-5 needs to call to “send remediation” in structured mode.

### 2) Agent-kind specific resume execution

Implement resume-by-id for supported agent kinds:

- **Codex**:
  - take `ExternalSessionCodex.thread_id` (+ optional `turn_id`),
  - build argv to resume and execute a new structured turn.
- **Claude Code**:
  - take `ExternalSessionClaude.session_id`,
  - build argv to resume and execute a new structured turn.

Notes:
- v0 does not need to support every possible CLI flag permutation; it needs a deterministic “known-good” resume behavior for the supported harnesses we ship.
- Capabilities should reflect reality:
  - when resume-by-id is implemented and the session has a resume handle, set `can_resume_by_id = true`.

### 3) Prompt delivery must be structured-safe

The remediation prompt must be delivered without polluting the structured output stream:

- Prefer stdin file redirection (the same mechanism we already use for structured startup prelude injection).
- Do not use tmux send-keys for structured sessions.

### 4) Output routing and event plumbing

Ensure the continuation turn’s structured output is routed to the same per-session log path so AgentDriver can tail it and emit:

- `agent.turn_started`
- `agent.turn_completed`
- `agent.assistant_message`

If we adopt a “one process per turn” model, ensure exit code capture still works (for observability and error surfacing), but do not require process exit as the only completion signal.

### 5) Turn correlation (deterministic boundary)

Provide a robust way to correlate “the remediation we sent” with “the turn completed”:

- Preferred: correlate via `ExternalSessionCodex.turn_id` (or the equivalent) if available.
- Fallback: use an event-id boundary:
  - record the latest `agent.turn_completed` `Event.id` for the session before launching the continuation,
  - after launching, require an `agent.turn_completed` with `Event.id > baseline`.

This correlation mechanism must be exposed to T-5 so it can safely gate auto-resume.

### 6) Safety + idempotency

- Do not run a continuation turn if another structured turn for the same session is already running.
- If the resume handle is missing/unknown, fail fast with an actionable error (and degrade to manual flow in callers like T-5).
- Avoid sending duplicate remediation prompts during retries; callers should be able to treat the continuation request as idempotent.

### 7) Tests

Add tests that validate the “structured continuation turn” contract without calling external agent binaries:

- Given an `external_session_ref` and a prompt, the runner constructs the expected argv/env/stdin payload for each supported agent kind.
- The correlation boundary logic is correct (turn-id preferred; event-id fallback).
- Capabilities are updated correctly when resume-by-id is available and applicable.

## Acceptance criteria

- The codebase has a single, reusable “structured continuation turn” API that does not depend on tmux keystrokes.
- For Codex and Claude Code, Redesmyn can run a follow-up structured turn using a persisted resume handle and a prompt payload.
- There is a deterministic correlation boundary for “the remediation turn completed” suitable for T-5 gating.
- T-5 can be updated to depend on this task and remove tmux-based structured messaging assumptions.
