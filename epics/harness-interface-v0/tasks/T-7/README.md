# T-7 AgentDriver (session supervisor: liveness + logs + semantics + events)

## Metadata

```yaml
id: T-7
epic: harness-interface-v0
stacked_on: T-1
branch:
  suggested: rn/harness-interface-v0/T-7-agent-driver
```

## Problem

T-1 defines the semantic agent interface (status + capabilities) and provides a generic fallback, but it does not yet provide the runtime plumbing that:

- owns an interpreter instance per running session,
- consumes incremental output (logs / tmux / pty),
- updates/persists semantic status, and
- emits status updates to the UI.

Additionally, we already have a local-only `agent_monitor` background loop that tries to keep DB state in sync with tmux session liveness and ensures log capture via `tmux pipe-pane`.
If we add a new AgentDriver loop without consolidating responsibilities, we risk competing sources of truth for runtime state.

Without this shared, single-authority “driver/supervisor”, T-3/T-4 would each need to reinvent output tailing/cursoring and persistence/broadcast behavior, and the system will be fragile when sessions are started/stopped outside the happy path (server restart, manual tmux kill, etc.).

## Goal

Implement the shared AgentDriver loop as the single executor-side session supervisor:

- **Liveness/attach/log supervision** (replaces `agent_monitor`)
- **Output → semantic status** (runs the interpreter)
- **Edge-triggered event emission** (UI updates without polling)

Establish the seams so T-2 can select the right interpreter and T-3/T-4 can focus on detection logic rather than plumbing.

## Scope / responsibility boundaries

- AgentDriver is responsible for updating persisted runtime truth on `AgentSession` (and any task-scoped pointer/config introduced by T-8).
- Events are notifications derived from those persisted updates; they are not the source of truth.
- The session transport implementation (tmux/pty/process IO) remains separate; AgentDriver consumes its output and uses a narrow transport API for “send text” / “interrupt” when supported.

## Requirements

### 1) Replace and remove the existing `agent_monitor`

- Remove the standalone `redesmyn/agent_monitor.py` loop and replace it with AgentDriver supervision.
- Ensure there is exactly one background loop responsible for:
  - detecting “session is running” vs “session is gone”,
  - attaching/repairing log capture (e.g. `tmux pipe-pane`),
  - maintaining attach metadata (tmux session name, log path),
  - transitioning `AgentSession.status` and timestamps appropriately.

Implementation note:
- Tests and dev currently use `enable_agent_monitor` / `REDESMYN_NO_AGENT_MONITOR`. Preserve a compatible kill-switch (same env var or a deprecated alias) so disabling the loop in tests remains straightforward.

### 2) Local supervisor implementation (tmux-first)

In local runner mode, AgentDriver must supervise tmux-backed sessions:

- Determine active sessions by reading tmux session list and matching the configured prefix (see `REDESMYN_TMUX_SESSION_PREFIX`, default `rn-a`).
- For each active tmux session that corresponds to a task:
  - ensure there is an active `AgentSession` row for the task (create if missing),
  - set/update `status`, `started_at/ended_at`, and `attach.log_path`,
  - ensure log capture is attached (idempotent `pipe-pane`).
- For each DB session marked Running/Blocked with no corresponding tmux session:
  - set session to ended (`ended_at`) and mark `status=error` (or the v0 equivalent), so UI/merge safety checks do not remain stale.

### 3) Own the output cursor and run the interpreter

- Track a per-session cursor so output is consumed incrementally (no “re-read entire log on every tick”).
- Feed only new output to the selected agent interpreter (`consume_output`).
- Persist semantic status updates to the session (capabilities + `turn_state` + detail).
- Update/broadcast only on changes (edge-triggered); avoid noisy DB writes.

### 3.1) Persist external session identifiers (resume handles)

AgentDriver should persist the agent program’s own “resume handle” when available (session-scoped).

Key findings (Jan 2026):

- **Codex**: `thread_id` is the session identifier; use it as the resume handle.
- **Claude Code**: JSON output includes a `session_id`; use it as the resume handle (and treat “continue in cwd” as a separate capability).

This needs to be modeled so “attach” (tmux) and “resume” (new process) can coexist:

- tmux-backed interactive sessions: `attach` is present; external resume handle may be absent.
- programmatic/streaming sessions: external resume handle is present; `attach` may be `none`.

### 4) Emit session/agent status events

### 4) Emit session/agent status events

- Emit events for significant transitions:
  - session lifecycle (`running → error/stopped`, `started_at` set, `ended_at` set),
  - semantic status changes (`turn_state` changes, readiness changes, degraded mode reasons).
- Events should include enough identifiers to correlate in the UI:
  - `task_id`, `agent_session_id`, and the new status payload.

### 5) Testing expectations (must not require tmux)

AgentDriver replaces `agent_monitor`, so we need tests that validate behavioral equivalence without depending on an actual tmux binary.

Add tests that cover (using monkeypatch/fakes for tmux + transport):

- **Liveness sync**
  - creates a session row when a tmux session exists but DB has no active session,
  - updates an existing session row to Running and sets attach metadata,
  - marks a DB-running session Error/ended when tmux session disappears.
- **Log capture hookup**
  - calls “pipe to log” idempotently with the expected target session + path.
- **Semantic status persistence**
  - consumes new output via cursoring,
  - persists semantic status changes to the DB only when it changes.
- **External session id persistence**
  - when the interpreter reports a Codex `thread_id` or Claude `session_id`, AgentDriver persists it on the session (and does not thrash it on every tick).
- **Kill-switch behavior**
  - loop is disabled in tests when the flag/env var is set (mirrors current `enable_agent_monitor` usage).

## Acceptance criteria

- There is exactly one runtime loop supervising sessions in local mode (AgentDriver replaces `agent_monitor`).
- Session status in the DB matches tmux reality (no long-lived “running” sessions after tmux exit).
- Logs continue to be captured and attach metadata continues to point to a valid log path.
- Semantic status updates appear in the DB and can be pushed to the UI via events.
- Tests cover the liveness/log behaviors previously provided by `agent_monitor` without requiring tmux.
