---
epic: gpui
branch:
  suggested: rn/gpui/T-35-daemon-exec-session-supervisor
rn:
  parent: T-23
---

# T-35 Daemon exec-session supervisor (process lifecycle + event streaming + backpressure) (Domain 4)

## Problem

For structured agents (Codex, Claude Code, future app-server), we need a daemon-owned runtime that:

- spawns processes in the correct repo/worktree context,
- streams structured session events to the control plane,
- supports interrupts/cancellation,
- and never wedges or floods memory when output is large.

If we implement agent runners ad-hoc per agent kind, we will duplicate process lifecycle logic and create inconsistent session semantics.

## Goal

Build a daemon-side **exec session supervisor** that is the shared substrate for structured agents.

This supervisor owns:

- process spawn/kill/interrupt,
- stdout/stderr ingestion,
- parser invocation (Codex/Claude parsers from T-33/T-34),
- durable session event emission (T-14),
- and backpressure + size limits.

## Requirements

### 1) Explicit lifecycle and state machine

Define a clear runtime model:

- sessions are per-task, but must allow multiple historical sessions per task.
- at most one *active* session per task per interface mode (structured vs interactive) unless explicitly allowed.

Expose a minimal API internally (names are illustrative):

- `start_session(task_id, spec) -> session_id`
- `interrupt_session(session_id)`
- `stop_session(session_id)` (graceful then hard kill)
- `shutdown()` (graceful supervisor shutdown)

### 2) Output ingestion + parsing

- Read stdout/stderr incrementally (non-blocking, bounded buffering).
- Feed text into the selected parser (Codex/Claude/app-server).
- Emit semantic events (ultimately `SessionEvent`) derived from parser output.

### 3) Backpressure, size limits, and “no giant payloads”

Hard requirements:

- Durable session events must respect T-14 size intent: no giant blobs.
- Streaming/chunk/delta text may be used for live UI, but it must be:
  - bounded,
  - optional,
  - and not required for persisted conversation history.

If output is huge:

- store content as an artifact (T-14 `ArtifactRef`),
- and emit a small `ArtifactEmitted` session event referencing it.

### 4) Interrupt semantics

Implement a principled interrupt story:

- interrupt is best-effort and agent-dependent:
  - for exec-based agents, default is SIGINT then escalate.
- the runtime must report command/session state transitions clearly to avoid silent failures.

### 5) Repo/worktree integration (no path leakage)

- The daemon starts sessions only within attached repos (T-24).
- Worktree selection is by stable ids (scope) + task/worktree mapping (T-27).
- Control plane never passes paths; paths remain daemon-local.

### 6) Observability + determinism

- Use tracing spans with stable ids (host_id, repo_id, task_id, session_id).
- Provide deterministic tests using mock agent processes (do not require Codex/Claude installed).

## Acceptance criteria

- The daemon can start a mock exec agent, ingest output, and emit session events to the control plane stream.
- Backpressure behavior is correct (no unbounded memory growth on large output).
- Interrupt/stop works and yields observable lifecycle updates (no silent actions).
- A deterministic integration test covers:
  - start session → emit events → stop session.

## Dependencies / sequencing

- Depends on daemon runtime skeleton (T-23).
- Depends on repo attachment + worktrees (T-24, T-27).
- Uses session event contract (T-14) and parsers (T-33/T-34).

## Reference implementation (today; for behavior orientation only)

- Agent runtime + launcher (Python today):
  - `redesmyn/agent_runtime.py` (`start_task_agent`, `stop_task_agent`, `run_task_agent_resume_by_id_turn`).
  - `tests/test_agent_runtime_launcher.py` (launcher script semantics in v0).
- Session supervision loop (Python today):
  - `redesmyn/agent_driver.py` (`supervise_once`, incremental log read, semantic event emission).
  - `tests/test_agent_driver.py`

