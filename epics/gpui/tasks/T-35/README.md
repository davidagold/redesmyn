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

Architecture note (important):

- Despite the “exec” name, this ticket should produce a **reusable subprocess supervision core**
  (process lifecycle + bounded IO + backpressure) that can support multiple structured runtime kinds.
- Protocol-specific framing/decoding must live behind **adapter types** (runtime-kind-specific),
  so we do not bake “JSONL on stdout” assumptions into the supervisor core.
- Concretely, we expect:
  - Structured exec runners (T-37/T-38) to use an adapter that feeds decoded records into
    the Codex/Claude parsers (T-33/T-34).
  - App-server runners (T-39/T-68) to use an adapter that speaks a bidirectional framed protocol
    (e.g. JSON-RPC with Content-Length framing) and maps notifications/diffs into `SessionEvent`s
    (T-14).

This supervisor owns:

- process spawn/kill/interrupt,
- stdout/stderr ingestion,
- wire decoding + semantic mapping via adapters (parsers for structured exec; protocol codecs for app-server),
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

Also define explicit seams (this is what keeps app-server migration easy):

- The daemon exposes a **provider/runtime-agnostic** session runtime surface (start/stop/interrupt/send-turn), consumed by the control plane.
- The supervisor is an internal implementation detail: structured exec runners (T-37/T-38) and app-server runners (T-39/T-68) must be implemented
  behind interfaces/traits so the control plane never needs to know whether a turn was satisfied by:
  - `codex exec --json … resume <thread_id> -` (structured exec), or
  - JSON-RPC `userMessage` (app-server).

### 2) Output ingestion + parsing

- Read stdout/stderr incrementally (non-blocking, bounded buffering).
- Route bytes through a runtime-kind adapter:
  - Structured exec: decode records (often JSONL) → feed into the selected parser (T-33/T-34).
  - App-server: apply protocol framing/decoding (T-39/T-68) and dispatch request/response/notifications.
- Emit semantic events (ultimately `SessionEvent`) derived from adapter output.
  - Turn boundary events (`TurnStarted` / `TurnCompleted`) should carry `external_session_ref` when available.

Supervisor-core requirement:

- The core must support **bidirectional** IO so an app-server adapter can write requests to stdin
  while concurrently reading framed responses/notifications (with backpressure).
- For exec-based sessions, stdin is treated as **closed by default** (no implicit piping). Any
  prompts/config must be passed via argv/env/artifacts until T-39 introduces an explicit
  bidirectional IO adapter.

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
- stdout/stderr logs are treated as artifacts as well (debuggable by default); small-log retention is
  configurable but defaults to keeping logs and emitting references.

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

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on daemon runtime skeleton (T-23).
- Depends on repo attachment + worktrees (T-24, T-27).
- Uses session event contract (T-14).
- For structured exec adapters: uses parsers (T-33/T-34).
- App-server runtime work (T-39/T-68) builds on the same lifecycle/backpressure conventions, but
  implements its own protocol framing/decoding.

## Reference implementation (today; for behavior orientation only)

- Agent runtime + launcher (Python today):
  - `redesmyn/agent_runtime.py` (`start_task_agent`, `stop_task_agent`, `run_task_agent_resume_by_id_turn`).
  - `tests/test_agent_runtime_launcher.py` (launcher script semantics in v0).
- Session supervision loop (Python today):
  - `redesmyn/agent_driver.py` (`supervise_once`, incremental log read, semantic event emission).
  - `tests/test_agent_driver.py`
