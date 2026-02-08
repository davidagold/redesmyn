---
epic: gpui
branch:
  suggested: rn/gpui/T-36-daemon-shell-tmux-runtime
rn:
  node:
    branch: rn/gpui/T-36-daemon-shell-tmux-runtime
  parent: T-23
---

# T-36 Daemon Shell (tmux) session runtime (attach/send/interrupt + log artifacts) (Domain 4)

## Problem

We need a compatibility path for “shell-like” agents that do not expose reliable structured semantics.

We still want:

- a stable way to start/stop/attach,
- reliable “send message” for interactive sessions (keystrokes),
- interrupt semantics,
- and durable artifacts/logs that can be surfaced in the UI.

If we treat shell sessions as second-class, we will lose core functionality during the port.

## Goal

Implement daemon-side support for **Shell (tmux)** sessions as a first-class agent runtime:

- start a tmux session bound to a task/worktree,
- attach to it,
- send text (with optional interrupt),
- capture output into artifacts,
- and emit durable session events for the parts we can know deterministically (user messages sent, session start/stop, artifacts).

## Requirements

### 1) Naming + semantics

Use `Shell` terminology (not “Generic”) in Rust-facing surfaces.

Shell sessions are explicitly “interactive/unstructured”:

- we do not pretend to parse assistant messages reliably,
- but we still provide useful UX via log artifacts and sent-message history.

### 2) tmux integration

Implement a small tmux facade:

- list sessions (best-effort),
- start session per task,
- attach,
- send text,
- send ctrl-c,
- pipe pane to log file (or equivalent).

Ensure behaviors are:

- safe (avoid accidental cross-task session interference),
- deterministic where possible,
- and robust when tmux server is not running.

### 3) Durable artifacts + events

Even for Shell sessions, emit durable records:

- `SessionStarted` / `SessionEnded`
- `UserMessage` for messages we inject
- `ArtifactEmitted` for log outputs / snapshots

Rule: do not stuff huge terminal output into session events; use artifacts (T-14).

### 4) “Send message” compatibility

Shell message delivery must preserve the user-visible behavior:

- optional interrupt before sending,
- submit/enter behavior,
- and clear error reporting when no session is running.

### 5) Test strategy

Do not require real tmux in CI if we can avoid it:

- implement the tmux facade behind a trait,
- use a fake implementation for deterministic tests.

## Acceptance criteria

- The daemon can start a Shell/tmux session for a task and provide an attach handle.
- The daemon can send text + interrupt into the session.
- Output is captured as an artifact and referenced by session events.
- A deterministic test covers:
  - start → send text → stop → artifact emitted.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on daemon runtime skeleton (T-23) and worktree mapping (T-27).
- Uses artifact/session event contract (T-14).

## Reference implementation (today; for behavior orientation only)

- tmux runtime (Python today):
  - `redesmyn/agent_runtime.py` (`send_task_agent_text`, tmux helpers).
  - `redesmyn/agent_driver.py` (pipes pane to log; derives “running” by tmux session presence).
- Tests (Python today):
  - `tests/test_agent_driver.py` (fake tmux supervision patterns).

