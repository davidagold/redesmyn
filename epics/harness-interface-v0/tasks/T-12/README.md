# T-12 UI: task card messaging (structured + interactive send, resume, interrupt)

## Metadata

```yaml
id: T-12
epic: harness-interface-v0
stacked_on: T-11
branch:
  suggested: rn/harness-interface-v0/T-12-task-card-messaging
linear:
  issue_id: null
  identifier: null
```

## Problem

The UI exposes a “send message” composer when the user expands a task card, but it is not wired up: submitting a message does not actually reach the agent.

This blocks two important workflows:

- **Human → agent steering** without attaching to tmux.
- **Automation** (e.g. conflict remediation) that needs a reliable “send message” capability.

The system already has the right building blocks:

- Structured mode event plumbing (T-10) and structured continuation transport (T-11).
- An event stream that can surface “turn started / turn completed” and assistant messages.

What’s missing is the user-facing “send message” orchestration: picking the right active session, starting/resuming a turn, and handling “turn in progress” safely.

## Goal

Wire the task-card message composer to a single, coherent “send message” capability that works in both:

- **structured mode** (exec + semantic events), and
- **interactive mode** (tmux-style).

This should become the canonical path used by both:

- manual user messaging, and
- automation (e.g. T-5 conflict assist) when it needs to deliver a prompt.

## Requirements

### 1) Define “active session” selection

When the user sends a message from a task card, Redesmyn must resolve which agent session it should target.

v0 policy:

- Prefer the most recent `AgentSession` for the task that is:
  - compatible with the configured harness command (structured vs interactive), and
  - resumable when possible (`external_session_ref` present for structured).
- If there is no suitable session, create one as part of sending the message.

### 2) Structured mode semantics

Treat “send message” as “run one structured turn” (not tmux keystroke injection).

Scenarios:

1) **No resumable id available** (`external_session_ref` missing):
   - Start a new structured exec turn for the task.
   - Capture/persist the resulting resumable id (`external_session_ref`) so subsequent messages resume the same conversation.
   - This becomes the “active session” for later automation (e.g. conflict remediation follow-ups).

2) **Resumable id available**:
   - Start a new structured exec turn **resuming** the existing external session.

3) **Resumable id available AND a turn is already in progress**:
   - Prompt the user: “Interrupt the current turn and send your message?”
   - If the user confirms:
     - issue an interrupt (only if supported), then start the new turn with the message.
   - If the user declines:
     - do not send; keep the draft message (optional: offer “send after completion” as a follow-up enhancement).

Implementation guidance:

- Use the structured continuation transport from T-11 for “resume-by-id” turns.
- Ensure turn boundaries remain deterministic (don’t rely on prompt heuristics).
- Enforce “one turn at a time” per external session (no concurrent structured turns).
- When available, capture the assistant’s **final message** for the turn (see 2.1).

#### 2.1 Structured: capture final assistant message (best-effort)

Motivation: structured streams are ideal for machine-readable progress, but they are not always sufficient for reliably capturing the assistant’s full final natural-language response (e.g. truncation, partial streaming, or consumers that only keep a small message preview). Codex explicitly supports a “write final message to file” flag for this use case.

Requirement:

- When running a structured Codex turn (start or resume), Redesmyn should **best-effort** capture the assistant’s final message for that turn, when supported by the harness.
- The captured final message should be forwarded through the semantic event stream so the UI can display it without attaching to tmux.

Implementation guidance (Codex):

- For `codex exec --json ...` turns, include Codex’s final-message capture flag (per Codex CLI docs), e.g. `--output-last-message <path>` (or `-o <path>`), so the assistant’s final message is written to a file. (Codex docs explicitly recommend pairing `--json` with `--output-last-message` in CI.)
- Prefer a deterministic path under the agent session directory rather than OS temp directories so it:
  - is writable under worktree sandboxing, and
  - is easy to persist/debug (e.g. `.redesmyn/tasks/<task_id>/agent-sessions/<agent_session_id>/codex_last_message.txt`).
- After the turn completes, if the file exists and contains a message:
  - emit a semantic event that includes the final assistant message content (typed payload),
  - update the session’s “message preview” snapshot from this final message (bounded/truncated),
  - and prefer this final message over “reasoning/progress” text for the task-card preview once the turn is complete.

Notes:

- This capture must be **structured-safe**: it must not pollute the structured output stream (i.e. do not rely on tmux keystrokes or prompt scraping).
- This is best-effort: if the harness does not support final-message capture, or the file is missing/empty, fall back to the last observed `agent.assistant_message` event as we do today.

### 3) Interactive mode semantics

Interactive mode cannot rely on structured turn boundaries, but should still be ergonomic and safe.

Scenarios (best-effort):

1) **No running/attachable session exists**:
   - Start the task agent (interactive) and deliver the message after startup.

2) **Session exists and appears ready** (when we can detect readiness):
   - Send the message via the interactive transport (e.g. tmux send-keys / stdin).

3) **Session exists but appears busy** (when we can detect “turn in progress”, or status is clearly non-idle):
   - Prompt the user to interrupt before sending.
   - If interrupt is unsupported, surface a clear warning and allow “send anyway” as an explicit choice.

Notes:

- Interactive “busy” detection should prefer declared capabilities (`can_detect_ready_for_input`, `can_detect_turn_complete`, `can_interrupt`) and semantic status when available.
- Do not invent fragile prompt scraping just to support this feature.

### 4) API contract (control plane)

Introduce an explicit API endpoint that the UI uses for messaging, e.g.:

- `POST /v1/tasks/{task_id}/agent/message`

Request shape should include:

- `message: str`
- `on_conflict: enum` describing what to do if the agent is busy/in progress (v0 options):
  - `fail` (default): return 409 with a machine-detectable conflict kind
  - `interrupt_turn`: interrupt an in-progress turn (when supported) before sending
  - `stop_session_and_start_new`: stop the currently running session and start a new one to deliver the message (destructive; breaks continuity)
- optional `mode_hint` / `preferred_interface_mode` (so UI can be explicit)

Response should include enough for the UI to update immediately:

- `agent_session_id` (the active session that received the message)
- whether a new session was created or an existing one resumed
- any warnings (e.g. “sent to interactive session; no structured turn tracking”)

### 5) UI behavior

- Sending a message must show immediate feedback (ties into UI v0 progress-indicator work):
  - disable the send button while the request is in flight,
  - show a subtle “sending…” animation (no spinners).
- On success:
  - clear the composer input,
  - keep focus behavior sane (don’t collapse the card),
  - rely on subsequent `agent.*` events to update preview/semantic status.
- On failure:
  - show an actionable error (e.g. “agent not running”, “daemon not connected”, “cannot resume session”) and keep the draft message.

## Acceptance Criteria

- Structured mode:
  - First message creates/persists a resumable id when none exists.
  - Subsequent messages resume the same external session and produce new structured turn events.
  - If a turn is in progress, the UI prompts before interrupting.
  - When supported by the harness, the assistant’s final message is captured and shown in the task card after turn completion.
- Interactive mode:
  - Message delivery works without attaching to tmux.
  - “Busy” cases prompt before interrupting/sending when possible.
- The message composer becomes the canonical “send message” surface used by both humans and automation.
