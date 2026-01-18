---
epic: gpui
branch:
  suggested: rn/gpui/T-66-session-viewer-testability
rn:
  parent: T-48
  after:
    - T-59
    - T-63
    - T-64
---

# T-66 Session viewer AI-testability: UI driver actions + semantic snapshot + tests (Domain 7)

## Problem

The session viewer is one of the most important user experiences in the app and must be easy for AI (and CI) to test.

If we rely on pixel screenshots or coordinate clicks, tests will be:

- flaky,
- slow,
- and hard to debug.

## Goal

Extend the desktop UI driver (T-48) and semantic UI snapshot surfaces so AI-driven tests can:

- create and pin a chat session,
- send messages,
- observe durable session history updates,
- and assert key UI states (pending, conflicts, new messages indicator, etc.).

## Requirements

### 1) Driver actions (high-level)

Add UI driver actions sufficient to exercise SessionView:

- `OpenEpic { epic }`
- `OpenPinnedChatSession` (no-op if none; returns session_id?)
- `CreateChatSessionAndPinToEpic { title? } -> { session_id }`
- `UnpinChatSessionFromEpic { epic_id }`
- `CloseChatSession { session_id }`

Session viewer interactions:

- `SessionCompose { text }`
- `SessionSend` (or `SessionComposeAndSend`)
- `SessionResolveConflict { action: fail|interrupt|stop_and_start_new }`
- `SessionScrollToBottom`
- `SessionLoadOlder`

### 2) Semantic UI snapshot additions

Expose stable, machine-readable state including:

- current epic id/slug
- pinned chat session id (0/1) for current epic
- selected task id (if any) + latest task session id (if shown)

Session view state (for whichever session view is visible/focused):

- `session_id`
- `session_kind` (task|chat)
- `render_mode` (feed|interactive_placeholder)
- `composer`:
  - `draft_len`
  - `send_pending`
  - `last_error` (short, actionable)
  - `conflict_prompt` (kind + available actions)
- `feed`:
  - `visible_item_count`
  - `at_bottom`
  - `new_messages_pending` (bool)
  - `loading_older` (bool)

### 3) Waiting primitives integration

Tests should be able to:

- `WaitForIdle` (quiescence window),
- `WaitForSessionEvent` (e.g. “assistant message appended”), and/or
- `WaitForUiPredicate` (e.g. “new_messages_pending == false”).

Use the global wait primitives contract (T-15) rather than sleeps.

### 4) Tests

Add at least one deterministic test that:

1) launches the desktop app in fixture mode,
2) selects an epic,
3) creates and pins a chat session,
4) sends a message,
5) observes a durable `UserMessage` event + a simulated `AssistantMessage` event (mock daemon),
6) asserts UI snapshot states across the workflow.

Optionally add a test for the interactive placeholder:

- show an interactive session and assert attach command and pending state transitions.

## Acceptance criteria

- A session viewer workflow can be executed end-to-end via UI driver actions with no coordinate clicks.
- The semantic snapshot contains enough information to make assertions without pixel diffs.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on:
  - UI driver foundations (T-48),
  - session viewer foundations (T-59),
  - left pane + task details integrations (T-63/T-64).
