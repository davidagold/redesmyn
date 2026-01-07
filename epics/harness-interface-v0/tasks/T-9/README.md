# T-9 UI: task card agent message preview (1-line, truncated)

## Metadata

```yaml
id: T-9
epic: harness-interface-v0
stacked_on: T-10
branch:
  suggested: rn/harness-interface-v0/T-9-agent-message-preview
linear:
  issue_id: 847901fe-17cd-418b-890d-f105558f747b
  identifier: RED-26
```

## Problem

Once we implement structured agent integration (T-10), we need a clear UI “upshot” that:

- makes the graph feel alive (agent output is visible without attaching to tmux),
- stays compact (no transcript blocks under every card), and
- creates a foundation for a later Timeline/Activity rail.

Today, agent output exists primarily as logs and an Agent details section. That is useful, but it does not provide an at-a-glance sense of “what the agent just said” while you’re scanning the graph.

## Goal

Add a minimal on-graph affordance:

- each task card can optionally show a single **one-line, truncated** preview of the most recent **assistant message** emitted by the agent (e.g. Codex), updated live.

This should be a small, non-busy enhancement: it should not re-layout the graph dramatically and should not introduce “chat UI” density on every node.

## Using T-10 (structured exec + semantic events)

T-10 already provides the backend plumbing this task should consume:

- Structured mode is **command-driven**:
  - **Codex** is structured only when the harness command includes `exec` and `--json` (e.g. `codex exec --json …`).
  - **Claude Code** is structured only when the harness command includes `--print`/`-p` and `--output-format stream-json` (e.g. `claude --print --output-format stream-json …`).
- AgentDriver persists a session-scoped `agent_preview` JSON payload onto `AgentSession` and updates it when it observes structured `agent.assistant_message` events.
- The preview is included in the `task.agent_session_update` event snapshots (so the UI can update live without polling).

This task should **not** scrape interactive tmux logs. If a session does not emit structured assistant-message events, the preview should simply remain absent.

## Requirements

### 1) Display the persisted preview on the task card

Use `AgentSession.agent_preview.last_assistant_message_preview` (and optionally `…_at`) to render a single-line preview on the task card:

- always single line (line clamp 1),
- truncation with subtle fade/ellipsis,
- only shown when preview is present (no placeholder / “not available” state),
- visually secondary to title/status (avoid heavy borders or chat bubbles).

### 2) Live updates (no polling)

Ensure the graph updates live when the preview changes:

- listen to `task.agent_session_update` events and apply the updated `agent_preview` to task nodes in the client cache/state
- do not add a separate polling loop for previews

### 3) Testing expectations

Add tests that validate:

- UI renders preview safely and does not break layout when missing/empty.

## Acceptance criteria

- When Codex emits a new assistant message, the corresponding task card shows a single-line preview within ~1s.
- The graph remains readable and non-busy: no multi-line transcripts on cards.
- The preview data is persisted on `AgentSession` (so it survives reloads) and can serve as a future Timeline source.
