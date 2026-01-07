# T-9 UI: task card agent message preview (1-line, truncated)

## Metadata

```yaml
id: T-9
epic: harness-interface-v0
stacked_on: T-10
branch:
  suggested: rn/harness-interface-v0/T-9-agent-message-preview
```

## Problem

Once we implement Codex integration (T-3), we need a clear UI “upshot” that:

- makes the graph feel alive (agent output is visible without attaching to tmux),
- stays compact (no transcript blocks under every card), and
- creates a foundation for a later Timeline/Activity rail.

Today, agent output exists primarily as logs and an Agent details section. That is useful, but it does not provide an at-a-glance sense of “what the agent just said” while you’re scanning the graph.

## Goal

Add a minimal on-graph affordance:

- each task card can optionally show a single **one-line, truncated** preview of the most recent **assistant message** emitted by the agent (e.g. Codex), updated live.

This should be a small, non-busy enhancement: it should not re-layout the graph dramatically and should not introduce “chat UI” density on every node.

## Requirements

### 1) Persist a “preview” snippet on `AgentSession` (not the full transcript)

We should not store full transcripts in the DB as part of v0. We only need a stable snippet for UI preview and Timeline integration.

Add a session-scoped persisted preview model (shape can evolve, but must be typed/validated):

- `last_assistant_message_preview: str | null` (already truncated; max length enforced)
- `last_assistant_message_at: datetime | null`
- optional: `last_message_turn_id: str | null` (if the agent provides it; informational)

Implementation guidance:

- Prefer a JSON column validated by a Pydantic model (repo convention), e.g. `agent_preview` with an `AgentPreview` model.
- Ensure defaults are valid JSON across SQLite/Postgres (avoid the 0017 default pitfall).

### 2) Wire preview updates through the AgentDriver

AgentDriver (T-7) is the single supervisor and should own:

- consuming output (tmux log tail / programmatic stream),
- interpreting it via the agent backend, and
- persisting derived state onto the `AgentSession`.

Codex backend (T-3) should surface assistant message text in a way the driver can consume without UI coupling.

Suggested seam (additive to v0):

- extend `AgentEvent` with an optional message event (e.g. `agent_message` with `{role, text, external_turn_id?}`), or
- add a dedicated backend property for “last assistant message” (but keep it session-scoped and edge-triggered).

### 3) UI: show a single-line preview on the task card

Update the graph node card UI to render a one-line preview when available:

- always single line (line clamp 1),
- truncation with subtle fade/ellipsis,
- only shown when there is a recent assistant message preview,
- visually secondary to title/status (avoid heavy borders or chat bubbles).

Copy/layout guidance:

- do not add “Agent:” labels; the affordance should read naturally.
- avoid duplicating what the status badge already conveys (this is content, not state).

### 4) Eventing and live updates

Preview changes must update live without polling:

- include the preview fields in the existing `task.agent_session_update` payload, or emit a dedicated event (but keep a coherent event model).
- ensure the UI updates the selected node and any visible node cards when the event arrives.

### 5) Testing expectations

Add tests that validate:

- Codex backend extracts assistant message text from the chosen signal path(s) (stream-json / JSONL / heuristics).
- AgentDriver persists preview only when changed (no thrash).
- UI renders preview safely and does not break layout when missing/empty.

## Acceptance criteria

- When Codex emits a new assistant message, the corresponding task card shows a single-line preview within ~1s.
- The graph remains readable and non-busy: no multi-line transcripts on cards.
- The preview data is persisted on `AgentSession` (so it survives reloads) and can serve as a future Timeline source.
