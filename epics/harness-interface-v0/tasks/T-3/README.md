---
epic: harness-interface-v0
branch:
  suggested: rn/harness-interface-v0/T-3-codex-agent
rn:
  linear:
    issue_id: 7fcfebc5-9f96-4f31-8f03-9d8c3e03ab4b
    identifier: RED-20
  parent: T-7
---

# T-3 Codex agent interface implementation (turn detection + capabilities)

## Problem

Codex appears to provide higher-quality signals than a generic terminal process (e.g. “turn done” notifications / predictable UI patterns).
We want to leverage these signals to implement:

- “ready for input” detection
- “turn complete” detection
- safe message delivery for automated workflows (like conflict remediation)

## Goal

Implement a Codex-specific agent interface implementation that:

- reliably detects “turn complete” (within reasonable heuristics and explicit timeouts)
- exposes explicit capabilities so callers can gate advanced features
- publishes semantic status updates to the control plane/UI

## Notes / implementation guidance

## Research requirement

The implementer should do web research (docs + GitHub if open source) to identify the most reliable signals available for Codex completion/idle detection.

This is intentionally not prescriptive about *how* Codex signals completion.
The v0 implementation can use a layered strategy:

1) Prefer explicit signals if available (notifications, OSC sequences, known markers).
2) Fall back to robust heuristics:
   - prompt re-appearance detection
   - idle/quiet windows with a stable prompt state
3) Always include timeouts and “unknown” states (avoid wedging merge runs forever).

### Concrete findings (Jan 2026)

- Codex non-interactive mode supports a structured JSONL stream (`codex exec --json`) that includes:
  - `thread.started` with `thread_id` (session identifier),
  - `turn.started` / `turn.completed` with a per-turn `turn_id`.
- Codex notifications (“agent turn complete”) include `thread-id` (session identifier) and `turn-id` (turn identifier).

Implications for this task:

- Treat Codex `thread_id` as the external resume handle for the Redesmyn session.
- Prefer mapping `turn.*` events into `turn_state` transitions when available; avoid heuristics when the structured stream is present.
- Keep heuristics as a fallback for tmux-first interactive sessions (where structured output may not be available or enabled).

## Acceptance criteria

- Codex sessions produce `turn_complete` events in the common case.
- The implementation never blocks indefinitely: if signals can’t be determined, it returns to `unknown` with an explicit reason.
- The UI can accurately show when Codex is “done” vs “working” vs “unknown”.
