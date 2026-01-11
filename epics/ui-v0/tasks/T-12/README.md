# T-12 UI: show in-flight progress for all actions (no “silent” requests)

## Metadata

```yaml
id: T-12
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-12-inflight-progress-indicators
```

## Problem

Too many UI actions initiate backend work (API requests, daemon commands, git operations) without showing any immediate indication that something is happening.
This creates a “did my click work?” moment and encourages double-clicking / repeated actions, which can produce confusing states.

We need a consistent, tasteful pattern so that **every user-initiated request** has an associated visible “in progress” signal.

## Goal

Whenever the user triggers an action that can take non-trivial time (including “submit + wait”), the UI should:

- show a clearly visible but refined progress indication immediately,
- prevent accidental duplicate requests, and
- revert/transition cleanly on success or failure.

Constraints:

- No spinning loader wheels.
- Shimmering text is reserved for LLM generation only.
- Prefer refined-but-visible patterns: animated ellipses, subtle glow/pulse, or other low-noise motion.

## Requirements

### 1) Define a shared UI pattern for in-flight state

Introduce a single reusable approach that can be applied broadly:

- a shared component (e.g. `AsyncActionButton`) and/or
- a shared hook/util that wraps mutations and exposes an “in flight” state.

The pattern must support:

- “pending request” (HTTP in flight),
- “submitted background work” (e.g. merge run started) where completion is event-driven, and
- “failed” state with an actionable error surface.

### 2) Apply to core workflows (minimum set)

Ensure immediate progress indication exists for at least:

- merge / restack / merge-run actions,
- agent lifecycle actions (start/restart/stop),
- “mark merge ready” / other task state mutations,
- daemon/executor-related actions,
- any other long-ish git-related actions surfaced in the UI.

### 3) Visual design guidance

Use subtle motion and clear affordances:

- Animated ellipsis after the action label (e.g. “Merging…”).
- Subtle glow border/pulse (T-9 in harness-interface-v0 is expected to introduce a glow style; reuse if appropriate).
- Avoid adding borders everywhere; keep the UI calm and consistent.

### 4) Behavior and accessibility

- Actions should disable while in flight (unless explicitly safe to run concurrently).
- Keyboard interactions should remain predictable (no focus traps).
- Progress indicators should be visible in both light and dark themes.
- Ensure the user can still discover what’s happening (e.g. via callouts/events) for longer-running operations.

## Acceptance Criteria

- Every action the user can trigger produces an immediate, clearly visible “in progress” indication.
- There are no “dead air” interactions where the UI appears unchanged for >100ms after a click.
- The UI remains calm: no spinners, no noisy attention-grabbing animation, and shimmer text remains reserved for LLM generation.
