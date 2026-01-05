# T-2 Harness identification + user override (Codex/Claude/Generic)

## Metadata

```yaml
id: T-2
epic: harness-interface-v0
stacked_on: T-1
branch:
  suggested: rn/harness-interface-v0/T-2-harness-identification
```

## Problem

To offer harness-specific features safely, we must know which harness implementation is in use.
Today, the user provides a shell command (e.g. `codex …`, `claude …`, or arbitrary) and we treat it as opaque.

We want:

- best-effort automatic detection (to keep UX smooth)
- explicit user override (to avoid “wrong inference” footguns)
- clear persistence semantics (what does “this task uses Codex” mean across restarts?)

## Goal

Add a first-class “harness kind” selection mechanism, defaulting to **Auto**, with options:

- Auto (infer from command)
- Generic
- Codex
- Claude Code

This selection drives which harness adapter is used, which capabilities are available, and whether features like conflict auto-assist can be enabled.

## Requirements

### 1) Best-effort inference from command

Infer harness kind from the configured harness command (and/or resolved argv):

- if argv[0] looks like `codex` → Codex
- if argv[0] looks like `claude` (Claude Code CLI) → Claude Code
- else → Generic

Inference should be:

- deterministic and explainable
- tolerant of wrappers (e.g. `uv run codex`, `npx claude`, etc.) if feasible

### 2) User override

Provide a way for the user to override the inferred harness kind:

- UI: in the Configure panel, a small selector (Auto / Generic / Codex / Claude Code)
- CLI: optional flag on start/restart (and/or config) to set harness kind explicitly

When overridden, the selected harness kind must be persisted as part of the agent config/session so restarts behave predictably.

### 3) Capability-driven feature gating

The UI must not offer features that the selected harness does not support.
Examples:

- conflict auto-assist requires `can_detect_turn_complete` and `can_send_text`
- (future) notifications require `can_receive_notifications`

### 4) UX copy

Communicate the meaning of selection:

- Auto: “We inferred this harness from your command; switch if wrong.”
- Generic: “Works with any command, but advanced features are disabled.”

## Acceptance criteria

- Starting/restarting an agent uses the selected harness adapter (Auto→inferred, override→forced).
- Users can change the harness kind when an agent is stopped/errored (safe to change).
- The UI clearly indicates when advanced features are unavailable because the harness is Generic/unknown.

