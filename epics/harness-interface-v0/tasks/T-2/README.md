# T-2 Agent kind identification + user override (Codex/Claude/Generic)

## Metadata

```yaml
id: T-2
epic: harness-interface-v0
stacked_on: T-7
branch:
  suggested: rn/harness-interface-v0/T-2-agent-kind-identification
linear:
  issue_id: 8923f91c-4b6c-418f-9e18-83becf08f4d8
  identifier: RED-19
```

## Problem

To offer agent-specific features safely, we must know which agent kind/interpreter is in use.
Today, the user provides a shell command (e.g. `codex …`, `claude …`, or arbitrary) and we treat it as opaque.

We want:

- best-effort automatic detection (to keep UX smooth)
- explicit user override (to avoid “wrong inference” footguns)
- clear persistence semantics (what does “this task uses Codex” mean across restarts?)

## Goal

Add a first-class “agent kind” selection mechanism, defaulting to **Auto**, with options:

- Auto (infer from command)
- Generic
- Codex
- Claude Code

This selection drives which agent interpreter is used, which capabilities are available, and whether features like conflict auto-assist can be enabled.

## Research requirement

The implementer should do web research on how these CLIs are invoked in practice (wrappers, subcommands, common installation paths).
The goal is not perfect detection, but a robust best-effort inference that matches real-world usage patterns.

## Requirements

### 1) Best-effort inference from command

Infer agent kind from the configured agent command (and/or resolved argv):

- if argv[0] looks like `codex` → Codex
- if argv[0] looks like `claude` (Claude Code CLI) → Claude Code
- else → Generic

Inference should be:

- deterministic and explainable
- tolerant of wrappers (e.g. `uv run codex`, `npx claude`, etc.) if feasible
- tolerant of common subcommands that appear in programmatic runs (e.g. `codex exec …`, `claude -p …`)

### 2) User override

Provide a way for the user to override the inferred agent kind:

- UI: in the Configure panel, a small selector (Auto / Generic / Codex / Claude Code)
- CLI: optional flag on start/restart (and/or config) to set agent kind explicitly

When overridden, the selected agent kind must be persisted as part of the task/session config so restarts behave predictably.

### 2.1) Interaction with external “resume handles”

Key findings (Jan 2026):

- Codex uses `thread_id` as its session identifier (resume handle).
- Claude Code JSON output includes a `session_id`.

If an `AgentSession` already has a persisted external resume handle, treat that as strong evidence of the agent kind (and avoid “flapping” between kinds on restart unless the user explicitly overrides).

### 3) Capability-driven feature gating

The UI must not offer features that the selected agent kind does not support.
Examples:

- conflict auto-assist requires `can_detect_turn_complete` and `can_send_text`
- (future) notifications require `can_receive_notifications`

### 4) UX copy

Communicate the meaning of selection:

- Auto: “We inferred this agent from your command; switch if wrong.”
- Generic: “Works with any command, but advanced features are disabled.”

## Acceptance criteria

- Starting/restarting an agent uses the selected agent interpreter (Auto→inferred, override→forced).
- Users can change the agent kind when an agent is stopped/errored (safe to change).
- The UI clearly indicates when advanced features are unavailable because the agent is Generic/unknown.
