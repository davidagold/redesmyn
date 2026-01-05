# T-7 AgentDriver (output → semantic status)

## Metadata

```yaml
id: T-7
epic: harness-interface-v0
stacked_on: T-1
branch:
  suggested: rn/harness-interface-v0/T-7-agent-driver
```

## Problem

T-1 defines the semantic “agent program” interface (status + capabilities) and provides a generic fallback, but it does not yet provide the runtime plumbing that:

- owns an interpreter instance per running session,
- consumes incremental output (logs / tmux / pty),
- updates/persists semantic status, and
- optionally broadcasts status updates to the UI.

Without this shared “driver”, T-3/T-4 would each need to reinvent output tailing/cursoring and persistence/broadcast behavior, which will make downstream workflows (T-5) harder to build reliably.

## Goal

Implement the shared AgentDriver loop that turns session output into `turn_state`/readiness updates (starting with Generic behavior), and establish the seams so T-2 can select the right interpreter and T-3/T-4 can focus on detection logic rather than plumbing.

