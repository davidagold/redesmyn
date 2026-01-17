---
rn:
  node:
    branch: rn/agent-orchestration/T-5-messages-commands
  parent: T-3
---

# T-5 Messages + commands (moved to separate epic)

## Plan

Messages and commands are intentionally being designed and implemented in a separate epic:

- `epics/messages-commands/README.md`

This stub remains here as a reminder to keep the event stream and session model extensible, but it is not on the critical path for agent/git activity dogfooding.

## Acceptance Criteria

- `T-6` and the dashboard “presence/activity” work can proceed without implementing messaging/commands.
- The agent-orchestration event stream and session schemas reserve space for later message/command events.
