---
id: T-1
stacked_on:
node:
  branch: rn/agent-orchestration/T-1-agent-runtime-architecture
---

# T-1 Agent runtime architecture + contracts

## Brief (local)

- Define the v0 control-plane/runner boundary, including what runs where in local-first vs cloud deployment.
- Specify the data model and contracts for:
  - `Host` (runner host)
  - `AgentSession` (running harness instance)
  - liveness + activity signals (heartbeat, commits, worktree health)
- Specify the harness integration model:
  - profile schema (launch/attach/capabilities)
  - validation/doctor contract (preflight + smoke checks)
  - git enforcement strategy (PATH shim + cooperative skill guidance)
- Specify the minimal WebSocket event vocabulary for “live graph” (agent/session + git/worktree activity).
- Reserve namespaces / extensibility points for messages/commands, but defer detailed modeling and UX to `epics/messages-commands/README.md`.
- Specify the attach/detach strategy (tmux preferred; fallback mode) and how it is represented in session metadata.

## Acceptance Criteria

- `epics/redesmyn/README.md` contains the updated canonical decisions (control plane vs runner, sessions, WebSocket, tmux strategy, host-local worktree paths).
- The v0 API surface for sessions/telemetry/real-time is described (endpoints + payload shapes).
- The event stream contract is described (topics/subscriptions, core presence/activity event types, versioning/resume strategy) with reserved namespaces for messages/commands.
