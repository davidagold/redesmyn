---
id: T-14
stacked_on: T-15
node:
  branch: rn/agent-orchestration/T-14-harness-opencode
---

# T-14 Harness adapter: OpenCode

## Brief (local)

- Add an OpenCode launch configuration (launch/attach/capabilities) and verify it works end-to-end with the generic runner/session model.
- Identify any available hooks and use them opportunistically to enrich session state (optional).

## Acceptance Criteria

- OpenCode is runnable via a profile (no bespoke code required unless validation proves necessary).
- `rn agent doctor opencode` passes (or reports explicit, actionable degraded-mode warnings).
- Sessions behave consistently with the runner/session model and surface actionable errors when unsupported features are requested.
