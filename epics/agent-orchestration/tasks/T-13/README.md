---
rn:
  node:
    branch: rn/agent-orchestration/T-13-harness-amp
  parent: T-15
---

# T-13 Harness adapter: Amp

## Plan

- Add an Amp launch configuration (launch/attach/capabilities) and verify it works end-to-end with the generic runner/session model.
- Identify any available hooks and use them opportunistically to enrich session state (optional).

## Acceptance Criteria

- Amp is runnable via a profile (no bespoke code required unless validation proves necessary).
- `rn agent doctor amp` passes (or reports explicit, actionable degraded-mode warnings).
- Sessions behave consistently with the runner/session model and surface actionable errors when unsupported features are requested.
