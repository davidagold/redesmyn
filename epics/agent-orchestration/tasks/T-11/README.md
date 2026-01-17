---
rn:
  node:
    branch: rn/agent-orchestration/T-11-harness-claude-code
  parent: T-15
---

# T-11 Harness adapter: Claude Code

## Plan

- Add a Claude Code launch configuration (launch/attach/capabilities) and verify it works end-to-end with the generic runner/session model.
- Use hooks if available to enrich session state and message/command integration (optional, best-effort).
- Document limitations and the recommended UX when hooks/attach are constrained.

## Acceptance Criteria

- Claude Code is runnable via a profile (no bespoke code required unless validation proves necessary).
- `rn agent doctor claude-code` passes (or reports explicit, actionable degraded-mode warnings).
- Sessions behave consistently with the runner/session model; hooks (if present) only enrich.
