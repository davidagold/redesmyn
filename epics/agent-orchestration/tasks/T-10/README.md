---
rn:
  node:
    branch: rn/agent-orchestration/T-10-harness-codex
  parent: T-15
---

# T-10 Harness adapter: Codex

## Brief (local)

- Add a Codex launch configuration (launch/attach/capabilities) and verify it works end-to-end with the generic runner/session model.
- If Codex supports hooks, use them opportunistically to enrich session state; otherwise rely on generic mechanisms (process lifecycle + repo observer).
- Document the recommended workflow for Codex (including any skill-based guidance and degraded-mode notes).

## Acceptance Criteria

- Codex is runnable via a profile (no bespoke code required unless validation proves necessary).
- `rn agent doctor codex` passes (or reports explicit, actionable degraded-mode warnings).
- Starting a Codex session produces reliable liveness updates and commit activity via the repo observer.
