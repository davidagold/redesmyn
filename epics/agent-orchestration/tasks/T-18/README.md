---
rn:
  node:
    branch: rn/agent-orchestration/T-18-sandboxing
  parent: T-17
---

# T-18 Sandboxing: run harnesses in a controlled environment

## Plan

Allow agents to run with tighter filesystem/network constraints while remaining usable:

- Add a `SandboxProvider` abstraction with a `worktree` policy.
- Ensure common harness tooling still works inside the sandbox:
  - PTY enabled (for CLI harnesses)
  - temp directory writes permitted
  - git worktree metadata writes permitted (so `git` works within a worktree)
  - harness home/state can be provisioned in the sandbox (e.g. Codex credentials/state)
- Expose sandbox config in the dashboard “Configure” panel.

## Acceptance Criteria

- Users can choose sandbox policy + network behavior via repo config and see it in the UI.
- Starting an agent under sandbox does not break basic workflows (git, harness auth/state, logs).
- Failures surface actionable errors (not silent no-ops).
