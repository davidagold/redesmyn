---
rn:
  node:
    branch: rn/agent-orchestration/T-17-agent-model-simplify
  parent: T-16
---

# T-17 Agent model + status: simplify + make tmux the source of truth

## Plan

Simplify the agent/session ontology and make status reflect reality:

- Model an agent as a runner-backed resource for a task (tmux-first).
- Ensure agent status reflects the actual tmux/process state:
  - no tmux session ⇒ no running agent
  - stopping an agent should reflect “stopped” (not “idle”)
- Keep the UX graph-first and clear about what is and isn’t running.

## Acceptance Criteria

- Starting/stopping/restarting agents results in accurate, stable status in the dashboard.
- Status is derived from runner reality (tmux session/process presence), not guesswork.
- The model is simpler than the prior AgentSession split while remaining extensible.
