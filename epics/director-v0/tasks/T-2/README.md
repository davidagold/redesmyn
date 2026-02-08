---
rn:
  node:
    branch: rn/director-v0/T-2-merge-queue-model
  parent: T-1
---

# T-2 Merge queue model + conductor actions

## Implementation Boundary

- Implement queue state, conductor actions, and orchestration execution in Rust control-plane/desktop surfaces.
- Do not add or rely on legacy Python/webview implementations for this task.

## Plan

- Define the merge queue primitives:
  - queue item identity and state machine (draft → ready → mergeable → merged/blocked/etc.)
  - dependency/blocker modeling (A approved pending B; A needs follow-up wiring commit)
- Define action authority model on top of the queue:
  - director agent chooses queue actions and executes via `rn`
  - conductor can override/pause/approve policy-sensitive transitions
  - command outcomes can also mutate queue state (e.g. merge success/failure, request-changes completion)
- Define the conductor interaction model:
  - approve/reject/defer/requeue
  - request changes (with a structured reason)
  - “approve pending” semantics to support staged merges
- Define queue event semantics for controller/director integration:
  - queue-affecting changes produce durable events
  - queue transitions include source actor (`director`, `conductor`, `system-command-result`)
- Ensure the model cleanly represents:
  - Scenario 1 (A depends on B; A approved pending B; B merges; A updated; A merges)
  - Scenario 2 (A/B also need to account for C; queue re-evaluation)

## Acceptance Criteria

- The queue can represent pending dependencies and staged approvals without inventing ad-hoc states.
- Director-driven actions and conductor overrides are both explicit, durable, and explainable.
- The director can re-evaluate ordering when new information arrives (e.g. task C appears).
- Queue-affecting updates from non-director actors are visible and durable for wake-trigger evaluation.
- Implementation targets Rust runtime paths (control plane + desktop integration), not legacy paths.
