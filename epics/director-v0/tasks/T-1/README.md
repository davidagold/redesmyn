---
rn:
  node:
    branch: rn/director-v0/T-1-director-run-semantics
  parent: null
---

# T-1 Director run semantics (cursor + idempotency)

## Plan

- Specify the director’s run model:
  - triggers (manual, periodic, “wake on new event”)
  - how concurrent wakeups are handled (coalescing is OK)
- Specify event consumption semantics:
  - persistent cursor
  - per-run high-water mark
  - exactly-which events are visible to the director in a run
- Specify how the director produces intents:
  - intent types (merge actions, gate requests, “request changes”, etc.)
  - idempotency keys so the same intent isn’t re-issued on replay
- Specify the minimal state the director must persist (cursor + last run metadata).

## Acceptance Criteria

- The “new events while running” story is explicit and does not require a separate “ack table”.
- The director can be restarted without losing correctness (cursor replay is sufficient).
- Intent emission is idempotent and safe under retries.
