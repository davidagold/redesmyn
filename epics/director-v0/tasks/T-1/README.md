---
rn:
  node:
    branch: rn/director-v0/T-1-director-run-semantics
  parent: null
---

# T-1 Director run semantics (cursor + idempotency)

## Implementation Boundary

- Specify and implement these semantics in Rust control-plane/desktop integration paths only.
- Do not add or rely on legacy Python/webview implementations for this task.

## Plan

- Specify the director/controller lifecycle:
  - director is an epic-pinned LLM session
  - controller wakes director on significant event changes
  - concurrent wakeups are coalesced deterministically
- Specify event-consumption semantics:
  - persistent cursor for acknowledged consumption
  - high-water mark semantics at wake boundaries
  - exact visibility rules for "events available in this wake"
- Specify action idempotency around direct `rn` execution by the director:
  - repeated wake processing must not cause duplicate task starts/merges/requeues
  - command-side idempotency keys and/or state guards are explicit
- Specify the minimal persisted state:
  - per-epic director session identity
  - event cursor / last acknowledged event
  - last wake metadata (reason, size, timestamp)
  - last known director mode state (active/paused/error/resume-required)
- Specify resume payload semantics:
  - include compact summary (`cursor`, queue size, last wake reason/time)
  - include raw unacknowledged events in deterministic order
  - explicit user resume is required after director/session interruption in v0

## Acceptance Criteria

- The "new events while running" behavior is explicit and does not rely on polling.
- Director restarts preserve correctness via cursored replay.
- Reprocessing a wake is safe and does not duplicate orchestration actions.
- Replay/resume payload is specified as summary + raw events (not raw-only).
- Explicit resume semantics are defined for interrupted director sessions.
- Implementation targets Rust runtime paths (control plane + desktop integration), not legacy paths.
