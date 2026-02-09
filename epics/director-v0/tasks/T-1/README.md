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

## Run Semantics Contract

### 1) Lifecycle and deterministic coalescing

- Director is an epic-pinned session, with one controller run-context per epic.
- At most one wake is in-flight per epic.
- New significant events while in-flight are coalesced by high-water, not queued as independent wakes.
- Coalescing is monotonic:
  - `pending_high_water_event_id = max(pending_high_water_event_id, latest_event_id)`.
  - wake reasons are unioned and serialized deterministically (sorted).

### 2) Event window semantics

- `ack_cursor_event_id` is durable and monotonic.
- Each wake snapshots immutable `high_water_event_id`.
- Events visible in that wake are exactly:
  - `ack_cursor_event_id < event.id <= high_water_event_id`,
  - filtered to the epic scope.
- Events that arrive after wake snapshot (`event.id > high_water_event_id`) are excluded from the in-flight wake and consumed by the next wake window.

### 3) Ack semantics (valid and invalid)

- Director acks with `ack_event_id`.
- Valid range is `(ack_cursor_event_id, high_water_event_id]`.
- For valid ack:
  - advance `ack_cursor_event_id = ack_event_id`.
  - if `ack_event_id < high_water_event_id`, replay remaining suffix of the same wake window.
- Duplicate/stale ack (`ack_event_id <= ack_cursor_event_id`) is invalid.
- Out-of-range ack (`ack_event_id > high_water_event_id`) is invalid.
- Invalid ack handling is deterministic:
  - do not advance cursor,
  - keep wake window unchanged,
  - emit durable rejection event/signal containing `wake_id`, expected range, and received ack,
  - continue replay contract with unchanged wake payload window.

### 4) Action idempotency for direct `rn` execution

- Idempotency keys must be stable across wakes for semantically identical actions.
- Required action-fingerprint key format:
  - `director:{epic_id}:{action_kind}:{action_target}:{action_payload_hash}`
- `wake_id` must not be part of the idempotency key.
- `wake_id` is provenance metadata only (for observability/audit), attached to emitted command/event records.
- State guards are still required in addition to keys (e.g. merged-already checks, active-session checks, gate cache checks).

### 5) Interruption and explicit resume-required lifecycle (v0)

- On director interruption (session loss, operator stop, runtime failure), transition to `resume_required` mode.
- While `resume_required`:
  - no automatic resume in v0,
  - no new in-flight wake is dispatched,
  - controller continues coalescing backlog by updating `pending_high_water_event_id`.
- Resume is explicit user/conductor action.
- On explicit resume:
  - transition `resume_required -> active`,
  - open next wake from current `ack_cursor_event_id` to current coalesced/latest high-water,
  - preserve deterministic replay/idempotency guarantees.

### 6) Minimal persisted state (per epic)

- `epic_id`
- `director_session_ref`
- `ack_cursor_event_id`
- `in_flight_wake_id` (nullable)
- `in_flight_high_water_event_id` (nullable)
- `pending_high_water_event_id` (nullable)
- `director_mode` (`active | paused | error | resume_required`)
- `resume_required_at` (nullable)
- `resume_required_reason` (nullable)
- `last_wake_reason` (nullable)
- `last_wake_size` (nullable)
- `last_wake_at` (nullable)

## Acceptance Criteria

- The "new events while running" behavior is explicit and does not rely on polling.
- Director restarts preserve correctness via cursored replay.
- Reprocessing a wake is safe and does not duplicate orchestration actions.
- Replay/resume payload is specified as summary + raw events (not raw-only).
- Explicit resume semantics are defined for interrupted director sessions.
- Implementation targets Rust runtime paths (control plane + desktop integration), not legacy paths.
- Test vector: replay across wake boundaries (same action intent appears in wake N and replay/continuation wake N+1) does not duplicate side effects because idempotency key is action-fingerprint-stable across wakes.
- Test vector: interruption during/after wake transitions director to `resume_required`; controller coalesces backlog without dispatch; explicit resume re-enters `active` and dispatches from current cursor.
- Test vector: stale/out-of-range/duplicate ack is durably rejected with expected-range metadata, cursor does not advance, and wake window remains unchanged for replay.
