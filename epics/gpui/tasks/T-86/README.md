---
epic: gpui
branch:
  suggested: rn/gpui/T-86-task-completion-semantics
rn:
  node:
    branch: rn/gpui/T-86-task-completion-semantics
  parent: T-21
  after:
    - T-20
    - T-30
    - T-8
---

# T-86 Task completion semantics + manual attestation (`MarkTaskDone`, `rn task done`) (Domain 2)

## Problem

We currently model task progress (`blocked`/`in_progress`/`done`) and merge readiness, but completion
semantics are still ambiguous in real workflows:

- merge can happen through multiple paths (future `rn merge`, direct git CLI, squash/rebase in hosted forges),
- "ready to merge" is not the same as "done",
- and users need an explicit way to attest completion when merge happened out-of-band.

Without a first-class completion contract, UI/CLI behavior becomes inconsistent and automation has to
guess from git topology.

## Goal

Define and implement explicit, durable task completion semantics that are shared across control plane,
UI, and CLI:

- "done" is an explicit lifecycle state, not an inferred branch property,
- merge commands can set done automatically,
- and users can manually attest done via control-plane API + `rn task done`.

## Requirements

### 1) Completion model and semantics

Define task completion semantics clearly:

- `merge_readiness` means "eligible to merge", not complete.
- `state=done` means work is accepted/integrated for workflow purposes.
- Completion is tracked with explicit metadata:
  - `completed_at`
  - `completion_source` enum (`merge_command`, `manual_attestation`, future sources allowed)
  - audit linkage (`completion_command_id` or equivalent) where available.

### 2) Command/API surfaces

Add a control-plane command/API for manual attestation:

- `MarkTaskDone { task_id, source?, note? }`
- default source is `manual_attestation`.
- idempotent behavior:
  - if already done with equivalent completion, return success with a warning/flag rather than error.
- command lifecycle follows Domain 2 command model (T-19) and is observable to clients.

### 3) CLI surface (`rn`)

Add CLI command:

- `rn task done --task <task-id|local-ref>`
- optional flags:
  - `--note <text>` (optional)
  - `--json`

Behavior:

- uses control-plane API (no direct DB writes),
- prints resulting task completion metadata,
- returns non-zero on structured errors (task not found, validation, auth/scope errors).

### 4) Merge interaction contract

Define one coherent contract between merge and done:

- successful merge flow (T-30) marks task(s) done automatically with `completion_source=merge_command`.
- future `rn merge` must not require a separate `rn task done` call.
- `rn task done` exists as fallback for out-of-band merges and acceptance workflows.

### 5) Read-model projection and graph/API visibility

Ensure completion metadata is visible in read models used by UI/CLI:

- `GetEpicGraph` task entries include `state`, `completed_at`, `completion_source`.
- projections update immediately after `MarkTaskDone` and merge-success updates.

### 6) Guardrails and warnings

Avoid over-constraining completion while still surfacing operator feedback:

- do not hard-require branch ancestry checks for completion success (supports squash/rebase/out-of-band flows),
- optionally emit warnings when git evidence appears inconsistent (for operator awareness),
- keep completion updates explicit and auditable.

### 7) Testability

Add deterministic tests for:

- `MarkTaskDone` happy path,
- idempotent re-mark behavior,
- completion metadata projection in `GetEpicGraph`,
- merge-success completion wiring (in integration with T-30),
- and CLI command behavior (`rn task done` with text + JSON output).

## Acceptance criteria

- Task completion semantics are explicit and documented in code-level contracts.
- `rn task done` exists and updates control-plane state through API commands.
- Out-of-band merge workflows can mark tasks done without manual DB edits.
- Merge-driven completion and manual completion produce consistent queryable metadata.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on command engine + API plumbing (T-19, T-20).
- Depends on EpicGraph read model (T-21) for surfacing completion fields.
- Integrates with merge execution behavior from T-30.
- Builds on Rust CLI foundations (T-8).

## Reference implementation (today; orientation only)

- Current merge/readiness/state concepts in Python:
  - `redesmyn/db/models.py` (`Task.state`, merge-readiness and merge-run related fields)
  - `redesmyn/api.py` (`/v1/epics/{epic}/graph`, merge endpoints)
- Current Rust command surfaces:
  - `rust/crates/redesmyn_control_plane` (command lifecycle + orchestration APIs)
  - `rust/crates/rn/src/main.rs` (`rn task ...` command group)
