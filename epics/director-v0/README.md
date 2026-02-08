---
rn:
  slug: director-v0
  name: Director v0
  root_branch: main
  linear:
    project_id: null
---

# Director v0 Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Director v0** epic: intent, v0 spec, sequencing, and key
decisions. Keep it current.

## Terminology

- **Director agent**: the LLM session pinned to an epic that decides orchestration actions and runs `rn` directly.
- **Controller**: mechanical wake/queue layer that observes events and sends messages to the director agent.
- **Conductor**: human operator that can pause/override/approve policy-sensitive actions.
- **Remote execution**: where/with-what credentials commands run. This is intentionally out of scope for this epic.

## 1) Vision

Add a **director** that can drive multi-agent work towards a coherent outcome:

- Observe the control plane’s durable event stream.
- Decide merge/start/request-change/gating actions for an epic.
- Execute those actions via `rn` commands from the director agent session.
- Keep the human “conductor” in the loop for approvals/overrides.

The control plane remains the source of truth for state and execution. The director agent is the required
decision-maker for orchestration, while the controller provides deterministic wakeup and backlog delivery.

## 2) Key decisions (v0)

### 2.1 Push-based controller -> director wake protocol (no polling in v0)

- The director should be woken by the controller when significant new information arrives.
- v0 does not require the director to poll event queues directly.
- Wakeups can be coalesced, but wake payload correctness must be deterministic (cursor-based).

### 2.2 Wake payload policy: include all available unacknowledged events

- For expediency and backlog control, each wake includes all unacknowledged events available at wake time.
- If the payload is too large for one message, the controller should chunk deterministically without dropping
  events.
- Director acknowledgements advance an explicit cursor so the controller knows what has been consumed.

### 2.3 Director executes orchestration via `rn`

- The director agent directly runs `rn` commands for orchestration actions (start task, merge, run gates, etc.).
- Event emission comes from normal command handling paths; no separate intent-runner is required for v0.
- Controller responsibilities are wakeup, delivery, dedupe/coalescing, and lifecycle coordination.

### 2.4 Merge queue: explicit, human-steerable

The director operates on an explicit merge queue that supports:

- candidate refs (typically commit SHAs or task branch refs),
- dependencies/blockers (e.g. “A approved pending B”),
- conductor actions (approve, defer, request changes, requeue),
- and visibility into what’s gated/blocked and why.

### 2.5 Gating is a first-class command, applied sparingly

Gates (tests, lint, typecheck, build, etc.) are modeled as commands that:

- run on a designated executor,
- emit durable progress + results into the event log,
- publish logs/artifacts for review,
- and can be cached by `(candidate_ref, base_ref_used, policy)`.

### 2.6 Review path: director-owned in v0, pluggable provider later

- In v0, the director agent may perform review itself.
- Future direction: delegate review to a separate review mechanism (LLM session/tool/CLI flow) and have the
  director consume structured review outputs.
- Event schema should preserve this evolution path (e.g. `review_requested` / `review_completed`-style events).

### 2.7 UI is part of the director contract, kept minimal in v0

- The director surface is the pinned epic session view.
- Composer is disabled while automatic direction is active, unless the user explicitly pauses automatic direction.
- The UI includes a subtle integrated visual treatment indicating an active director session (avoid badge-only UI).
- A small controller overlay in graph view shows wake/queue status only (not duplicate task progress surfaces).

## 3) v0 scope

- Director/controller run semantics (wakeups, backlog delivery, cursor/ack, idempotency).
- Merge queue + conductor controls.
- Gate policy and gate execution plumbing (as commands).
- Director-driven orchestration via direct `rn` command execution.
- Director UI v0 as pinned session + controller overlay.

## 3.1 Implementation boundary (Rust/GPUI only)

- Director v0 implementation lands in the Rust application stack:
  - `crates/redesmyn_desktop` (GPUI desktop shell + UI orchestration surfaces),
  - `crates/redesmyn_control_plane` (command/event orchestration semantics),
  - `crates/redesmyn_ui_graph` + `crates/redesmyn_ui_session` (director/task UI surfaces),
  - related Rust protocol/storage crates as needed.
- Legacy Python backend and legacy webview application are out of implementation scope for this epic.
- Any temporary compatibility glue must not become the source of truth for director semantics.

## 3.2 Explicit out-of-scope (moved to sibling epic)

The following is tracked in `epics/remote-execution-v0/README.md`:

- Remote change transport (e.g. git bundle artifact workflows).
- Remote daemon/executor AuthN/AuthZ and deployment security posture.

## 4) Non-goals (v0)

- Fully autonomous merging without human approval.
- Multi-user shared control plane with fine-grained org policies.
- Perfect scheduling; v0 can be “wake on event + manual run”.
- Solving remote execution transport/security in this epic.

## 5) Tasks

- `epics/director-v0/tasks/T-1/README.md`: Director run semantics (cursor/high-water/idempotency).
- `epics/director-v0/tasks/T-2/README.md`: Merge queue model + conductor actions.
- `epics/director-v0/tasks/T-3/README.md`: Gate policy + caching as commands.
- `epics/director-v0/tasks/T-4/README.md`: Controller <-> director wake protocol + event backlog delivery.
- `epics/director-v0/tasks/T-5/README.md`: Director UI v0 (pinned session + controller overlay).

## 6) Sequencing intent (parallelizable)

- Land `T-1` first to pin run semantics (cursor/high-water/idempotency).
- After `T-1`, execute in parallel:
  - `T-2` merge queue model + conductor actions.
  - `T-3` gate policy + caching as commands.
  - `T-4` wake protocol + backlog delivery.
- Land `T-5` after `T-4`, with alignment against `T-2`/`T-3` so the UI mirrors queue/gate/controller state
  without duplicating task-card information.
