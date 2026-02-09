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
- Decide start/re-enqueue/request-change/merge actions for an epic (policy-controlled).
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

- The director agent directly runs `rn` commands for orchestration actions (start task, re-enqueue, request
  changes, merge).
- Event emission comes from normal command handling paths; no separate intent-runner is required for v0.
- Controller responsibilities are wakeup, delivery, dedupe/coalescing, and lifecycle coordination.

Notes:

- Director-requested changes commonly require messaging a task agent. v0 supports this by routing through the
  control plane (e.g. `rn task send --intent request_changes`) so the message and any resulting command
  outcomes are durable and observable.

### 2.4 Merge queue: explicit, human-steerable

The director operates on an explicit merge queue that supports:

- candidate refs (typically commit SHAs or task branch refs),
- dependencies/blockers (e.g. “A approved pending B”),
- conductor actions (approve, defer, request changes, requeue),
- and visibility into what’s blocked and why.

### 2.5 Merge authority policy (default safe, optional YOLO)

- Merge authority is configurable by policy.
- Global default controls baseline behavior for new epics; epic-level override is supported.
- v0 default is conservative:
  - `yolo_merge = false` (director does not autonomously merge),
  - director may still prepare merge queue and surface merge-ready suggestions.
- Optional per-epic YOLO mode:
  - `yolo_merge = true` allows autonomous merge under configured constraints.

### 2.6 Wake trigger set for v0 (no gates)

Controller wakeups must be emitted for:

- task session turn completion events,
- terminal command outcomes that affect task or merge-queue state,
- conductor override actions,
- queue-affecting outcomes from non-director actors.

Gates are intentionally excluded from v0 trigger requirements.

### 2.7 Replay/resume payload and lifecycle

- Replay/resume wake payload includes:
  - a compact summary (`cursor`, queue size, last wake reason/time),
  - and raw unacknowledged events in deterministic order.
- Director reconnect/resume is explicit in v0 (no automatic silent resume after process/application restart).

### 2.8 Review path: director-owned in v0, pluggable provider later

- In v0, the director agent may perform review itself.
- Future direction: delegate review to a separate review mechanism (LLM session/tool/CLI flow) and have the
  director consume structured review outputs.
- Event schema should preserve this evolution path (e.g. `review_requested` / `review_completed`-style events).

### 2.9 Director mode UX contract (v0)

- User-facing label is **Director mode** (v0 baseline; `auto-direct` can be explored later).
- Activation affordance is a floating, non-scrolling control in the top-right of the session timeline viewport.
- Start flow:
  - user picks `Run in current session` or `Run in new session`,
  - once intent is selected, activation is one click.
- Composer behavior in director mode:
  - normal send path becomes inline two-step `Pause & Send` (no modal/dialog),
  - `Steer` mode allows one-shot manual instruction without pausing orchestration,
  - `Steer` is a composer toggle with shortcut `Cmd+.`.
- UI visual treatment differentiates director `active` vs `idle` without badge clutter.

### 2.10 Direction overlay scope (v0)

- Overlay is read-only in v0 and does not auto-collapse.
- Overlay includes two sections:
  - event wake queue (ready/unacked backlog + expandable recent history),
  - merge queue preview (current queue ordering/state summary).
- Overlay must avoid duplicating task-card status surfaces.

## 3) v0 scope

- Director/controller run semantics (wakeups, backlog delivery, cursor/ack, idempotency).
- Merge queue + conductor controls.
- Merge authority policy (global default + epic override, YOLO disabled by default).
- Director-driven orchestration via direct `rn` command execution.
- Director mode UI as pinned session + direction overlay.

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

The following is intentionally deferred beyond director-v0:

- Gate policy, gate execution, and gate caching UX/mechanics.

## 4) Non-goals (v0)

- Fully autonomous merging by default across all epics.
- Multi-user shared control plane with fine-grained org policies.
- Perfect scheduling; v0 can be “wake on event + manual run”.
- Solving remote execution transport/security in this epic.

## 5) Tasks

- `epics/director-v0/tasks/T-1/README.md`: Director run semantics (cursor/high-water/idempotency).
- `epics/director-v0/tasks/T-2/README.md`: Merge queue model + conductor actions.
- `epics/director-v0/tasks/T-3/README.md`: Gate policy + caching (deferred; not in v0 delivery set).
- `epics/director-v0/tasks/T-4/README.md`: Controller <-> director wake protocol + event backlog delivery.
- `epics/director-v0/tasks/T-5/README.md`: Director mode session UX (activation + pause/send + steer).
- `epics/director-v0/tasks/T-6/README.md`: Director mode lifecycle + merge authority policy surfaces.
- `epics/director-v0/tasks/T-7/README.md`: Direction overlay + queue projections.
- `epics/director-v0/tasks/T-8/README.md`: Task agent messaging + history (rn surfaces).

## 6) Sequencing intent (parallelizable)

- Land `T-1` first to pin run semantics (cursor/high-water/idempotency).
- After `T-1`, execute in parallel:
  - `T-2` merge queue model + conductor actions.
  - `T-4` wake protocol + backlog delivery.
  - `T-6` director mode lifecycle + policy surfaces.
- Land `T-5` after `T-4` and `T-6` so session UX uses final lifecycle/policy contract.
- Land `T-7` after `T-2` and `T-4` so overlay projections mirror queue/wake state without duplicating task-card
  information.
- `T-3` is explicitly deferred from v0.
