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

## 1) Vision

Add a **director** that can drive multi-agent work towards a coherent outcome:

- Observe the control plane’s durable event stream.
- Propose merge actions (order, dependencies, wiring work) and gating.
- Route intent into the existing command system so the control plane can delegate to appropriate executors.
- Keep the human “conductor” in the loop for approvals/overrides.

The control plane remains the source of truth for state and execution; the director is an automation consumer
of that state, not an alternate authority.

## 2) Key decisions (v0)

### 2.1 Event consumption: cursor + high-water mark (no separate ack queue)

Director runs consume events from the durable event log using:

- a **persistent cursor** (last processed event ID / timestamp), and
- a **run high-water mark** taken at run start (process “events ≤ HWM” for determinism).

New events arriving during a run are picked up on the next run; wakeups can be lossy because correctness comes
from cursor replay.

### 2.2 Merge queue: explicit, human-steerable

The director operates on an explicit merge queue that supports:

- candidate refs (typically commit SHAs or task branch refs),
- dependencies/blockers (e.g. “A approved pending B”),
- conductor actions (approve, defer, request changes, requeue),
- and visibility into what’s gated/blocked and why.

### 2.3 Gating is a first-class command, applied sparingly

Gates (tests, lint, typecheck, build, etc.) are modeled as commands that:

- run on a designated executor,
- emit durable progress + results into the event log,
- publish logs/artifacts for review,
- and can be cached by `(candidate_ref, base_ref_used, policy)`.

### 2.4 Changes are identified by commit SHA when possible; transport is an artifact concern

The source of truth for “what changed” should be **git objects** (commit SHA + reachable history). When the
control plane cannot fetch those objects directly (e.g. remote executor has no push creds), the change can be
delivered as a **git bundle artifact** via the protocol/artifact channel.

### 2.5 Network/security: start with VPN

Remote parity should assume private networking first (VPN). AuthN/AuthZ still matters even over VPN:

- daemons authenticate to the control plane (v0: shared token; later: mTLS/OIDC),
- control plane authorizes commands per actor/role,
- and repo executor leases fence mutating operations.

## 3) v0 scope

- Director run semantics (wakeups, cursoring, idempotency).
- Merge queue + conductor controls.
- Gate policy and gate execution plumbing (as commands).
- “Change delivery” via git bundle artifacts for remote executors.
- Minimal AuthN/AuthZ surfaces to support remote daemons safely.

## 4) Non-goals (v0)

- Fully autonomous merging without human approval.
- Multi-user shared control plane with fine-grained org policies.
- Perfect scheduling; v0 can be “wake on event + manual run”.

## 5) Tasks

See `epics/director-v0/tasks/` for the task breakdown.

## 6) Sequencing intent (parallelizable)

- Land `T-1` first to pin run semantics (cursor/high-water/idempotency).
- After `T-1`, execute in parallel:
  - `T-2` merge queue model + conductor actions.
  - `T-3` gate policy + caching as commands.
  - `T-5` AuthN/AuthZ v0 for remote daemons + executors.
- Then land `T-4` after `T-3`, with an explicit alignment pass against `T-2` for queue ref semantics before/after bundle import.
