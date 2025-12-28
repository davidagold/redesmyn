# Runner Control Plane Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Runner Control Plane** epic: intent, sequencing, and key decisions. Keep it current.

## Metadata

```yaml
slug: runner-control-plane
name: Runner Control Plane
root_branch: main
linear:
  project_id: null
```

## 1) Vision

Productionize Redesmyn’s “separable control plane + host runner” architecture so the system can be commercialized as:

- A **cloud control plane** (AWS-hosted): state, orchestration, APIs/UI, persistence.
- A **local runner daemon** (user machine / repo host): worktrees, process/session lifecycle, telemetry.

The UX should remain “one command + one UI” even though the system is logically split:

- `rn up` starts (or connects) a runner for the current repo.
- The dashboard clearly indicates whether a runner is connected and whether telemetry is fresh.

This epic intentionally focuses on runner connectivity + lifecycle. Harness-specific adapters remain tracked in `epics/agent-orchestration/README.md`.

## 2) Key decisions

### 2.1 “Observer” becomes a runner capability

Git/worktree telemetry is a **runner concern**. The current “observer” is an implementation detail (a module/capability), not a product concept.

### 2.2 Server cannot reach the runner; runner connects outbound

To work behind NAT/firewalls, the runner maintains an **outbound long-lived connection** (WebSocket) to the control plane, used for:

- Runner → server: telemetry/events, heartbeats, status
- Server → runner: desired-state updates and commands

### 2.3 Event log is authoritative in the control plane

The control plane persists an append-only event log and drives the UI’s realtime projections. In cloud mode, runners emit events to the server (runners should not write the server DB directly).

### 2.4 Local dev remains simple

Local-first workflows should remain ergonomic:

- `rn dev` may run server + runner co-located.
- Naming/structure should preserve the conceptual split even when co-located.

## 3) Scope (v1)

- Runner connection protocol (handshake/auth/versioning/resync).
- Runner lifecycle management (`rn up/down/status`, logs).
- Server-side runner presence + command delivery.
- Dashboard “runner online/offline” surfaces and guidance.

## 4) Non-goals (this epic)

- Full multi-tenant auth/product onboarding flows.
- Deep harness hooks/adapters (tracked separately).
- Multi-repo orchestration or multi-user collaboration semantics.

## 5) Task map

See `epics/runner-control-plane/tasks/`.
