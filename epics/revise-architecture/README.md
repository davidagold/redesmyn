# Revise Architecture Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Revise Architecture** epic: intent, sequencing, and key decisions. Keep it current.

## Metadata

```yaml
slug: revise-architecture
name: Revise Architecture
root_branch: main
linear:
  project_id: null
```

## 1) Vision

Simplify Redesmyn’s mental model and make the system deployment-ready by drawing a strict boundary between:

- A **control plane (server)**: state, configuration, desired state, APIs/UI, persistence.
- A **daemon (host-local)**: worktrees, process/session lifecycle, telemetry, host-local git actions.

Terminology note: “agent runner” refers to the harness process launched by the daemon. Avoid using “runner” to mean “daemon” (too ambiguous).

The UX should remain “one command + one UI” even though the system is logically split:

- `rn up` starts (or connects) the host daemon (one per host).
- `rn run --epic <slug> --fleet-size <n>` provisions sessions for tasks in the current repo (or an explicit repo) and starts a local fleet automatically.
- The dashboard clearly indicates whether a daemon is connected and whether telemetry is fresh.

This epic focuses on the daemon/control-plane architecture revision. Harness-specific adapters remain tracked in `epics/agent-orchestration/README.md`.

## 2) Key decisions

### 2.1 “Observer” becomes a daemon capability

Git/worktree telemetry is a **daemon concern**. The current “observer” is an implementation detail (a module/capability), not a product concept.

Run one daemon per host. For a given repo, run at most one observation loop; epic scoping is not a primary UX concept, so remove the need to run telemetry with `--epic` filtering.

### 2.2 Control plane does not run host-local actions

The control plane must be runnable in a container/remote host with no repo filesystem access. Host-local actions happen in the daemon:

- git reads/writes (including graph projections like trunk timeline)
- tmux/process lifecycle
- worktree inspection/management

### 2.3 Server cannot reach the daemon; daemon connects outbound (WebSocket)

To work behind NAT/firewalls, the daemon maintains an **outbound long-lived connection** (WebSocket) to the control plane, used for:

- Daemon → server: telemetry/events, heartbeats, status
- Server → daemon: configuration updates, desired-state updates, and commands

This provides “RPC-like” behavior over a message protocol; design for idempotency, acks/dedupe, and resync on reconnect.

### 2.4 Merge Node into Task

Nodes are just tasks as represented in the graph. Consolidate these concepts so there is a single graph primitive (`Task`) with branch/topology metadata.

### 2.5 Remove git API; keep git proxy local

- Remove server endpoints/paths that proxy or execute git.
- Keep `rn git` as a local proxy (optional enforcement); the daemon’s telemetry picks up effects and reports them to the control plane.
- Any graph projections requiring git history must be produced by the daemon and sent to the control plane (events and/or snapshots).

### 2.6 Event log is authoritative in the control plane

The control plane persists an append-only event log and drives the UI’s realtime projections. In cloud mode, daemons emit events to the server (daemons should not write the server DB directly).

### 2.7 Local dev remains simple

Local-first workflows should remain ergonomic:

- `rn dev` may run control plane + daemon co-located.
- Naming/structure should preserve the conceptual split even when co-located.

### 2.8 Multi-repo identity is explicit (workspace/repo keys)

The control plane will eventually serve multiple repos across multiple organizations/workspaces.

In v1 we do not yet have a full identity/auth model, but we still need stable multi-repo identity.
Introduce the minimally sufficient iteration:

- A single implicit “default workspace” (until real org/workspace identity exists), identified by a stable `workspace_id`.
- Repos are identified by an immutable `repo_id` scoped to a `workspace_id`.

- The daemon handshake must include an explicit repo identity (not a local filesystem path).
- Repos have a unique `repo_name` within a workspace, but identifiers must be stable across renames:
  - Prefer immutable `workspace_id` + `repo_id` for primary identity.
  - Treat names/slugs as display fields and enforce uniqueness per workspace separately.

### 2.9 Integrations authenticate at the client system level (v1)

In v1, integrations like Linear/GitHub authenticate on the client (daemon host) at the system level.

- The daemon owns tokens/credentials and emits normalized integration events to the control plane.
- The control plane persists integration-derived events and projections, but does not require provider credentials.
- Server-managed OAuth/credential storage is a future step (requires multi-tenant user auth + secret storage).

### 2.10 Daemon presence is a projection (not a user-facing DB object)

Daemons are runtime processes (one per host in v1), not user-managed entities.

- Presence (“online/offline”, `last_seen`, capabilities) is derived from connection + heartbeat events (and may be projected per repo when a daemon manages multiple repos).
- The control plane may materialize a presence table for fast queries/UI, but the event log remains the source of truth.

### 2.11 Desired state lives on tasks (not agents)

Desired state should be modeled on the graph primitive (`Task`) and persisted in the control plane.

- Users express orchestration intent via task topology and task-level desired state.
- Agents report observed state/telemetry; they are not the source of truth for desired state.

## 3) Scope (v1)

- Daemon connection protocol (handshake/auth/versioning/resync).
- Daemon lifecycle management (`rn up/down/status`, logs).
- Server-side daemon presence + command delivery.
- Dashboard “daemon online/offline” surfaces and guidance.
- Data model migration: merge Node into Task.
- Git boundary: remove server git execution/proxying; consolidate git proxying locally.

## 4) Non-goals (this epic)

- Full multi-tenant auth/product onboarding flows.
- Deep harness hooks/adapters (tracked separately).
- Multi-repo orchestration or multi-user collaboration semantics.

## 5) Task map

- `epics/revise-architecture/tasks/T-1/README.md`: Daemon/control-plane naming + protocol contract.
- `epics/revise-architecture/tasks/T-2/README.md`: Control plane endpoint for daemon connection + command delivery.
- `epics/revise-architecture/tasks/T-3/README.md`: Daemon process (connect + telemetry + command execution).
- `epics/revise-architecture/tasks/T-4/README.md`: CLI UX (`rn up/down/status`) and concept consolidation.
- `epics/revise-architecture/tasks/T-5/README.md`: Dashboard daemon status + offline guidance.
- `epics/revise-architecture/tasks/T-6/README.md`: Merge `Node` into `Task` (single graph primitive).
- `epics/revise-architecture/tasks/T-7/README.md`: Remove server git execution and keep git proxying local.
