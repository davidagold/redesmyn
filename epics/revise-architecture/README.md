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

- `rn daemon up` starts (or connects) the host daemon (one per host).
- `rn run --epic <slug> --fleet-size <n>` provisions sessions for tasks in the current repo (or an explicit repo) and starts a local fleet automatically.
- The dashboard clearly indicates whether a daemon is connected and whether telemetry is fresh.

Repo selection (v1):

- When `rn` is run inside a git repo, that repo is the default target.
- When run outside a git repo, commands that need a repo must take an explicit repo selector (exact flag is a CLI design choice; avoid path-based identifiers at the control plane boundary).

This epic focuses on the daemon/control-plane architecture revision. Harness-specific adapters remain tracked in `epics/agent-orchestration/README.md`.

## 2) Key decisions

### 2.1 “Observer” becomes a daemon capability

Git/worktree telemetry is a **daemon concern**. The current “observer” is an implementation detail (a module/capability), not a product concept.

Run one daemon per host. For a given repo, run at most one observation loop; epic scoping is not a primary UX concept, so remove the need to run telemetry with `--epic` filtering.

### 2.2 Control plane does not run host-local actions

The control plane must be runnable in a container/remote host with no repo filesystem access.

~~Host-local actions happen in the daemon:~~
Repo-local actions happen in a **repo executor** (in v1: the host-local daemon; in future cloud mode: a server-side repo worker/executor):

- git reads/writes (including graph projections like trunk timeline)
- tmux/process lifecycle
- worktree inspection/management

### 2.3 Server cannot reach the daemon; daemon connects outbound (WebSocket)

To work behind NAT/firewalls, the daemon maintains an **outbound long-lived connection** (WebSocket) to the control plane, used for:

- Daemon → server: telemetry/events, heartbeats, status
- Server → daemon: configuration updates, desired-state updates, and commands

This provides “RPC-like” behavior over a message protocol; design for idempotency, acks/dedupe, and resync on reconnect.

In v1, prefer a **single WebSocket connection per host daemon**, multiplexing repo-scoped messages by `workspace_id` + `repo_id`.

#### Repo attachment (“attach”) semantics (v1)

“Attach repo” means: **activate a repo on the host daemon** so it can:

- start the repo’s observation loop (git/worktree/session telemetry → events), and
- start reconciling repo-scoped desired state + commands locally.

In the common local workflow, this should be a **local** operation (`rn` talks to the daemon on the same host) and should not require a control plane round-trip.

The control plane may still request attach/detach in these cases:

- UI-driven orchestration where the daemon needs to begin managing a repo before it can act.
- Daemon reconnect/resync (“re-attach these repo ids”).

The control plane must not send host filesystem paths for repo access. Attach/detach should be expressed in terms of stable repo identity (`workspace_id` + `repo_id`); the daemon resolves repo roots from its local registry.

### 2.4 Merge Node into Task

Nodes are just tasks as represented in the graph. Consolidate these concepts so there is a single graph primitive (`Task`) with branch/topology metadata.

### 2.5 Remove git API; keep git proxy local

- Remove server endpoints/paths that proxy or execute git.
- Keep `rn git` as a local proxy (optional enforcement); the daemon’s telemetry picks up effects and reports them to the control plane.
- ~~Any graph projections requiring git history must be produced by the daemon and sent to the control plane (events and/or snapshots).~~
  Any graph projections requiring git history must be produced by the **repo executor** and sent to the control plane (events and/or snapshots).

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

This is strictly about identity and event attribution; multi-repo orchestration semantics (cross-repo desired state, multi-repo fleets) remain out of scope for this epic.

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

### 2.12 Repo execution targets: executors + leases (v1)

As we move toward cloud deployability, “control plane can’t run git” must not imply “git always runs on a user’s machine”.
Instead, we introduce an explicit **repo executor** role:

- A **repo executor** is the component that has filesystem access to a specific repo instance and is allowed to execute git/worktree mutations.
  - Local-first: the host-local daemon is the repo executor for the local checkout/worktrees.
  - Cloud: a server-side worker can be the repo executor for a server-managed checkout/bare repo.
- The control plane issues **high-level intent** (e.g. “merge task T-7 and restack descendants”), not a raw git RPC.
  The repo executor turns that intent into concrete git steps, executes them, and reports progress/results back as events.

To avoid ambiguity when multiple daemons could be attached to the “same repo” (e.g. two laptops, duplicate daemons, etc.), git-mutating intents must have a single writer:

- Define a single “primary” repo executor via a time-bounded **lease** (aka primary/ownership).
- The control plane routes git-mutating commands to the lease-holder; other connected daemons may remain attached for read-only telemetry.

For v1, keep this simple:

- Treat `Host` (via `hosts.host_key`) as the stable daemon/executor identity; a separate persisted “Daemon” model is not required.
- Route commands over the runtime WebSocket connection keyed by `host_key` (enforce at most one active connection per host_key; reject/replace duplicates).
- Execution targets for repo-mutating commands must be scoped to `workspace_id + repo_id` plus a specific executor identity (explicit `host_key` or implicit via lease).

## 3) Scope (v1)

- Daemon connection protocol (handshake/auth/versioning/resync).
- Daemon lifecycle management (`rn daemon up/down/status`, logs).
- Server-side daemon presence + command delivery.
- Dashboard “daemon online/offline” surfaces and guidance.
- Data model migration: merge Node into Task.
- Git boundary: remove server git execution/proxying; consolidate git proxying locally.
- Data model cleanup: split agent config from run history (`AgentConfig` + `AgentSession`).

## 4) Non-goals (this epic)

- Full multi-tenant auth/product onboarding flows.
- Deep harness hooks/adapters (tracked separately).
- Multi-repo orchestration or multi-user collaboration semantics.

## 5) Task map

- `epics/revise-architecture/tasks/T-1/README.md`: Daemon/control-plane naming + protocol contract.
- `epics/revise-architecture/tasks/T-2/README.md`: Control plane endpoint for daemon connection + command delivery.
- `epics/revise-architecture/tasks/T-3/README.md`: Daemon process (connect + telemetry + command execution).
- `epics/revise-architecture/tasks/T-4/README.md`: CLI UX (`rn daemon up/down/status`) and concept consolidation.
- `epics/revise-architecture/tasks/T-5/README.md`: Dashboard daemon status + offline guidance.
- `epics/revise-architecture/tasks/T-6/README.md`: Merge `Node` into `Task` (single graph primitive).
- `epics/revise-architecture/tasks/T-7/README.md`: Remove server git execution and keep git proxying local.
- `epics/revise-architecture/tasks/T-8/README.md`: Split agent “identity/config” from “session/run” (`AgentConfig` + `AgentSession`).

## Updates

### 2025-12-31

- Added (temporary) “stack in sync with upstream” UI surfacing implemented via a `stackInSync` field on the graph node response.
  Today this is computed in the control plane via direct git calls, which conflicts with §§2.2/2.5.
  This must be migrated to a repo-executor-sourced projection in **T-7 + T-3 + T-2**.
- The current `stackInSync` field is on `NodeResponse`; it must move during **T-6 (Node → Task)** so the public API/UI does not retain a separate node concept.
