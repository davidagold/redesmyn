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

- Repo-scoped messages (and attach/detach) must include an explicit repo identity (not a local filesystem path).
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

### 2.13 Daemon ↔ control plane protocol (v1 contract)

This section defines the **v1** daemon/control-plane contract. It is intended to be concrete enough to implement both sides without inventing new semantics.

#### 2.13.1 Naming + deprecations

- **Control plane** and **server** are synonyms. Prefer **control plane** in user-facing docs.
- **Daemon** is the host-local orchestrator (one per host) that connects outbound to the control plane.
- **Repo executor** is the component allowed to execute git/worktree mutations for a specific repo instance.
  - v1: the host-local daemon is the repo executor for its local checkout/worktrees.
  - cloud: a server-side worker/executor is the repo executor for a server-managed checkout/bare repo.
- **Host identity (v1)**: treat `Host` (`hosts.host_key`) as the stable daemon/executor identity; no persisted “Daemon” model required.
- **Observer** is deprecated as a product concept. “Repo observation loop” refers only to an internal daemon capability (telemetry/observation).
- **Agent runner** refers to the harness process launched by the daemon (Codex/Claude/etc). Avoid using “runner” to mean “daemon”.

#### 2.13.2 Transport + connection scoping

- Transport: **WebSocket** over TLS in production (`wss://…`); `ws://localhost` in local dev.
- v1 uses **one WebSocket connection per host daemon** (`host_key`), multiplexing messages by `workspace_id + repo_id`.
- The control plane never opens inbound connections to daemons; daemons reconnect outbound.

#### 2.13.3 Authentication (v1)

- The daemon authenticates at connection time using a bearer token (for example: `Authorization: Bearer <daemon_token>`).
- Tokens are issued/rotated by the control plane. v1 assumes a single implicit default workspace but still requires `workspace_id` in repo-scoped messages.
- Local dev may run with a “dev token” provisioned automatically, but the protocol flow remains identical.

#### 2.13.4 Versioning strategy

- The protocol has an explicit `protocol` field with **major/minor** (example: `"1.0"`).
- Breaking changes increment **major**. Additive backwards-compatible changes increment **minor**.
- Both sides must reject unknown **major** versions and may warn (but accept) unknown **minor** versions.

#### 2.13.5 Message envelope

All messages are JSON objects with a shared envelope.

Repo-scoped messages must include a `scope` for routing:

```json
{
  "protocol": "1.0",
  "type": "telemetry.event_batch",
  "msg_id": "01J9ZP2Y9J3Z7Y7Z2Y3XJ2Y1Y0",
  "sent_at": "2025-01-01T00:00:00Z",
  "scope": {
    "workspace_id": "w-…",
    "repo_id": "r-…"
  },
  "payload": {}
}
```

Envelope invariants:

- `msg_id` is a unique, stable idempotency key for the message (ULID/UUID). Receivers must dedupe on `msg_id`.
- `sent_at` is informational; ordering is not guaranteed. Ordering-sensitive flows use explicit sequence/ack fields per message type.
- The schema of `payload` is determined by `type`.

#### 2.13.6 Handshake + host identity

After the WebSocket is established, the daemon sends a `daemon.hello` message:

```json
{
  "protocol": "1.0",
  "type": "daemon.hello",
  "msg_id": "…",
  "sent_at": "…",
  "payload": {
    "host_key": "h-<stable-host-identity>",
    "host_instance_id": "hi-<ephemeral-process-identity>",
    "capabilities": ["repo_execution", "git_observation", "worktrees", "processes", "integrations.linear"],
    "client": {
      "hostname": "my-host",
      "platform": "darwin",
      "version": "redesmyn/0.x"
    }
  }
}
```

Notes:

- `host_key` is the stable daemon/executor identity (stored on the host; persisted via the `Host` row).
- Enforce **at most one active connection per `host_key`** (reject/replace duplicates).
- `host_instance_id` changes on daemon restart and helps distinguish “same host, new process”.
- Repo identity is carried in `scope` on repo-scoped messages (and attach/detach), not as a host filesystem path.

The control plane responds with `daemon.hello_ack`:

- Confirms accepted protocol version and returns `connection_id`.
- May include a “server snapshot” used for resync (see below).

#### 2.13.7 Executor lease (single-writer)

When multiple daemons could attach to the same `(workspace_id, repo_id)`, repo-mutating intents need an unambiguous execution target.

- The control plane maintains a time-bounded **lease** designating the “primary” repo executor.
- The lease is scoped to `(workspace_id, repo_id)` and identifies the lease-holder by `host_key`.
- Only the lease-holder is eligible to receive repo-mutating commands by default; other attached daemons may continue read-only telemetry.

Lease messages (v1 shapes; exact naming is flexible):

- `server.lease_grant` / `server.lease_revoke` (server → daemon)
- `daemon.lease_renew` (daemon → server; optional if server uses heartbeat-based renewal)

#### 2.13.8 Heartbeats + liveness

- Either side may send `heartbeat.ping`; the peer responds with `heartbeat.pong` echoing the ping `msg_id`.
- The control plane considers the daemon offline when:
  - the socket closes, or
  - heartbeats have not been observed within the liveness window.
- The daemon uses exponential backoff with jitter on reconnect attempts.

#### 2.13.9 Resync + acknowledgements

Reconnects are expected. v1 uses:

- **Server snapshot**: after `daemon.hello`, the control plane may send `server.snapshot` containing the minimum per-repo desired-state/config required for the daemon to operate.
  - Include lease state (`lease_holder_host_key`, `lease_expires_at`) so the daemon knows whether it is eligible to execute repo-mutating commands.
- **Daemon acknowledgements**:
  - For each `server.snapshot` and each delivered `server.command`, the daemon must respond with an ack message (`daemon.ack`) that includes:
    - `ack_type` (snapshot|command|event_batch)
    - `ack_id` (the acknowledged message’s `msg_id` or stable command/run id)
    - `status` (ok|error) and optional error details

The control plane must treat acks as idempotent and safe to replay.

#### 2.13.10 Telemetry + events

- Telemetry/events are pushed daemon → control plane as `telemetry.event_batch`.
- Each batch contains an ordered list of events with stable ids so the control plane can dedupe on reconnect.
- The control plane persists events to the authoritative event log, then acks ingestion.

Example payload shape (illustrative; event vocab is an additive surface):

```json
{
  "events": [
    {
      "event_id": "01J9ZP…",
      "event_type": "git.refs_changed",
      "created_at": "2025-01-01T00:00:00Z",
      "data": { "refs": [] }
    }
  ]
}
```

Invariants:

- `event_id` is unique within a repo event stream; receivers must dedupe by `(workspace_id, repo_id, event_id)`.
- The daemon may resend unacked batches on reconnect (at-least-once delivery).
- Event vocab should key domain concepts by `task_id` (not “node”), since **Node → Task** is an explicit migration in T-6.

#### 2.13.11 Command delivery model (intent, progress, resume)

- Commands are pushed control plane → daemon as `server.command` and represent **high-level intent** (not raw git RPC).
  - Example intents: merge+restack, restack, checkout/attach, refresh projections.
- Each command includes a stable `command_id` and a `dedupe_key` (may be the same value).
- Targeting:
  - Either explicitly target a `host_key`, or
  - route implicitly to the current lease-holder for `(workspace_id, repo_id)`.
- The daemon must be able to safely handle duplicate delivery (at-least-once transport):
  - If a command with the same `command_id` (or `dedupe_key`) has already been applied, the daemon must no-op and re-ack as `ok`.

Merge/restack commands must be designed with resumable progress and manual conflict resolution in mind:

- The executor should emit progress events (e.g. `merge.run_started`, `merge.step_completed`, `merge.conflict`, `merge.run_completed`) with a stable `run_id`.
- If a conflict requires manual resolution, the executor pauses and emits a “blocked” event; the user resolves locally, then the control plane issues a `server.command` to **resume** the run by `run_id`.

Command execution results are reported via `daemon.command_result` (and/or `daemon.ack` with details).

#### 2.13.12 Git-derived projections (executor-sourced)

Any projection that requires git history or worktree inspection must be produced by the **repo executor** and sent to the control plane (events and/or snapshots).

Examples:

- trunk timeline
- “stack in sync / out of sync” indicators
- ahead/behind counts, merge-base, conflict status

The control plane persists/broadcasts these projections but does not compute them via git directly.
See **Updates** for the current temporary mismatch (`stackInSync` computed in the control plane today).

#### 2.13.13 Local dev: co-located mode invariants

When the control plane and daemon run on the same machine (for example via `rn dev`):

- They still communicate over the same WebSocket protocol (usually `ws://127.0.0.1`).
- Repo identity remains `(workspace_id, repo_id)`; local paths are not substituted for identity.
- Authentication may be simplified (dev token), but message shapes and resync/ack/lease behavior must remain identical so co-located dev exercises real protocol behavior.

#### 2.13.14 Migration note (v0 → v1)

Migration note: **current local DB-writing observer → daemon emitting events + executor projections to the control plane**.

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
- `epics/revise-architecture/tasks/T-11/README.md`: Remove `rn git` / `git_proxy`; rely on daemon-side observation for projections.
- `epics/revise-architecture/tasks/T-12/README.md`: Consolidate overlapping backend mechanisms (reduce surface area).
- `epics/revise-architecture/tasks/T-8/README.md`: Split agent “identity/config” from “session/run” (`AgentConfig` + `AgentSession`).
- `epics/revise-architecture/tasks/T-9/README.md`: Repo instances + canonical executor routing (migration).

## Updates

### 2025-12-31

- Added (temporary) “stack in sync with upstream” UI surfacing implemented via a `stackInSync` field on the graph node response.
  Today this is computed in the control plane via direct git calls, which conflicts with §§2.2/2.5.
  This must be migrated to a repo-executor-sourced projection in **T-7 + T-3 + T-2**.
- The current `stackInSync` field is on `NodeResponse`; it must move during **T-6 (Node → Task)** so the public API/UI does not retain a separate node concept.

### 2026-01-03

- Clarified that the system needs to support multiple repo executors for the same logical repo:
  - local-first (co-located server + daemon, single checkout),
  - cloud UI with a server-managed repo instance (canonical executor), and/or
  - remote compute resources (daemon host is neither the UI nor the control plane host).
- Introduced an explicit repo instance identity: `(workspace_id, repo_id, host_key)`.
  - The logical repo (`workspace_id + repo_id`) remains the unit of desired state, events, and UI navigation.
  - Repo instances represent concrete checkouts/executors and are the unit of git/worktree execution.
- Split executor commands into two targeting modes:
  - Instance-scoped commands explicitly target a `host_key`.
  - Repo-scoped “canonical” commands route to a single writer (a “primary” executor) to prevent concurrent canonical mutations.
- Added **T-9** to migrate existing T-2/T-5/T-7 implementations to this refined model and to ensure T-3 (daemon execution) implements the same semantics.
