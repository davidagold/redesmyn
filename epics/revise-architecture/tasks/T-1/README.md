# T-1 Daemon ↔ control plane contracts + naming

## Metadata

```yaml
id: T-1
stacked_on:
node:
  branch: rn/revise-architecture/T-1-daemon-contracts
```

## Brief (local)

- Define the canonical naming and mental model:
  - control plane vs daemon
  - “server” vs “daemon” vs “observer” (deprecations/aliases)
  - “agent runner” (harness process) vs “daemon” (host-local orchestrator)
  - **repo executor**: a role/interface for “the thing that can mutate git/worktrees for a repo instance”
    - v1: the host-local daemon satisfies this role
    - cloud: the role may be satisfied by a server-side worker/executor for a server-managed repo
- Specify the daemon ↔ control plane protocol:
  - handshake fields (daemon/executor id, workspace/repo identity, capabilities)
    - v1: include `workspace_id` even if there is only a single default workspace
    - clarify how daemon identity maps onto DB state:
      - in v1, daemon/executor identity is `hosts.host_key` (no separate persisted Daemon model required)
      - enforce at most one active connection per `host_key` (reject/replace duplicates)
  - auth strategy (token/key, rotation hooks)
  - message envelope + versioning strategy
  - heartbeats/liveness + reconnect/backoff
  - resync strategy (server snapshots + daemon acks)
  - command delivery model (push vs pull, idempotency + dedupe)
    - include “high-level intent” messages for git/worktree mutations (e.g. merge/restack plans) rather than raw git RPC
    - include a targeting story for git-mutating intents when multiple daemons may be attached to the same `workspace_id + repo_id`:
      - explicit executor target (`host_key`) or implicit via a lease/primary executor
- Define the “local dev co-located” mode and its invariants (what changes when server == daemon).

## Acceptance Criteria

- `epics/revise-architecture/README.md` describes the v1 contract clearly enough to implement both sides.
- The glossary in `epics/redesmyn/README.md` is consistent with the naming (avoid “observer” as a user-facing concept).
- A short migration note exists: “current local DB-writing observer → daemon emitting events to control plane”.

## Updates

### 2025-12-31

- Updated the epic control doc to formalize **repo executors** + **leases** (see `epics/revise-architecture/README.md` §2.12).
  This task’s protocol contract should explicitly cover executor identity (`host_key`) and lease/primary semantics so that
  server-driven git features (merges, stack restacks, git-derived projections) have an unambiguous execution target.
