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
- Specify the daemon ↔ control plane protocol:
  - handshake fields (daemon id, org/repo identity, capabilities)
  - auth strategy (token/key, rotation hooks)
  - message envelope + versioning strategy
  - heartbeats/liveness + reconnect/backoff
  - resync strategy (server snapshots + daemon acks)
  - command delivery model (push vs pull, idempotency + dedupe)
- Define the “local dev co-located” mode and its invariants (what changes when server == daemon).

## Acceptance Criteria

- `epics/revise-architecture/README.md` describes the v1 contract clearly enough to implement both sides.
- The glossary in `epics/redesmyn/README.md` is consistent with the naming (avoid “observer” as a user-facing concept).
- A short migration note exists: “current local DB-writing observer → daemon emitting events to control plane”.
