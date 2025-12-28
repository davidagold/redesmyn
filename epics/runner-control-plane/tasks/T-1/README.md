# T-1 Runner ↔ control plane contracts + naming

## Metadata

```yaml
id: T-1
stacked_on:
node:
  branch: rn/runner-control-plane/T-1-runner-contracts
```

## Brief (local)

- Define the canonical naming and mental model:
  - control plane vs runner
  - “server” vs “daemon” vs “observer” (deprecations/aliases)
- Specify the runner ↔ control plane protocol:
  - handshake fields (runner id, repo identity, capabilities)
  - auth strategy (token/key, rotation hooks)
  - message envelope + versioning strategy
  - heartbeats/liveness + reconnect/backoff
  - resync strategy (server snapshots + runner acks)
  - command delivery model (push vs pull, idempotency + dedupe)
- Define the “local dev co-located” mode and its invariants (what changes when server == runner).

## Acceptance Criteria

- `epics/runner-control-plane/README.md` describes the v1 contract clearly enough to implement both sides.
- The glossary in `epics/redesmyn/README.md` is consistent with the naming (avoid “observer” as a user-facing concept).
- A short migration note exists: “current local DB-writing observer → runner emitting events to control plane”.
