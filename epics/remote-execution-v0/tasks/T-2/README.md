---
rn:
  node:
    branch: rn/remote-execution-v0/T-2-authn-authz-v0
  parent: null
---

# T-2 AuthN/AuthZ v0 for remote daemons + executors

## Plan

- Define v0 AuthN for daemon/control-plane connectivity:
  - shared token handshake in v0,
  - explicit upgrade path to mTLS and/or OIDC.
- Define v0 AuthZ capability boundaries:
  - who can enqueue commands, execute gates, and perform mutating git operations,
  - repo executor lease/fencing model for mutating operations.
- Define deployment posture and operations:
  - VPN-first networking assumptions,
  - credential storage/rotation guidance,
  - operator-visible auditability requirements.

## Acceptance Criteria

- v0 secrets/credentials and rotation expectations are explicit and implementable.
- Authorized capabilities per actor class are explicit and enforceable.
- The design does not assume "private network == trusted".
