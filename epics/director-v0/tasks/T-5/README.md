---
rn:
  node:
    branch: rn/director-v0/T-5-authn-authz-v0
  parent: T-1
---

# T-5 AuthN/AuthZ v0 for remote daemons + executors

## Plan

- Define the v0 AuthN story for daemons connecting over VPN:
  - shared token in handshake (v0)
  - upgrade path to mTLS and/or OIDC later
- Define the v0 AuthZ story:
  - which actors can enqueue commands, run gates, and perform mutating git operations
  - how repo executor leases fence writers (defense-in-depth)
- Specify concrete deployment sketches:
  - laptop control plane + EC2 daemon over VPN
  - optional SSH port-forwarding as an ergonomic supplement (not a substitute for AuthZ)

## Acceptance Criteria

- It’s clear what secrets/credentials exist in v0, where they live, and how they rotate.
- It’s clear which capabilities are authorized for which actors in v0.
- The design doesn’t assume “private network == trusted”.
