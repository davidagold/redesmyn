---
epic: gpui
branch:
  suggested: rn/gpui/T-25-lease-primary-enforcement
rn:
  parent: T-23
---

# T-25 Lease/primary executor management + enforcement (Domain 3)

## Problem

We must avoid “two writers” executing git mutations against the same repo identity:

- multiple daemons could exist (two laptops, stale processes, etc.),
- only one must be allowed to execute mutating repo commands at a time.

We already have a conceptual lease/primary model. In the Rust port, this must be:

- explicit in the protocol,
- enforced in the daemon,
- and observable for UI/CLI (no confusing failures).

## Goal

Implement the daemon side of lease/primary behavior:

- acquire/refresh/lose primary status per repo scope,
- enforce “mutations require primary”,
- and emit clear status/telemetry about lease ownership.

This ticket focuses on daemon behavior and protocol usage; control-plane policy/routing is implemented in Domain 2/3 follow-ups.

## Requirements

### 1) Lease model

Define a lease model that includes:

- `repo_scope`
- `primary_host_id`
- `lease_expires_at`

Daemon must track:

- whether it is primary for each attached repo scope,
- when it must renew,
- and when it must stop executing mutating operations.

### 2) Acquisition/renewal protocol

Use the daemon stream protocol (T-11) to:

- request lease acquisition/renewal, OR
- receive lease assignments from the control plane.

Pick a single approach for v0 and document it.

Preference:

- control plane is authoritative; daemon requests renewal and control plane grants/denies.

### 3) Enforcement

For repo-mutating commands (merge/restack/worktree writes/etc):

- if daemon is not primary, it must reject the command with a structured error:
  - category: conflict or unavailable
  - message: “Not primary executor for repo; primary is <host_id>”
  - include enough detail for UI to render a helpful next step.

### 4) Graceful lease loss

If the daemon loses primary while executing:

- it must stop accepting new mutating commands immediately,
- and for in-flight operations:
  - finish the current safe boundary if possible, or
  - fail the command with a clear “lost lease” error.

### 5) Observability

Emit status updates that allow UI/CLI to display:

- current primary for each repo scope,
- lease freshness/expiry horizon,
- and whether the daemon is eligible to execute.

## Acceptance criteria

- Daemon enforces primary requirements correctly for mutating operations.
- Lease renewal logic is robust and testable.
- UI/CLI can surface “who is primary” and “why my command was rejected” without guesswork.

## Dependencies / sequencing

- Depends on daemon skeleton (T-23) and daemon protocol contract (T-11).
- Coordinated with Domain 2 command routing (T-19) once implemented.

