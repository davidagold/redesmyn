---
rn:
  slug: remote-execution-v0
  name: Remote execution v0
  root_branch: main
  linear:
    project_id: null
---

# Remote Execution v0 Epic: Control Doc (Canonical)

This file is the canonical "control doc" for the **Remote execution v0** epic: intent, scope, and task map.
Keep it current.

## 1) Why this epic exists

Remote execution capability is orthogonal to director decisioning. The director should operate against the same
command/event contract whether execution is local or remote.

This epic tracks the remote-specific substrate:

- secure daemon/executor connectivity and authorization,
- transport of change artifacts when direct git object transfer is unavailable,
- and operational invariants for safe remote mutation.

## 2) Relationship to director-v0

- `director-v0` owns queue/gate/planner/UI orchestration semantics.
- `remote-execution-v0` owns where/how commands run remotely and how artifacts/security are handled.

Director should not encode deployment assumptions (VPN, token modes, bundle import details) as core logic.

### Migration note

This epic absorbs scope previously tracked in `director-v0`:

- `director-v0/T-4` (remote change delivery) -> `remote-execution-v0/T-1`
- `director-v0/T-5` (remote AuthN/AuthZ) -> `remote-execution-v0/T-2`

## 3) v0 scope

- Remote change delivery via immutable artifact workflows (git bundle path).
- AuthN/AuthZ v0 for remote daemons and executors.
- Deployment and operational guidance for safe remote execution in early environments.

## 4) Non-goals (v0)

- Full production-grade zero-trust platform hardening.
- Multi-tenant policy engines and centralized identity federation.
- Reworking director queue/planner semantics.

## 5) Tasks

- `epics/remote-execution-v0/tasks/T-1/README.md`: Remote change delivery via git bundle artifacts.
- `epics/remote-execution-v0/tasks/T-2/README.md`: AuthN/AuthZ v0 for remote daemons + executors.

## 6) Sequencing intent

- `T-1` and `T-2` can proceed in parallel.
- Director integrations should consume only stable command/event contracts from this epic.
