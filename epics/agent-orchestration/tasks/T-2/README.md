# T-2 DB + API: hosts, sessions, and orchestration mutations

## Metadata

```yaml
id: T-2
stacked_on: T-1
node:
  branch: rn/agent-orchestration/T-2-hosts-sessions-api
```

## Brief (local)

- Add persistence for runner hosts and agent sessions (migrations + ORM + schemas).
- Add a `harness_profiles` registry table (string PK) and reference it from sessions:
  - Archetypal profile lives in `harness_profiles` (built-in + user-defined).
  - Session records the selected `harness_profile_id` plus a resolved profile snapshot (for audit/debug).
- Add mutation endpoints so the dashboard can act as a control surface:
  - register/list hosts
  - list profiles (and optionally CRUD user profiles)
  - create/list sessions; start/stop/restart session
  - assign/unassign agent ⇄ node
  - update agent/session liveness (`last_seen_at`, status)
- Ensure `/v1/epics/{epic}/graph` returns enough data for the graph to render:
  - agent status for nodes
  - session status for nodes (if session concept is separate from agent)

## Acceptance Criteria

- OpenAPI includes session/host endpoints and types.
- The dashboard can mutate core state without shelling out to `rn` for common workflows (assignment, session start/stop).
- Host-local paths are treated as opaque strings at the control plane boundary (no “server file exists” assumptions in API behavior).
- DB enforces core runtime invariants via constraints:
  - At most one *active* session per node (v0)
  - At most one *active* session per agent (v0)
  (implemented as partial unique indexes, e.g. “unique where `ended_at IS NULL`”).
