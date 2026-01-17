---
id: T-7
stacked_on: T-1
node:
  branch: rn/revise-architecture/T-7-remove-git-api
linear:
  issue_id: f4f4dd84-233e-4b7b-a9db-67ea1d7bce4f
  identifier: RED-36
---

# T-7 Remove server git execution; consolidate git proxying locally

## Brief (local)

~~Eliminate “server == host” assumptions by moving all git execution to the daemon:~~
Eliminate “server == host” assumptions by moving all git/worktree execution out of the **control plane** and into a **repo executor**
(host-local daemon in v1; server-side worker/executor in cloud mode):

- Remove server-side codepaths that execute git directly or require repo filesystem access.
- ~~Define the daemon → control plane contract for git-derived projections needed by the UI:~~
  Define the **repo executor → control plane** contract for git-derived projections needed by the UI:
  - trunk timeline / commit strings
  - branch merge-bases / topology derivations
  - “stack in sync with upstream” / “left-behind descendant” indicators (out-of-sync)
  - commit and ref movement telemetry
- Keep `rn git` as a local proxy (optional enforcement); remove any server “git proxy” surface so git mutations remain host-local.
  - note: “host-local” here means “local to the repo executor”; in cloud mode the executor may be server-side for a server-managed repo
- Define the command + event flow for server-driven git mutations (merges/restacks):
  - control plane sends a high-level merge intent to the repo executor (not raw git RPC)
  - repo executor executes the plan locally (ff merges, rebases) and emits progress/results
  - conflicts become an explicit “resumable” state; user resolves locally and then resumes via control plane intent
  - when multiple daemons could attach to the same `workspace_id + repo_id`, git-mutating intents must be routed to a single writer (primary executor lease)

## Acceptance Criteria

- The control plane can run in a container/remote host with no repo filesystem access.
- The dashboard can render required git-derived UI from daemon-provided events/snapshots.
- Git enforcement (when enabled) is implemented purely in the client/daemon side (`rn git` and/or hooks), not via server git APIs.

## Updates

### 2025-12-31

- Recent work added `stackInSync` UI surfacing but currently computes it in the control plane via direct git.
  This task must move that computation to the repo executor and define the projection payload contract.
- Recent work also added a merge/restack UX with resumable merge runs.
  Under the revised architecture, the control plane must not execute the git steps for merges; it must route the merge intent to the repo executor.
- **Coordination note:** The updates in **T-2 + T-3 + T-7** must be considered in concert to enable server-driven merges while keeping git execution off the control plane.
