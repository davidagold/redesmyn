# T-7 Remove server git execution; consolidate git proxying locally

## Metadata

```yaml
id: T-7
stacked_on: T-1
node:
  branch: rn/revise-architecture/T-7-remove-git-api
```

## Brief (local)

Eliminate “server == host” assumptions by moving all git execution to the daemon:

- Remove server-side codepaths that execute git directly or require repo filesystem access.
- Define the daemon → control plane contract for git-derived projections needed by the UI:
  - trunk timeline / commit strings
  - branch merge-bases / topology derivations
  - commit and ref movement telemetry
- Keep `rn git` as a local proxy (optional enforcement); remove any server “git proxy” surface so git mutations remain host-local.

## Acceptance Criteria

- The control plane can run in a container/remote host with no repo filesystem access.
- The dashboard can render required git-derived UI from daemon-provided events/snapshots.
- Git enforcement (when enabled) is implemented purely in the client/daemon side (`rn git` and/or hooks), not via server git APIs.
