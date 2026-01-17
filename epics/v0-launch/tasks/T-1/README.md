---
rn:
  parent: null
---

# T-1 V0 CLI contract: remove `rn observer` + remove `rn dev`

## Plan

This epic is standardizing on a clean server/daemon split. For v0 to feel coherent, the CLI must:

- present a small, unsurprising command menu, and
- avoid exposing “implementation detail” commands that imply alternate architectures.

### Work

1) Remove `rn observer`

- Delete the `rn observer` command group from the CLI help/menu.
- If we still need foreground observation for debugging, move it behind a clearly non-product namespace
  (e.g. `rn debug observe …`) or a script entrypoint. It should not be required for normal operation.
- Ensure all docs / errors route users to daemon lifecycle commands instead.

2) Remove `rn dev`

- Delete `rn dev` from the CLI help/menu.
- Provide a clear error message if someone attempts to use it (either via a compatibility shim or release note),
  pointing them to `just dev`.

3) Establish the v0 vocabulary in help output

- `rn server …`: control plane lifecycle.
- `rn daemon …`: executor lifecycle (repo attach, agent lifecycle, telemetry).
- `rn up/down`: user-facing “start/stop local system”.

### Design notes

- This task intentionally does not change runtime behavior (that’s later tasks); it narrows the surface area first so
  subsequent changes land into a stable UX.

## Acceptance Criteria

- `rn --help` no longer lists `observer` or `dev`.
- Any prior entrypoint that referenced `rn dev` or `rn observer` is redirected to `just dev` or `rn daemon …` with actionable errors.
- The CLI help text makes the split explicit:
  - server = persistence/UI/API
  - daemon = repo executor (git/worktrees/agents/telemetry)
