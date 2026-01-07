# T-7 `rn up/down`: single-command startup/shutdown

## Metadata

```yaml
id: T-7
stacked_on: T-6
node:
  branch: rn/v0-launch/T-7-rn-up-down
```

## Brief (local)

Create the v0 user-facing happy path:

- `rn up`:
  - ensures repo is initialized (or guides to `rn init`)
  - ensures daemon is running and attached
  - starts the control plane server
  - opens the dashboard URL (optional, best-effort)
- `rn down`: stops the control plane server (and optionally the daemon; clarify semantics in CLI help).

This should replace `just run --local` and any “spawn two processes” dev shim behavior.

## Acceptance Criteria

- From a clean install, the user can reach the dashboard with `rn up` (plus a minimal, explicit init step if required).
- There is exactly one “official” way to start the system locally (no competing `just run` behaviors).

