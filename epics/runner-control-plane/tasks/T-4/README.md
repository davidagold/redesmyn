# T-4 CLI UX: `rn up/down/status` + consolidate daemon/observer concepts

## Metadata

```yaml
id: T-4
stacked_on: T-3
node:
  branch: rn/runner-control-plane/T-4-rn-up
```

## Brief (local)

- Introduce a user-friendly runner lifecycle:
  - `rn up` (start runner in the background; connect to control plane)
  - `rn down` (stop runner)
  - `rn status` (runner online/offline, last_seen, server url)
- Clarify command taxonomy:
  - “daemon” is the orchestrator service (control plane when local; server in cloud)
  - “runner” is the host-local component
  - “observer” becomes a debug-only alias or subcommand
- Provide ergonomic “copy this command” strings for the UI to surface (start/connect, logs, attach).

## Acceptance Criteria

- A new user can get to “runner connected” with a single command (`rn up`).
- Existing local workflows remain usable (`rn dev` still works; no confusing duplicate processes).
- `rn status` answers “is my runner online and feeding telemetry?” quickly.
