# T-4 CLI UX: `rn daemon up/down/status` + consolidate server/daemon concepts

## Metadata

```yaml
id: T-4
stacked_on: T-3
node:
  branch: rn/revise-architecture/T-4-rn-up
```

## Brief (local)

- Introduce a user-friendly daemon lifecycle:
  - `rn daemon up` (start daemon in the background; connect to control plane)
  - `rn daemon down` (stop daemon)
  - `rn daemon status` (daemon online/offline, last_seen, server url)
- Make repo registration/attachment automatic:
  - any `rn` command checks “is this a git repo, and is it registered?”
  - if unregistered, register and persist the `workspace_id` + `repo_id` locally for future runs
  - if registered but not attached, attach so the daemon can start telemetry + reconciliation for this repo
  - prefer local attach (`rn` → daemon on the same host) to avoid a CLI → control plane → daemon → control plane loop
- Define how orchestration commands relate:
  - `rn run`: sets desired state for a task fleet (control-plane intent) and waits for the daemon to reconcile when requested.
  - `rn agent run`: single-task convenience wrapper around `rn run` (optionally attaches or tails logs).
  - `rn run` / `rn agent run` should fail with actionable guidance (or auto-start) if the local daemon is not running.
- Clarify command taxonomy:
  - “daemon” is the host-local orchestrator for worktrees + sessions + telemetry
  - “control plane/server” is the API/UI persistence layer
  - “observer” becomes a debug-only alias or subcommand (daemon capability)
- Provide ergonomic “copy this command” strings for the UI to surface (start/connect, logs, attach).

## Acceptance Criteria

- A new user can get to “daemon connected” with a single command (`rn daemon up`).
- Existing local workflows remain usable (`rn dev` still works; no confusing duplicate processes).
- `rn daemon status` answers “is my daemon online and feeding telemetry?” quickly.
