---
rn:
  node:
    branch: rn/revise-architecture/T-4-rn-up
  linear:
    issue_id: f1b429df-7ad2-4c7d-9082-14bc93b32c60
    identifier: RED-33
  parent: T-3
---

# T-4 CLI UX: `rn daemon up/down/status` + consolidate server/daemon concepts

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
  - clarify (and implement) how a canonical executor is chosen when multiple daemons could attach to the same `workspace_id + repo_id`:
    - v1: the **control plane** maintains a time-bounded “primary executor” lease for the repo (T-9), scoped to `(workspace_id, repo_id)` and held by a specific `hosts.host_key`
    - lease freshness should be driven by daemon connectivity + **heartbeats that include attached repo keys**, not by ad-hoc CLI renewal loops
    - the CLI’s role is to ensure the repo is attached (so the daemon heartbeat can refresh the lease) and to **surface** current executor/lease state clearly
    - provide actionable guidance when the lease is held elsewhere (“another host is primary for this repo”) and a deliberate “take over” affordance
      (e.g. `rn daemon claim-primary` / `rn daemon take-primary --force`), rather than silently stealing the lease
- Define how orchestration commands relate:
  - `rn run`: sets desired state for a task fleet (control-plane intent) and waits for the daemon to reconcile when requested.
  - `rn agent run`: single-task convenience wrapper around `rn run` (optionally attaches or tails logs).
  - `rn run` / `rn agent run` should fail with actionable guidance (or auto-start) if the local daemon is not running.
- Clarify command taxonomy:
  - “daemon” is the host-local orchestrator for worktrees + sessions + telemetry
  - “control plane/server” is the API/UI persistence layer
  - “observer” becomes a debug-only alias or subcommand (daemon capability)
- Provide ergonomic “copy this command” strings for the UI to surface (start/connect, logs, attach).
  - include any needed “take over / claim lease” affordance commands in the guidance vocabulary

## Acceptance Criteria

- A new user can get to “daemon connected” with a single command (`rn daemon up`).
- Existing local workflows remain usable (`just dev` still works; no confusing duplicate processes).
- `rn daemon status` answers “is my daemon online and feeding telemetry?” quickly.
- When multiple daemons are attached to the same repo, `rn daemon status` (or a repo-scoped status subcommand) clearly indicates:
  - which host is **primary** for canonical repo mutations, and
  - which hosts are merely attached (read-only telemetry/observation).
- If the current host is not primary, the CLI provides clear next steps (wait for expiry, detach the other host, or explicitly claim/take over).

## Updates

### 2025-12-31

- The epic control doc now formalizes a **repo executor** role and a “primary executor” **lease** for git-mutating actions (see `epics/revise-architecture/README.md` §2.12).
  This task should ensure the CLI can establish/inspect the repo’s executor/lease state so UI-driven merges and restacks have a clear target.

### 2026-01-04

- T-9 is merged and implements primary executor selection via a lease refreshed by daemon heartbeats (repo keys in `attached_repos`).
  This task should align its CLI ergonomics with that model:
  - prefer “ensure attach” as the default path to becoming eligible for canonical mutations in local-first mode, and
  - provide explicit, deliberate commands for lease takeover rather than implicit renewal/stealing.
