---
id: T-7
stacked_on: T-6
must_land_after:
  - T-3
---

# T-7 `rn up/down`: single-command startup/shutdown

## Plan

This is the user-facing “happy path” entrypoint for v0. It should be boring, repeatable, and the only thing a new user
needs to know to get a running dashboard against a repo.

This task intentionally depends on the control plane being “pure” (T-3) so `rn up` does not accidentally start a hybrid
system.

### Work

1) Define and implement `rn up`

Minimum behavior:

- Accept a repo selector (T-2) and determine the target repo identity/DB.
- Ensure repo initialization (either:
  - auto-init if safe, or
  - emit a single explicit instruction to run `rn init`).
- Ensure daemon is running (`rn daemon up` behavior) and repo is attached.
- Start the control plane server.
- Print the dashboard URL (and optionally open it best-effort).

2) Define and implement `rn down`

- Clarify whether `rn down` stops:
  - server only (recommended; daemon is a host service), or
  - both server and daemon.
- Ensure it’s safe to call even if nothing is running.

3) Make it the canonical way to start the system

- Remove/replace any legacy “two-process” shims (`just run --local`) in favor of `rn up`.
- Ensure error messages across commands consistently point to `rn up` / `rn daemon up` as the fix.

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
