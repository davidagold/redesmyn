---
id: T-6
stacked_on: T-2
---

# T-6 `rn daemon up/down/status`: background daemon + attach/registry ergonomics

## Plan

The v0 experience needs a single way to get “executor connected”. The revise-architecture epic’s T-4 calls out the desired
daemon UX (`rn daemon up/down/status`) and repo attachment semantics (registry + attach, plus primary executor lease).

This task focuses on lifecycle and ergonomics; it should not require the user to understand WS protocols or internal
models.

### Work

1) Implement daemon background lifecycle (v0 best-effort)

- `rn daemon up`: start (or ensure) a daemon process is running on this host.
  - Decide the v0 “background” mechanism (e.g., tmux session, subprocess + pidfile, launchd optional).
  - Ensure “up” is idempotent.
- `rn daemon down`: stop the running daemon.
- `rn daemon status`: report:
  - connected/not connected to control plane
  - display name / host key
  - attached repos
  - primary executor status for the selected repo

2) Repo registry + attachment ergonomics

- Persist a host-local mapping from “repo identity” to “repo root path” so the daemon can attach without receiving paths
  from the control plane.
- Ensure repo attach happens automatically when reasonable (e.g., `rn up`, `rn run`, merge actions), otherwise provide a
  one-command fix.

3) Primary executor / lease UX

- When the current host is not primary, surface:
  - who is primary
  - why canonical operations are blocked
  - the explicit command to claim/take over (no silent stealing).

Implement the v0 daemon lifecycle UX described in revise-architecture T-4:

- `rn daemon up`: start (or ensure) a background daemon process for this host.
- `rn daemon down`: stop it.
- `rn daemon status`: show whether it’s online, where it’s connected, and what repos are attached.

Also implement “repo attachment” ergonomics:

- Automatic repo registration/lookup for `workspace_id + repo_id` (stored host-locally).
- `rn` commands that require execution should ensure attach (or emit actionable guidance).
- Provide explicit “claim/take primary” affordances if another host holds the lease (v0 can be a minimal deliberate command).

## Acceptance Criteria

- A new user can get to “daemon connected” with one command (`rn daemon up`).
- `rn daemon status` answers “am I connected + attached + primary?” quickly.
- Duplicate/double-start is safe (idempotent “up”).
