---
id: T-3
stacked_on: T-5
---

# T-3 Remove local mode: make the control plane pure

## Plan

The current “local mode” makes the control plane behave like an executor:

- the server can start agents directly, and
- the server runs repo/agent reconciliation loops in-process.

This is the main source of confusion and architectural drift. After this task lands, the control plane must be runnable
without repo filesystem access, and all repo-local behavior must flow through a daemon connection.

This task must land only once daemon-owned execution is end-to-end functional (agent lifecycle + monitoring + telemetry),
otherwise the product becomes unusable.

### Work

1) Remove `runner_mode=local` as a supported mode

- Decide whether the setting is deleted vs deprecated-but-ignored (v0 likely deletes).
- Ensure config/env surfaces do not imply a local execution mode.
- Concrete code removals (target end state):
  - Delete `RedesmynSettings.runner_mode` (and any env/config keys that set it).
  - Delete `LocalRunnerBackend` and “server starts agents directly” pathways.
  - Delete “server can be a repo executor” pathways (e.g., `LocalRepoExecutor` / `RepoExecutorTarget.is_local`).
  - Remove any server-side “auto-acquire primary on demand” behavior that is keyed on local mode.

2) Delete server-owned repo-local loops

- Remove server in-process repo observer loop.
- Remove server in-process agent monitor/driver.
- Remove server in-process “primary executor lease refresh” behavior.
- Remove/retire CLI and env affordances that imply server-owned observation:
  - `rn server run --observer/--no-observer`
  - `REDESMYN_NO_OBSERVER`, `REDESMYN_NO_AGENT_MONITOR`

3) Harden “daemon required” pathways

- Any endpoint that requires repo-local execution must:
  - resolve an executor target, and
  - fail with explicit guidance when no daemon is connected/attached (and/or when not primary).
- Ensure canonical operations do not silently fall back to local execution when the daemon is missing.

4) Ensure the server can run without repo FS

- Audit server start-up to avoid touching repo paths beyond what is needed to locate DB/config.
- Where repo path is currently required (e.g., to determine repo identity), ensure it’s derived from stored identity/DB,
  not by walking the filesystem.

Remove the hybrid “local mode” where the server can execute repo-local work:

- Delete or deprecate `runner_mode=local` and the corresponding in-process behaviors:
  - server-owned repo observer loop
  - server-owned agent monitor/driver loop
  - server-owned primary executor lease refresh
- Make repo-local execution always route to an attached daemon (even on localhost).
- Ensure the server can run without repo filesystem access.

## Acceptance Criteria

- The server process starts with no repo FS access (only DB access) and remains functional for read-only UI/API.
- All endpoints that require repo-local execution fail with actionable guidance when no daemon is connected/attached.
- No background tasks in the server process perform git/worktree inspection or agent supervision.
