---
epic: gpui
branch:
  suggested: rn/gpui/T-24-repo-registry-attach
rn:
  parent: T-23
---

# T-24 Repo registry + attachment semantics (no path leakage) (Domain 3)

## Problem

The control plane must not know repo filesystem paths.

The daemon must be able to:

- attach/detach repos by stable identity (`workspace_id` + `repo_id`),
- resolve those identities to local paths using local configuration/registry,
- and refuse to attach repos it cannot resolve safely.

Without a clear repo registry and attachment model, we risk:

- path leakage across boundaries,
- brittle “it works only on the dev machine” behavior,
- and unclear semantics when multiple repos exist.

## Goal

Implement a daemon-side repo registry and repo attachment state machine that is:

- explicit and typed,
- safe (no accidental attachment to the wrong path),
- and easy for AI and humans to drive/test.

## Requirements

### 1) Repo registry model

Define a persistent local registry mapping:

- `RepoScope { workspace_id, repo_id }` → `RepoRootPath`

Plus optional metadata:

- display name
- last_seen_at
- optional “trusted” flag

Storage approach:

- Use a simple local file (TOML/JSON) under the daemon state directory (from config).
- Do not store in the control plane DB.

### 2) Attach/detach semantics

Implement daemon attach rules:

- attach requests are expressed only in terms of repo identity (scope), never paths.
- daemon resolves identity to a local path via the registry.
- daemon validates the path is a git repository and computes/validates repo identity (repo_id) to prevent mismatches.
- attach is idempotent: attaching an already-attached repo is a no-op (but refreshes status).
- detach is idempotent.

### 3) Registration flow

Define how new repos enter the registry:

- `rn` (or desktop UI) issues a control-plane command that results in a daemon-side “register repo” operation, OR
- the daemon can “discover and register” a repo when run in a dev mode (explicit flag).

Pick one as the default for this epic and document it.

Constraint:

- even if registration is initiated by a client, the repo path must remain local-only (the client may run on the same host as the daemon in v0, but the control plane must not become the path broker).

### 4) Telemetry integration

On attach:

- daemon begins per-repo telemetry (T-28) and reports attached scopes in heartbeats.

On detach:

- daemon stops per-repo background tasks and reports detachment.

### 5) Testability

Provide tests that:

- create a temporary git repo fixture,
- register it locally,
- attach/detach by repo scope,
- and validate mismatch detection (registry identity does not match actual repo).

## Acceptance criteria

- Daemon can attach a repo by stable identity with no filesystem path crossing the daemon/control-plane boundary.
- Identity mismatch is detected and produces a structured, actionable error.
- Attach/detach is idempotent and safe.
- Tests are deterministic and cheap to run.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on daemon runtime skeleton (T-23).
- Informs observation/worktrees/git subsystems (T-27, T-28) which require an attached repo root.

## Reference implementation (today; repo identity/attach orientation only)

- Repo identity (Python today):
  - `redesmyn/repo_identity.py` (`DEFAULT_WORKSPACE_ID`, `RepoKey`, `compute_repo_id()`).
  - `redesmyn/db/models.py` (`Repository` row includes `workspace_id`, `repo_id`, `repo_root`).
  - `redesmyn/orchestrator.py` and `redesmyn/cli.py` (`rn init` and repo initialization paths).
- “Attach” semantics today (Python):
  - `redesmyn/ws_protocol.py` (`DaemonHello.attached_repos`, `DaemonHeartbeat.attached_repos`).
  - `redesmyn/api.py` (`daemon_ws()` stores attached repos on connection and updates on heartbeat).
  - `redesmyn/ws_runtime.py` (`DaemonConnectionRegistry` stores attached repo scopes and is queried by `/v1/daemons`).
- Tests (Python today):
  - `tests/test_daemon_ws_runtime_integration.py` (attaching a repo updates repo executor status in epic graph).
