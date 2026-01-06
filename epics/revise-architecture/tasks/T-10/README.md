# T-10 Control plane: repo-rootless mode + repo selection by repo key

## Metadata

```yaml
id: T-10
stacked_on: T-9
node:
  branch: rn/revise-architecture/T-10-control-plane-repo-rootless
```

## Problem / Motivation

The **revise-architecture** epic’s core constraint is that the control plane should be deployable remotely:

- It **must not require filesystem access** to a git checkout to function.
- It should scope operations by stable repo identity (`workspace_id`, `repo_id`) rather than by a host path.

Today, `redesmyn/api.py:_lifespan` unconditionally constructs a `RepoContext` from a `repo_root` (cwd / env / settings) and many API handlers
implicitly scope to the repository row where `Repository.repo_root == ctx.repo_root`.

Even when `runner_mode == "remote"`, this keeps an old assumption alive:

- “the server is started inside a repo and is scoped by that repo’s filesystem path”

This is not compatible with a hosted control plane (no repo checkout, multiple repos, users selecting repos via URL).

This task introduces a clean separation:

- A **control plane context** that can run without a `repo_root`.
- Repo selection for API/UI by **repo key** (and/or a user-visible repo selector).

## Desired behavior

### A) Repo-rootless server mode

The server should be runnable with:

- just a DB path (or DB URL), and
- no local git repo checkout.

In this mode:

- the dashboard can still show epics/tasks for repos present in the DB
- git mutations are routed through the `RepoExecutor` interface (to a daemon/executor) and never executed in-process

### B) Explicit repo selection

The control plane should not rely on `repo_root` path equality for scoping.

Instead, scoping should be explicit and stable:

- primary selector: `(workspace_id, repo_id)` (the logical repo identity already introduced in T-9)
- UX surface: a repo picker (later) and/or repo key embedded in routes

In v0/v1, it’s OK to retain “single repo by default” behavior, but it must be implemented as an explicit selector:

- e.g. “default repo key” stored in config/DB, not “whatever repo_root the process started from”

### C) Local-first remains ergonomic

Local dev should remain “run from the repo and it just works”:

- In local mode, starting from within a repo can still auto-select the active repo key and populate local-only conveniences.
- This must not leak into the remote mode assumptions.

## Proposed implementation plan

### 1) Separate contexts: control-plane vs repo executor

Introduce (names TBD):

- `ControlPlaneContext` (or `AppContext`): DB handle, state dir, settings; no `repo_root` required.
- Keep `RepoContext` for executor/local-only operations where filesystem access is actually required.

`api.py:_lifespan` should build only what the server needs:

- always: DB + logging
- local-only: repo_root/worktree_root + `RepoContext` for local executor fastpaths

### 2) Repo scoping for API routes

Refactor the API so handlers resolve a `RepoKey` explicitly rather than inferring by `repo_root`.

Candidate approaches (pick one for v0):

1. **Path-based routing**: routes include repo identity.
   - e.g. `/v1/repos/{workspace_id}/{repo_id}/epics/...`
2. **Query parameter**: existing routes accept optional `repo_id` selector.
   - e.g. `/v1/epics?repo_id=...`
3. **Server default repo key**: stored in DB/settings; UI uses it implicitly.
   - allows keeping current URLs while still avoiding `repo_root` coupling.

Whichever approach is chosen, the key requirement is: **no path-based repo_root equality in server logic**.

### 3) Dashboard + WS stream scoping

Ensure:

- dashboard queries identify which repo they’re requesting
- event stream (`/v1/ws`) scopes to the same repo identity

### 4) Migrations / data model notes

If needed, make `Repository.repo_root` optional or clearly “host-local metadata” rather than the server’s primary key.

## Acceptance Criteria

- Control plane can start in a mode where it has no git repo checkout and still serves the dashboard/API against the DB.
- Control plane repo selection is by `RepoKey` (or an explicit stored default), not by `repo_root` path.
- Local-first mode still works with minimal friction when started inside a repo.

