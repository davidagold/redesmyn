# T-4 Multi-repo DB + server scoping (design + UX)

## Metadata

```yaml
id: T-4
epic: backlog
stacked_on:
branch:
  suggested: rn/backlog/T-4-multi-repo-server-scope
```

## Problem

The DB schema supports multiple repositories (`repositories` table) and many epics/tasks across those repos.

However, the server today effectively runs in a **single-repo context**:

- `ctx.repo_root` is chosen at startup (`REDESMYN_REPO_ROOT` / settings / cwd).
- Most API handlers scope queries to the `Repository` row whose `repo_root == ctx.repo_root`.
- The dashboard routes (and WS stream) assume you’re looking at a single repo’s epics.

This is a sane default, but it can become confusing if:

- the user points multiple repos at a shared DB path, or
- stale rows exist in the DB from prior experiments, or
- the user expects a single long-running server to “manage multiple repos”.

In these cases, the server won’t necessarily “get confused” and operate on the wrong repo, but it can present a confusing UX:

- other repos in the DB are silently invisible
- it’s unclear whether the app is “multi-repo” or “single-repo”

## Goals

1. **Clarity**: make it obvious that the current server instance is scoped to one repo.
2. **Safety**: prevent accidental “multi-repo DB” setups from creating surprising behavior.
3. **Path to multi-repo** (optional): define what it would mean to support multi-repo in v0/v1.

## Proposed approach (v0)

### A) Keep single-repo server as the default

- Default remains: one `.redesmyn/` + one SQLite DB per git repo (`rn init`).
- Server continues to scope to exactly one `repo_root`.

### B) Detect and communicate “DB contains multiple repos”

On startup (and/or in `/v1/config` response), compute:

- the active repo (`ctx.repo_root`)
- number of other `Repository` rows present in the DB

If there are multiple:

- show a small, non-alarming info banner in the UI (or in the “Repo executor”/status card) like:
  - “Server scoped to <repo_root>. DB contains N other repos.”
- provide a “details” affordance listing the other repo roots, and a short explanation:
  - “Run the server from that repo (or set `REDESMYN_REPO_ROOT`) to view it.”

### C) Optional: add a guardrail configuration

Introduce a setting/flag like:

- `REDESMYN_ALLOW_MULTI_REPO_DB=1`

When false (default), and multiple repos are detected:

- either warn (soft) or refuse to start (hard) depending on which proves least annoying.

## Multi-repo mode (future)

If/when we want “one server manages many repos”, define it explicitly:

- UI: repo picker (or repo in URL)
- API: repo scoping parameter or per-session context for most routes
- Daemon/executor/lease routing: already repo-keyed in many places, but audit for remaining single-repo assumptions

This task should decide whether multi-repo is a real near-term goal or a long-term stretch.

## Acceptance Criteria

- The app communicates which repo the server is scoped to.
- If the DB contains multiple repos, the UI makes that discoverable and explains how to switch.
- No change in default behavior: launching from repo A still only shows repo A’s epics by default.

