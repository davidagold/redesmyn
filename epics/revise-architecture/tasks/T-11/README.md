---
rn:
  node:
    branch: rn/revise-architecture/T-11-remove-rn-git
  linear:
    issue_id: 8c20381d-f484-42d8-a126-2006535e4799
    identifier: RED-29
  parent: T-7
---

# T-11 Remove `rn git` / `git_proxy`; rely on daemon-side observation for projections

## Motivation / context

Today we have a special CLI surface area:

- `rn git …` (in `redesmyn/cli.py`)
- `redesmyn/git_proxy.py` (local “policy” for blocking certain mutating git subcommands)

This exists to:

1. Provide a “safe” way for humans to run git commands, with optional enforcement (block or warn on mutating commands).
2. After a successful mutating command, best-effort call `update_git_projections()` so the UI/DB reflect changes quickly.

However, this is *additional git API surface area* that we have to support and document, and it weakens the architectural story:

- It implies users need a special command to keep the system coherent after local git actions.
- It duplicates responsibilities that should belong to the daemon-side repo observation loop (telemetry/projections).
- It conflicts with the direction of “control plane doesn’t run git” *and* “git-derived projections come from the repo executor / daemon side”.

We want to remove `rn git` and `git_proxy` as first-class product concepts.

## Desired outcome

After this task:

- Users can run plain `git …` and the system will still converge quickly.
- Git-derived projections/events remain produced by the repo executor / daemon-side observation, not by a special CLI proxy.
- We reduce maintenance surface area by removing `rn git` and its “blocking rules” module.

## Dependencies

- This task is downstream of the “daemon-side observation loop exists and is connected” work (T-2/T-3).
- It should be compatible with repo executor routing/leases (T-9):
  - Projections should be attributed to a specific repo instance (`host_key`) and must not assume a single host.

## Scope

### 1) Remove `rn git` and `git_proxy`

- Remove the `rn git` CLI command entirely (and any docs/UI references to it).
- Delete `redesmyn/git_proxy.py` and any code paths that exist solely to support it.
- Ensure any existing users who may have been relying on it get a clear migration message:
  - If we keep a stub, it should print a short deprecation error with guidance.
  - Prefer removal if we are confident no other code depends on it.

### 2) Ensure projections update without `rn git`

We currently rely on `update_git_projections()` (or equivalent logic) being run after local git mutations.
After removing `rn git`, we need an alternate trigger that is always available:

- Integrate projection refresh into the daemon-side repo observation loop (preferred), or
- Add a small projection refresh loop per attached repo in local mode, or
- Trigger projection refresh after repo-mutating operations that we initiate (merge/restack/worktree operations), plus a periodic refresh for “out-of-band” git actions.

Implementation expectations:

- Changes should be debounced (avoid hammering git/DB).
- Use best-effort semantics: failures should not crash the daemon/server; they should log and retry.
- In local dev, the UI should see updated trunk timeline / merge-base / ref-moved state quickly after a `git commit`, `git rebase`, or `git checkout`.

### 3) Update documentation and architecture narrative

- Update `epics/revise-architecture/tasks/T-7/README.md` (and any other doc) to remove the “keep `rn git` as local proxy” recommendation.
- Clarify that “daemon observation / repo executor projections” is the single system-wide mechanism for discovering git changes (including those made outside Redesmyn).

## Non-goals

- Perfect “push-based” git change detection on every platform/filesystem (we can use polling + debouncing initially).
- Replacing `rn git`’s blocking UX with a new enforcement mechanism. If we want policy enforcement later, it should be designed separately (likely via hooks or repo executor policy), not via a CLI proxy.

## Acceptance criteria

- `rn git` no longer exists (or is a stub that cleanly errors with a short deprecation message).
- `redesmyn/git_proxy.py` is removed and there are no remaining references.
- After running a plain `git commit` in a task worktree, the dashboard updates:
  - the task branch SHA / trunk timeline / merge-base projections update without restarting the server/daemon.
- After running a plain `git rebase` (or `git checkout`) that changes branch tips, the dashboard converges similarly.
- The updated approach is documented in the revise-architecture epic docs (T-7 updated, and this task linked from the epic task map).
