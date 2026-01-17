---
rn:
  node:
    branch: rn/git-mechanics-v1/T-3-abort-merge-run
---

# T-3 Abort merge runs (cancel) + future rollback design

## Plan

Add an **Abort merge run** control (UI + API) that cancels an in-progress merge run safely and predictably, without attempting to undo completed git operations.

This task also records the **design requirements for a future “Rollback” feature** that *does* undo completed steps (but is explicitly out-of-scope for Abort).

### Terminology

- **Abort**: stop further execution of the merge run and (when applicable) abort the *currently in-progress git operation* (e.g. `git rebase --abort`). Abort **does not** attempt to revert already-completed steps (rebases already finished, base already fast-forwarded, etc.).
- **Rollback (future)**: undo completed steps by returning branches (and the epic base branch) to known previous SHAs captured at plan start.

## Abort semantics (v0)

### High-level contract

- Abort always results in the merge run being marked **`canceled`** (or equivalent terminal status) in the DB/UI.
- Abort must be explicit, confirmable, and should clearly communicate: **“Cancel does not roll back completed steps.”**
- Abort should not attempt to “rewind” already-applied rebases / fast-forwards.

### Behavior by merge run status

Assume merge run statuses include: `running`, `blocked`, `resumable`, `succeeded`, `failed`, `canceled`.

- **`running`**
  - Semantics: request cancellation; the executor stops at the next safe boundary (between plan steps).
  - Notes:
    - We should not try to kill an in-flight git subprocess mid-step.
    - Implementation requires an execution-time “should stop” check between steps.

- **`blocked`**
  - Expected typical case: a rebase conflict with an in-progress rebase in the blocked worktree.
  - Semantics:
    - If `blocked_step_kind == "rebase"` and the worktree indicates an in-progress rebase, run `git rebase --abort` in `blocked_worktree_path`.
    - Then mark the merge run `canceled`.
  - Edge cases:
    - If the worktree path is missing/unavailable, fail with a clear error message and remediation (“Open the worktree and abort manually…”).
    - If there is no in-progress rebase/merge operation, do not run abort; still allow canceling the run (it’s the run that is being canceled), but message “No git operation to abort”.

- **`resumable`**
  - Semantics: there is no in-progress git operation; abort simply marks the merge run `canceled`.

- **`failed` / `succeeded` / `canceled`**
  - Abort control should be hidden/disabled (terminal states).

### Interaction with merge strategies

There are multiple “merge+restack” strategies (e.g. strict restack-before-merge vs merge-then-restack).

Abort behavior is the same across strategies: **cancel the run; do not roll back.**

However, messaging should reflect likely outcomes:

- **Merge and Restack (strict)**: canceling while blocked likely means the epic base has *not* yet moved, but some rebases might already have completed.
- **Merge then Restack (relaxed)**: canceling while blocked in descendant restack likely means epic base/spine *may already have been fast-forwarded*; canceling leaves that merge intact and stops further restack.

## UI/UX requirements

### Where to surface Abort

Surface Abort in the same two places as Resume, but make it a slightly higher-friction action:

1) **Details panel → Merge Run callout (primary)**
   - Add an Abort button in the callout header row, next to `Resume merge` (when present) and the expand/collapse control.
   - Contextual label:
     - If `blocked_step_kind == "rebase"`: `Abort rebase`
     - Else: `Abort merge run`
   - Always behind a confirmation dialog.
   - Tooltip/explainer: “Cancels this merge run; does not roll back completed steps.”

2) **Graph node ellipsis menu (secondary)**
   - Add a destructive menu item at the bottom (separated): `Abort merge run` / `Abort rebase`.
   - Only shown/enabled when `status ∈ {running, blocked, resumable}`.

### Confirmation / safety rails

- Abort should always require confirmation.
- If abort affects running agents in the run’s affected set, we should reuse the existing batched confirmation pattern (“this affects running tasks/agents — proceed?”).

### Progress/events

- Abort should emit a websocket event indicating the merge run was canceled (and optionally whether a git operation was aborted).

## API / backend requirements

### Endpoint

Add an explicit endpoint (names illustrative):

- `POST /v1/merge-runs/{run_id}/abort`
  - Request: `{ allowRunning?: boolean }` (for batched confirmation parity)
  - Response: `{ runId: string }`

### Implementation notes

- For `running`: store a cancel request in the DB so the executor can stop between steps.
- For `blocked` with rebase in progress: run `git rebase --abort` in the blocked worktree before marking canceled.
- Persist cancellation and blocked/abort details in `merge_runs` so the UI can reflect accurate state.

## Future: Rollback (explicitly separate feature)

Abort intentionally does not roll back. For a future **Rollback** feature:

- Capture a snapshot of the **original** SHAs at plan start:
  - epic base branch SHA
  - each affected branch SHA (and optionally upstream/base refs)
- Store this snapshot in the merge run plan (`merge_runs.plan`) so rollback can be executed later.
- Rollback should likely be:
  - explicit and gated behind strong confirmation
  - forceful (will rewrite branch refs)
  - safety-checked (dirty worktrees / in-progress operations / running agents)
  - modeled as its own “rollback plan” with stepwise progress + resumability

## Acceptance Criteria

- UI surfaces Abort control in the Details panel Merge Run callout and the node ellipsis menu, with confirm dialogs.
- Aborting a `blocked` rebase runs `git rebase --abort` when applicable and marks the run `canceled`.
- Aborting a `running` merge run cancels at the next step boundary (best-effort).
- Aborting a `resumable` merge run marks the run `canceled` (no git operation performed).
- UI messaging clearly distinguishes Abort vs Rollback (“Abort does not roll back completed steps”).
- A follow-up note/design section exists (in this task) for a future Rollback feature and its required plan snapshot data.
