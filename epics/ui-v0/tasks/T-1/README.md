# T-1 “Ready to merge” should align with merge semantics

## Metadata

```yaml
id: T-1
epic: ui-v0
stacked_on:
  - (none)
branch:
  suggested: rn/ui-v0/T-1-ready-to-merge-spine
```

## Problem

Today, “Ready to merge” is a per-task toggle, but the merge operation is fundamentally **spine-oriented**:

- When merging a task with `scope=descendants`, the system will fast-forward merge the task **and its unmerged ancestors on the spine** into the epic base branch.
- Merge readiness gating is checked on the **active spine** (unmerged ancestors + requested task), not only on the requested task.

This creates a predictable UX footgun:

- Users often mark a leaf task “Ready to merge” and then try to merge.
- The merge plan rejects because some upstream spine tasks are not marked ready.
- The user assumes the system is broken or that “Ready to merge” doesn’t work.

## Goal

Make the UI behavior match the merge mental model:

- When a user marks a task “Ready to merge”, the UI should (by default) mark its **unmerged ancestors** ready as well.
- The UI should communicate that “Ready to merge” is a **spine** concept, and what it implies.

## Requirements

### 1) Default behavior: auto-ready unmerged ancestors

When the user toggles “Ready to merge” **on** for task `T`:

- Identify `T`’s **merge spine** (via `stacked_on` / `parent_task_id`), from root → leaf.
- Compute the **active spine** = spine tasks that are not already “merged” (v0 proxy: `TaskState.Done`).
- Mark all active spine tasks `merge_ready_at = now` in a single operation (best-effort, but prefer atomicity).

When the user toggles “Ready to merge” **off**:

- Only clear `merge_ready_at` for the current task by default.
  - Rationale: auto-unreadying upstream tasks is surprising and can interfere with other intended merges.
  - The UI should still make it obvious if the spine is “partially ready”.

### 2) Communication: be explicit in UI

- Tooltip/copy should indicate: “Marks this task + unmerged ancestors ready”.
- If the user would otherwise be surprised (e.g. multiple ancestors will be updated), show a small, tasteful confirmation the first time per session:
  - “Mark X tasks (this task + unmerged ancestors) ready to merge?”
  - Provide a “Don’t ask again” checkbox (session-only; do not persist yet).

### 3) API / implementation seam

Prefer a single backend operation to avoid partial updates:

Option A (preferred): extend `POST /v1/tasks/{task_id}/merge-ready` to support a mode:

```json
{ "ready": true, "scope": "spine" }
```

Option B: add a separate endpoint:

- `POST /v1/tasks/{task_id}/merge-ready/spine`

Option C (fallback): UI performs N calls to the existing single-task endpoint.

Acceptance should prefer A/B over C unless there is a strong reason to keep the API minimal.

### 4) Correctness and edge cases

- Do not mark tasks without `branch_name` ready (they can’t be merged/restacked).
- Stop at the first “merged” ancestor (v0 proxy: `TaskState.Done`) when computing “unmerged ancestors”.
- If the graph has cycles or missing parents (should not happen), fail safely:
  - Mark only the current task ready and show a non-blocking warning (“Could not resolve full merge spine.”).

## Acceptance Criteria

- Marking `T-4` ready also marks `T-1..T-3` ready when those ancestors are unmerged.
- Merge no longer fails with “spine task(s) not marked ready” in the common “mark leaf ready then merge” flow.
- UI copy clearly describes what the toggle means (spine-oriented).
- Implementation is strongly typed and reuses existing merge spine logic where possible (avoid re-implementing graph traversal in multiple places).

