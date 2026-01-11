# T-13 Git action status indicators: reflect run scope and true blocker

## Metadata

```yaml
id: T-13
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-13-git-action-status-indicators
```

## Problem

Git actions like merge/restack operate on a **run** that can touch many tasks (e.g. a rebase cascade).

Today, the “overall git action status” indicator is shown on the **task where the action was initiated** (e.g. a small status icon in the task card corner).
This can be visually confusing because:

- the run’s actual blocker is often **not** the initiating task (e.g. a downstream rebase conflict),
- the downstream blocker is already surfaced correctly via callouts on the affected task(s),
- the initiating task’s corner indicator can read as “something is wrong with this task”, even when the issue is elsewhere,
- it makes it harder to understand “what is happening now” and “where should I look” without scanning multiple surfaces.

(See the referenced screenshot example in this ticket description.)

## Goal

Revise git action status indicators so the UI communicates:

- the run’s *scope* (multi-task),
- the run’s *current phase* (what it is doing now),
- the *true blocker* (which task is blocked and why),

without implying that errors belong to the initiating task when they do not.

## Requirements

### 1) Separate run-level vs task-level state

Make a clear distinction between:

- **Run-level status** (merge/restack run: phase/progress/blocking reason), and
- **Task-level status** (this specific task is conflicted / blocked / updated / completed by the run).

Avoid a single icon that tries to represent both.

### 2) Place run-level status where it reads as global

Represent run-level status in a location that clearly implies “this is a run affecting multiple tasks”, such as:

- a merge/run panel/callout surface,
- a top/bottom run status pill in the graph view,
- or another global-but-calm container.

The run-level status should directly identify the current blocker, e.g. “Restack blocked on T-7 (conflict)”.

### 3) Task cards should only show task-local semantics

If task cards show a status indicator, it should be task-local and unambiguous (e.g. “this task is conflicted”, “this task was updated”, “this task is currently being rebased”).

If we keep any “initiated here” affordance, it must not look like a task-local error state.

### 4) Keep the UI calm and readable

- Avoid noisy indicators and heavy borders.
- No spinner wheels.
- Prefer refined, clearly visible patterns (e.g. subtle glow/pulse, animated ellipses, concise status text).
- Shimmering text remains reserved for LLM generation.

### 5) Make “where to go next” obvious

When blocked:

- The run-level status should provide a clear next step and a direct navigation affordance (e.g. select/focus the blocked task).
- The blocked task callout remains the source of truth for the detailed error and resolution guidance.

## Acceptance Criteria

- When a merge/restack run affects multiple tasks, the UI does not misattribute the run’s error state to the initiating task.
- A user can answer within ~1 second:
  - “What action is running?”
  - “Where is it blocked (which task)?”
  - “What should I do next?”
- The new indicators remain calm (no spinners, no noisy animation) and consistent with UI v0 style.
