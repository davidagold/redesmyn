---
epic: gpui
branch:
  suggested: rn/gpui/T-47-epic-scoped-session-pane
rn:
  parent: T-46
  after:
    - T-40
---

# T-47 Epic-scoped session selection (one per epic) + persistence seam (no “overseer” naming) (Domain 5)

## Problem

We want the desktop shell to host a persistently visible left session pane.

This pane represents an epic-scoped session:

- one per epic,
- not task-bound,
- used for check-ins/review/orchestration work,
- but **not** introduced as a new user-facing construct (“overseer” is developer shorthand only).

If we don’t define how this session is identified and persisted, later work will either:

- invent ad-hoc UI-only state (hard to test and not shareable across clients), or
- accidentally turn “overseer” into a first-class domain object.

## Goal

Define and implement the *minimal seam* that lets the UI obtain the epic-scoped session id and render it in the left pane without introducing a new conceptual object.

## Requirements

### 1) Session scoping model

Extend session identity to support a scope like:

- `Task { task_id }` (normal task sessions)
- `Epic { epic_id }` (epic-scoped session for the left pane)

This is still “just a session”; it is not a new user-facing entity.

### 2) Control plane API surface

Expose a method (name illustrative) over the client API:

- `GetOrCreateEpicSession { epic_id } -> { session_id }`

Semantics:

- exactly one epic-scoped session exists per epic (idempotent),
- session id is stable across restarts,
- session is repo-scoped (by the epic’s repo scope).

### 3) Persistence

Persist the mapping in the control plane (preferred), e.g.:

- store the epic session id on the epic row, or
- store in a small `epic_sessions` table keyed by `epic_id`.

Do **not** store this as UI-only local state; we want:

- multi-client consistency (desktop + `rn` + future clients),
- and AI-first testability (stable ids and queryable state).

### 4) UI integration

When the selected epic changes:

- UI resolves the epic’s session id via the control plane and binds the left pane to it.

Do **not** implement default targeting of selected tasks from this pane in the port.

### 5) Future direction (not in this ticket)

We may later add explicit UX affordances to “act on selected task(s)” from the epic session, but this is out of scope for the port.

## Acceptance criteria

- Selecting an epic results in a stable epic-scoped session id.
- The desktop left pane can bind to that session id (even if the view is still a placeholder).
- The mapping is persisted in control plane storage and is queryable.

## Dependencies / sequencing

- Depends on control plane session persistence (T-40) and client query surfaces (T-12/T-20).
- Builds on desktop shell chrome (T-46) for epic selection integration.

## Reference implementation (today; for orientation only)

- There is no epic-scoped session today; this is new.
- Task-scoped messaging and session hints today:
  - `dashboard/src/components/graph/TaskCard.tsx` (task message composer).
  - `redesmyn/task_agent_messaging.py` (send-message semantics).

