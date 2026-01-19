---
epic: gpui
branch:
  suggested: rn/gpui/T-55-expanded-task-card-details
rn:
  parent: T-46
  after:
    - T-44
---

# T-55 Expanded task card: details + selection integration (no sidebar drawer) (Domain 6)

## Problem

The graph is “graph-first”, but users need an “expanded” surface for a task that includes:

- task details and controls,
- session/chat interaction and history,
- and entrypoints to related capabilities (diffs, logs, merge/restack, etc.).

In the web UI today, selection expands the task card *and* opens a right-side details panel/drawer.

For the GPUI port we are intentionally changing this UX:

- remove the sidebar details drawer,
- and unify “expandedness” into the **expanded task card itself**.

This expanded task card must **push surrounding cards out** via layout recompute so it never overlaps other nodes as it grows.

## Goal

Implement an expanded task card surface that:

- expands when a task is selected (and collapses when selection clears),
- contains both:
  - a **SessionView slot** (left) and
  - a **Details surface** (right),
- and provides a stable, legible home for actions (even if many actions are stubbed initially).

## Requirements

### 1) Selection ↔ expandedness policy (one place; no sidebar drawer)

- There is no separate details drawer/panel in the graph workspace.
- Selecting a **task node** expands that task card.
- Clearing selection collapses it.
- Only the selected task is expanded (v0: single expanded card at a time).
- Expansion triggers a node size change and a relayout (T-51) so surrounding nodes move out of the way (no overlap).

### 2) Expanded task card internal layout (Session + Details)

Inside the expanded card, implement a two-column layout:

- **Left column: Session**
  - a reserved slot for the reusable session viewer (`SessionView`).
  - v0 can render a placeholder until T-64 lands, but the layout must be designed for the real SessionView component.
- **Right column: Details**
  - structured task details and controls (see below).

Avoid path leakage:

- do not require local filesystem paths to render.

### 3) Details information architecture (legible; scalable; no “tab soup”)

The details column must be more organized and legible than the current web sidebar.

v0 requirements:

- Use a **sectioned** layout with clear headings, spacing, and subtle separators (avoid “busy border soup”).
- Prefer a lightweight “table of contents” / in-card navigation affordance (e.g. a side rail or jump list) over tabs.
  - Tabs are allowed only as a fallback if we cannot keep the content navigable otherwise.
- Each section’s content should be readable and skimmable:
  - labels aligned, values copyable where useful,
  - status chips/badges consistent with the rest of the UI.

Suggested initial sections (subject to iteration):

- Overview (title, state, merge readiness, timestamps)
- Agent (status, command/config summary, session state)
- Merge/restack (readiness, required actions; eventually actions)
- Identifiers (task id, run ids, etc.)
- Artifacts/logs entrypoints (links/actions; no giant blobs inline)

### 4) Action pattern (no silent actions)

Even if the real actions land later:

- implement the “in flight” UI pattern for at least one dummy action (wired to a no-op command) to ensure the UX pattern is correct.

### 5) Scroll + interaction ergonomics

The expanded card will be large. Ensure:

- internal scroll regions behave predictably (session feed scroll separate from details scroll),
- graph panning/zooming does not “fight” with scrolling inside the expanded card,
- focus and keyboard navigation are sane (no focus traps).

### 6) Accessibility

- panel open/close must be keyboard accessible,
- no focus traps,
- preserve user input on errors.

## Acceptance criteria

- Selecting a task expands it into a two-column (Session + Details) view; clearing selection collapses it.
- Expanded cards never overlap other nodes; expansion triggers a relayout that pushes surrounding nodes out.
- The details column is sectioned and legible; at least one action demonstrates “in-flight” UI.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on desktop chrome and selection wiring (T-46) and UI foundations (T-44).
- Integrates with the graph scene and selection model from Domain 6 (T-50).
- Relies on variable node sizing + relayout behavior (T-51/T-52).

## Reference implementation (today; for behavior orientation only)

- Web task card expansion + sidebar (today):
  - `dashboard/src/components/graph/TaskCard.tsx`
  - `dashboard/src/components/layout/DetailsPanel.tsx`
  - Selection routing in `dashboard/src/routes/EpicView.tsx`
