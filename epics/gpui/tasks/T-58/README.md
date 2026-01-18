---
epic: gpui
branch:
  suggested: rn/gpui/T-58-graph-testability
rn:
  parent: T-48
  after:
    - T-50
---

# T-58 Graph testability surfaces (extend UI driver + semantic snapshot for graph) (Domain 6)

## Problem

We require AI-first testability:

- tests must be able to trigger actions,
- and assert both model and GUI state.

Domain 5 establishes a UI driver + semantic UI snapshot.

For the graph specifically, we need stable, semantic assertions for:

- selection state,
- camera state,
- and key UI affordances (details panel open, multi-select bar visible, etc.).

## Goal

Extend the desktop UI driver and semantic UI snapshot to cover graph behaviors so we can write deterministic UI tests without pixel diffs.

## Requirements

### 1) Driver actions

Add driver actions:

- select node by id
- clear selection
- toggle focus mode
- open/close details panel (if separate from selection)
- multi-select add/remove node
- (optional) zoom/pan to deterministic positions

### 2) Semantic snapshot fields

Add snapshot fields sufficient to assert graph state:

- selected node id / edge id
- multi-selected node ids (sorted)
- focus mode enabled
- details panel open + which entity it reflects
- camera: zoom + pan (coarse values acceptable; do not require exact floats)

### 3) Deterministic waiting

Provide a wait primitive based on semantic snapshot predicates:

- wait until “layout settled”
- wait until “selection applied”

Avoid sleeps.

## Acceptance criteria

- A deterministic test can:
  1) open an epic,
  2) select a node by id,
  3) assert selection + details panel state via semantic snapshot,
  4) toggle focus mode and assert the snapshot changes.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on the UI driver contract (T-48) and the graph scene (T-50).

## Reference implementation (today; for behavior orientation only)

- Browser-driven UI tests today:
  - `tests/e2e/test_ui_happy_path.py` (Playwright selectors such as task card click + details open).
  - Graph routing/selection in `dashboard/src/routes/EpicView.tsx` and `dashboard/src/components/graph/GraphView.tsx`.

