---
epic: gpui
branch:
  suggested: rn/gpui/T-56-graph-multiselect
rn:
  parent: T-50
  after:
    - T-44
---

# T-56 Bulk selection + action bar (multi-select UX) (Domain 6)

## Problem

Orchestrating many tasks requires bulk operations:

- start/stop multiple agents,
- mark merge readiness for multiple tasks,
- and batch review actions.

The web UI implements a lightweight selection bar when multiple nodes are selected.

## Goal

Implement multi-selection and a selection action bar in the GPUI graph:

- selection set management,
- visible selection bar when `selected.len() > 1`,
- and a minimal set of bulk actions (even if stubbed).

## Requirements

### 1) Selection semantics

- Click selects one.
- Cmd/Ctrl+Click toggles additive selection.
- Escape clears selection.
- Selection is reflected in semantic UI snapshot (T-58).

### 2) Selection bar behavior

- Appears when multiple nodes selected.
- Animates in/out calmly (no spinners).
- Shows:
  - count selected,
  - and at least one bulk action button.

### 3) No silent actions

- Bulk actions must show immediate in-flight feedback and prevent duplicate requests.

### 4) Guardrails

- Bulk actions should respect task state (e.g., do not “start agent” for done tasks).
- If an action is disabled, show an actionable disabled reason.

## Acceptance criteria

- Multi-selection works (additive toggle).
- Selection bar appears/disappears correctly and is keyboard accessible.
- At least one bulk action executes through the normal “command” pattern (even if it is a stub command in v0).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on graph scene scaffolding (T-50) and UI foundations (T-44).
- Real bulk actions will depend on control plane command surfaces (Domain 4/2), but the selection UX should be built independently.

## Reference implementation (today; for behavior orientation only)

- Web selection bar behavior (today):
  - `dashboard/src/components/graph/GraphView.tsx` (selection bar state and animations).

