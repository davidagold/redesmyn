---
epic: gpui
branch:
  suggested: rn/gpui/T-57-trunk-timeline
rn:
  parent: T-50
---

# T-57 Trunk timeline column (commit marks; base alignment; optional but planned) (Domain 6)

## Problem

The web graph includes a trunk timeline column that provides:

- a sense of base vs head positioning,
- commit-count context for edges,
- and an anchoring affordance for the graph layout.

This is useful but not strictly required for initial graph parity in the desktop port.

If we couple trunk rendering into the core graph too early, it may slow the port. If we ignore it entirely, we lose useful context for orchestration and review.

## Goal

Plan and implement trunk timeline support as an optional column:

- rendered in the graph scene,
- participates in layout offset/anchoring,
- but can be disabled without breaking the rest of the graph UI.

## Requirements

### 1) Data model

Use the EpicGraph trunk timeline data (from the control plane read model) when available.

### 2) Rendering

Render:

- a trunk “line”,
- commit marks (base/connector/ellipsis/commit),
- and basic label columns (sha/title) with zoom-aware LOD if needed.

### 3) Layout integration

Support trunk-aware layout anchoring:

- align trunk base mark with the root task column visually (as in the web UI),
- offset task columns by the trunk column width + gap.

### 4) Performance

- Use LOD and/or virtualization to avoid rendering hundreds of commit marks when zoomed out.

## Acceptance criteria

- When trunk data is present, the trunk column renders and the graph offsets appropriately.
- When trunk data is absent, the graph renders normally with no special casing leaks.

## Dependencies / sequencing

- Can be implemented after core graph rendering is stable (T-50..T-54).
- Should not block initial graph parity.

## Reference implementation (today; for behavior orientation only)

- Trunk node + layout anchoring (web today):
  - `dashboard/src/components/graph/GraphView.tsx` (trunk metrics and anchor calculation).
  - `dashboard/src/components/graph/TrunkNode.tsx`
  - `dashboard/src/components/graph/graphConfig.ts` (trunk constants).

