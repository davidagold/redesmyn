---
id: T-11
epic: ui-v0
branch:
  suggested: rn/ui-v0/T-11-graph-node-expand-reflow
---

# T-11 Graph: animate task card expansion with layout reflow

## Problem

Today, task cards are layouted as fixed-size nodes (constant height) and any “expanded” UI is rendered as out-of-node chrome.
This avoids layout churn, but it prevents the “click to expand” interaction from feeling native: expansion does not push neighboring nodes away.

We want a more delightful interaction: click the agent preview line → the task node expands smoothly, and the graph reflows to make room.

## Goal

Support variable-height task nodes in the graph layout so a task card can expand/collapse in-place with:

- smooth animation,
- no overlapping nodes or callouts,
- stable edge routing/handles, and
- acceptable performance for typical v0 graphs.

## Requirements

- Clicking the task card agent-preview row expands/collapses that card.
- Expanded state increases the node’s layout height (not just visual overflow).
- The graph recomputes layout when node heights change and animates nodes to their new positions (no “teleport”).
- Expanded content lives inside the card (not absolute callouts) so it participates in layout sizing.
- Expanded state should be scoped to selection (collapse when deselected) unless explicitly pinned (optional follow-up).
- Keep the UI visually calm: avoid adding borders around the expanded region; prefer spacing and subtle affordances.
- Composer UI should use the shared shadcn-style `Textarea` component; keep the send button below the textarea so the input can take full width.
- Avoid layout jitter:
  - changes should be deterministic and stable across repeated expand/collapse,
  - avoid oscillations caused by measurement feedback loops.
- Ensure callouts/error panels continue to work without overlapping nearby nodes (may require moving some callouts into the expanded region or reserving layout gap).

## Notes / Implementation sketch

- Lift “expanded node ids” state to the graph view layer so layout inputs can depend on it.
- Provide per-node `height` to ELK layout (instead of a global constant).
- Consider a small set of discrete heights (collapsed/expanded) first; avoid measuring dynamic content at runtime unless required.
- Animate:
  - node `position` changes (existing layout animation),
  - and the card’s internal height (`max-height`/`height` transition) for the expansion itself.
- Decide on handle behavior:
  - keep handles pinned at a stable Y (preferred to reduce edge “swim”), or
  - recenter handles as height changes (simpler but more motion).

## Acceptance Criteria

- Expanding a node pushes other nodes away (no overlap) and the graph remains readable.
- Expand/collapse feels smooth (positions and card height animate).
- No significant performance regression on ~100 node graphs.
- `just check` remains green.
