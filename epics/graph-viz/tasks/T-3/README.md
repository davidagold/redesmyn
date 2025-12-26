# T-3 Deterministic layout (ELK) + fit-to-view

## Metadata

```yaml
id: T-3
stacked_on: T-1
node:
  branch: rn/graph-viz/task-3-elk-layout
```

## Brief (local)

- Make the graph legible at-a-glance: a stable, deterministic layout dramatically reduces cognitive load when exploring the map of work.
- Integrate ELK to compute stable node positions from the branch topology (tree layout).
- Re-layout on graph changes and container resize (without jank).
- Add fit-to-view and “keep focus in view” behaviors.

## Acceptance Criteria

- Node positions are deterministic: reloading the page yields the same layout for the same topology.
- Layout updates are robust:
  - Re-layout triggers when epic graph data changes and when the viewport size changes.
  - Selection is preserved across re-layout (selected node/edge remains selected).
- Fit-to-view is available and feels good (can be auto-run on initial load; later exposed via a control).
- Implementation keeps future styling flexibility:
  - Layout constants (direction, spacing, node size assumptions) are centralized and easy to tweak.

## Notes / Contracts

- ELK should be treated as a pure layout step: input nodes/edges → positioned nodes/edges.
- Avoid tight coupling between layout and rendering (so we can later support alternate layouts or focus modes like “stack view”).
