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

### Target layout (v0)

- Render the epic root branch (`main`) as a horizontal **trunk** across the top.
- Each node is a horizontal **lane** below; the graph reads left → right.
- Parent → child edges branch down from the parent lane/trunk and curve into the child lane.
- X-axis is **commit-count scaled** (with summarization/clamping for long ranges so the viewport remains usable).
- Root nodes attach to the trunk at their **merge-base commit** (commit-accurate).

## Acceptance Criteria

- Node positions are deterministic: reloading the page yields the same layout for the same topology.
- Layout updates are robust:
  - Re-layout triggers when epic graph data changes and when the viewport size changes.
  - Selection is preserved across re-layout (selected node/edge remains selected).
- Fit-to-view is available and feels good (can be auto-run on initial load; later exposed via a control).
- Implementation keeps future styling flexibility:
  - Layout constants (direction, spacing, node size assumptions) are centralized and easy to tweak.
- The “trunk + lanes” layout is visible in the UI (even if commit scaling is initially coarse):
  - Trunk is distinct and anchored at the top.
  - Root nodes connect to trunk with a clear branch edge.
  - Child nodes connect to their parent lane with curved edges.

## Notes / Contracts

- ELK should be treated as a pure layout step: input nodes/edges → positioned nodes/edges.
- Avoid tight coupling between layout and rendering (so we can later support alternate layouts or focus modes like “stack view”).
- If ELK fights the commit-count-scaled x-axis, use ELK for y-ordering and apply a second deterministic pass to set x positions based on commit-lengths.
