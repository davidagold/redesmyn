# T-6 Graph polish: animations + performance budget

## Metadata

```yaml
id: T-6
stacked_on: T-5
node:
  branch: rn/graph-viz/task-6-graph-polish
```

## Brief (local)

- Make the graph feel “alive” and high-quality: tasteful motion and performance discipline are core to the cockpit experience.
- Smooth transitions for layout changes and graph updates (node movement + edge redraw).
- Level-of-detail rules for commit strings (avoid rendering every commit when zoomed out).
- Establish a perf budget and instrumentation (frame time, node count, edge complexity).

## Acceptance Criteria

- Layout changes animate (node movement is smooth; edges update without flashing).
- Commit-string LOD is tuned and documented (simple rules tied to zoom):
  - What renders at each zoom band.
  - How we cap per-frame work.
- A basic perf budget exists and is measurable (v0 target can be qualitative but explicit, e.g. “no visible jank for ~100 nodes on a modern laptop”).
- Implementation remains maintainable:
  - Animation/LOD parameters are centralized.
  - Rendering is structured to minimize re-renders (memoization where appropriate).
