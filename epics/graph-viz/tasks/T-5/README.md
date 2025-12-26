# T-5 Commit strings (rendering + edge interactions)

## Metadata

```yaml
id: T-5
stacked_on: T-3
must_land_after:
  - T-4
node:
  branch: rn/graph-viz/task-5-commit-strings-rendering
```

## Brief (local)

- Make “commit strings” feel like a real object in the graph (not a tooltip): selectable, inspectable, and stylable.
- Implement a custom edge renderer for “commit strings”:
  - Selectable/hoverable edges.
  - A minimal per-commit representation that scales with zoom (LOD).
- Surface edge selection in the Details panel (and later: commit selection).

## Acceptance Criteria

- Edges render via a custom edge component that we fully control (styling + hit targets).
- Edge interaction:
  - Hover and selection states are visually distinct but not noisy.
  - Selecting an edge updates the Details panel with commit-string information (at minimum: count + range base/head).
- Level-of-detail behavior exists:
  - Zoomed out: edges are simple (line + optional count).
  - Zoomed in: edges can show richer “commit string” detail (ticks/segments) without tanking performance.
 - Commit-count scaling is respected:
   - Edge geometry or glyph density communicates relative commit length (with summarization/clamping so long ranges don’t blow out the viewport).

## Notes / Contracts

- This task depends on:
  - XYFlow viewport foundation (T-1),
  - deterministic layout for stable geometry (T-3),
  - commit-string data availability (T-4).
- Avoid over-rendering: commit-level glyphs should appear only when useful (zoom threshold) and should be computed efficiently.
