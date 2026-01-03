# T-6 Projections / graph integration tests

## Metadata

```yaml
id: T-6
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-6-projections
```

## Goal

Validate that the graph/projection output (what the UI consumes) correctly reflects:

- repo/task topology
- task state
- merge run/agent status overlays

## Behaviors to validate

- A simple epic with a small task stack produces stable, expected graph output.
- Status overlays (agent running/errored, merge run running/blocked) appear in graph payloads.
- Projection updates are applied correctly after events (where applicable).

## Suggested approach

- Use scenario fixtures to create a small, deterministic set of tasks + branches + events.
- Assert on high-level shape (nodes/edges/status fields), not fragile pixel/layout details.

## Acceptance Criteria

- Graph/projection tests pass reliably and do not rely on UI rendering.

