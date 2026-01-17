---
id: T-6
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-6-projections
---

# T-6 Projections / graph integration tests

## Goal

Validate that the graph/projection output (what the UI consumes) correctly reflects:

- repo/task topology
- task state
- merge run/agent status overlays

## Behaviors to validate

- A simple epic with a small task stack produces stable, expected graph output.
- Status overlays (agent running/errored, merge run running/blocked) appear in graph payloads.
- Projection updates are applied correctly after events (where applicable).

## Keep assertions stable (avoid UI/layout coupling)

Prefer asserting on:

- nodes and edges derived from task topology (`parent_task_id`)
- overlay fields exposed in the API contract (merge-run summaries, agent session summaries)
- trunk timeline presence/absence semantics (if returned), but not pixel/layout placement

Avoid asserting on:

- exact node coordinates/layout decisions (that belongs to UI tests, and is brittle)
- incidental ordering unless the API contract promises ordering

## Proposed tests (names + intent)

- `test_epic_graph_includes_expected_task_nodes_and_parent_links()`
  - Given a small deterministic task tree, assert returned nodes include `parentTaskId` as expected.
- `test_epic_graph_includes_agent_session_overlay_fields()`
  - Seed agent session rows and assert the graph response includes the expected latest session per task.
- `test_epic_graph_includes_merge_run_overlay_fields()`
  - Seed merge run rows and assert they appear in the graph response with correct status/operation fields.
- `test_epic_graph_includes_trunk_timeline_when_available()`
  - If trunk data exists for the primary executor, assert it is returned; otherwise trunk is null.

## Suggested approach

- Use scenario fixtures to create a small, deterministic set of tasks + branches + events.
- Assert on high-level shape (nodes/edges/status fields), not fragile pixel/layout details.

## Acceptance Criteria

- Graph/projection tests pass reliably and do not rely on UI rendering.
