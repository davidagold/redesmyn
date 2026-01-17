---
id: T-8
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-8-playwright-happy-path
---

# T-8 Playwright thin happy path (UI end-to-end)

## Goal

Add a single Playwright test that proves the full system stays wired:

- backend is reachable
- dashboard loads
- core graph/task UI functions end-to-end

This is intentionally “thin” in v0: one high-value path, kept stable and low-flake.

## Marker / execution model (pytest strict markers)

This repo uses `--strict-markers`. The implementer should add a dedicated marker (e.g. `e2e`) and ensure Playwright tests run under that marker:

- add `e2e: browser-based end-to-end tests` to pytest markers (e.g. in `pyproject.toml`)
- the default test run should be able to exclude e2e when desired (e.g. CI splits)

## Constraints

- The test should be robust: explicit readiness checks and timeouts.
- Avoid depending on external services.
- Prefer seeding state via the backend/scenario setup rather than via manual UI steps when possible.

## Candidate happy path (proposal)

Exact flow can be adjusted, but it should cover wiring across:

1. Start backend (and any required companion processes) in a known test mode.
2. Navigate to an epic graph page.
3. Select a task card and open the details drawer.
4. Trigger a lightweight action that exercises a real API call and UI update (e.g., open git actions menu, or start an agent if safe in test mode).
5. Assert that expected UI elements appear and no fatal errors occur.

## Proposed tests (names + intent)

- `test_ui_happy_path_loads_graph_and_opens_task_details()`
  - Assert: dashboard loads, graph renders at least one task card, details panel opens, and a trivial action (menu open / refresh) succeeds.

## Deliverables

- Playwright config and a single test file.
- Minimal helper scripts to run the system under test reliably (ports, env vars, cleanup).
  - Prefer one command that starts the backend in a deterministic “test mode” (fixed port, isolated DB/state dir).

## Acceptance Criteria

- One Playwright test runs locally with a single command (documented in this task).
- The test is stable (no flaky sleeps; use waits for explicit conditions).

## Running locally

- Build prerequisites (once): `just install`
- Run the E2E test: `just e2e`
