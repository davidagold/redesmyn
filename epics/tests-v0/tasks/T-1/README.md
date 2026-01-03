# T-1 Test harness + fixtures + scenario bundles (trunk)

## Metadata

```yaml
id: T-1
stacked_on:
must_land_after: []
node:
  branch: rn/tests-v0/T-1-test-harness
```

## Goal

Establish a test framework that makes it easy to write **behavior-first** tests with minimal boilerplate, while still supporting targeted unit tests.

This task is the “trunk” dependency for the rest of `tests-v0`.

## Requirements

### Testing style

- Tests should be primarily module-level.
- Test function names should clearly communicate expected behavior.
- Encourage small, readable test bodies by moving setup/arrange work into fixtures and scenario helpers.

### Core mechanics

Implement a reusable “scenario” pattern:

- Low-level fixtures: a single resource (db, repo, app, client, daemon runtime).
- Scenario fixture(s): composed, typed bundles that represent “a working world”.
- State/scenario variants: scenario + a specific situation (e.g., “merged parent”, “running agent”, “conflicted merge run”).

### Proposed Python testing stack

- `pytest`
- `pytest-asyncio` (async tests; configured so `async def test_...` works naturally)
- `pytest-timeout` (prevent hangs)
- `httpx` (ASGI integration client)

We should avoid pulling in heavyweight frameworks unless they offer clear benefit.

## Deliverables

### 1) Test layout and configuration

- Create a top-level `tests/` tree for backend tests.
- Add pytest configuration (e.g., `pyproject.toml` or `pytest.ini`) with:
  - sensible defaults (e.g., strict markers, asyncio mode)
  - timeouts
  - short tracebacks by default (but easy to expand)

### 2) Scenario primitives (typed)

Add typed “scenario” building blocks (prefer dataclasses) to keep test bodies compact:

- `ScenarioDB`: temp db path, engine/sessionmaker, helper to run migrations.
- `ScenarioRepo`: temp git repo path, helpers to create commits/branches quickly.
- `ScenarioApp`: ASGI app instance, `httpx` client.
- `ScenarioDaemon`: helper to register a fake/real daemon connection and assert commands/events.
- `Scenario`: composed bundle (repo + db + app + daemon + helpers).

The goal is: most tests should be able to start with `scenario` (or a variant) and then focus on the behavior under test.

### 3) A small set of state/scenario variants

At minimum:

- `scenario_with_merged_parent` (effective-base relevant)
- `scenario_with_running_agent`
- `scenario_with_conflicted_merge_run`
- `scenario_without_primary_executor` (if applicable for canonical routing tests)

### 4) Conventions doc (for humans + AI)

Add a short, explicit guide for writing tests in this repo:

- naming conventions
- when to prefer integration vs unit tests
- how to create new scenario variants
- guidance on avoiding flaky tests

## Acceptance Criteria

- `uv run pytest` runs and executes at least:
  - one “smoke” integration test using the scenario fixture(s)
  - one unit-level test of a small pure-ish helper (so patterns are demonstrated)
- Test code demonstrates the intended readability (small test bodies, strong fixtures).
- Documentation clearly explains how to add new tests and scenario variants.

## Notes / Design Considerations

- Prefer **real git** in temp repos for git mechanics tests; mocks are likely to miss edge cases.
- Prefer applying Alembic migrations into temp dbs for integration tests; it catches schema drift.
- Keep the scenario API stable and additive; downstream tasks will build on it.

