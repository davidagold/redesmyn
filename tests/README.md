# Tests: Style Guide (v0)

This repo’s tests are intended to be:

- **Behavior-first**: most coverage should validate user-visible behavior across API + DB + git.
- **Readable**: test bodies should be short and “spec-like”.
- **Deterministic**: no flaky dependence on external services.
- **xdist-ready**: tests must not assume single-process execution or shared global state.

## Naming + structure

- Prefer module-level tests named like `test_<expected_behavior>`.
- Keep one behavior per test.
- Use marks to keep the suite legible:
  - `@pytest.mark.integration`: API + DB + git, in-process.
  - `@pytest.mark.unit`: pure-ish helpers (no IO).
  - (Later) `@pytest.mark.e2e`: Playwright happy path.

## Scenarios + fixtures

Prefer fixtures and typed “scenario” bundles over hand-rolled setup inside each test.

- **Low-level fixtures**: represent a single resource (temp git repo, temp DB, app client).
- **Scenario fixture**: a typed bundle that represents a coherent “world” (`Scenario`).
- **Variants**: scenario + one additional situation (e.g. “merged parent”, “running agent”).

Guidelines:

- Keep scenario primitives strongly typed (dataclasses are encouraged).
- Seed helpers should be named `seed_<situation>` and return a typed handle with the IDs/paths
  needed for assertions.
- Avoid reaching into “private” helpers across modules; if a helper is reused, make it part of
  the public scenario API.

## xdist readiness (required)

Write tests assuming they may run concurrently across multiple workers.

- **No shared global state**: avoid module-level singletons and caches that mutate at runtime.
- **No shared filesystem paths**: always use `tmp_path` (or `tmp_path_factory`) and isolate
  repo/worktree/db under it.
- **No shared ports**: prefer in-process clients (e.g. ASGI + `httpx`) over binding sockets.
- **No implicit env**: do not rely on ambient `REDESMYN_*` env vars across tests. If env is
  required for legacy code paths, set *all* required vars per-test and tear them down, or
  prefer passing explicit context/config objects.
- Ensure any background tasks started by app lifespans are reliably shut down by fixtures.

## When to add unit tests

Add a unit test when it provides leverage and stability:

- formatting / rendering helpers
- plan formatting
- small transformations that are hard to validate indirectly

Avoid unit tests for behavior that is already well-covered by integration tests.

## Running tests

- `just test`
- `uv run pytest`
- Integration only: `uv run pytest -m integration`
- Unit only: `uv run pytest -m unit`

When `pytest-xdist` is available, tests should also pass with:

- `uv run pytest -n auto`

