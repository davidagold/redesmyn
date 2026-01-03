# T-4 API integration tests (ASGI + DB)

## Metadata

```yaml
id: T-4
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-4-api-integration
```

## Goal

Add integration tests that validate core server behavior through the API surface, exercising:

- request validation
- status transitions
- error shaping (including request ids) and logging expectations

These tests should avoid real network calls by using an ASGI client.

## Behaviors to validate

- Starting / restarting an agent returns correct payload and persists expected DB state.
- Error handling:
  - unhandled exceptions return 500 with request id
  - 409 cases return actionable guidance (e.g., affected running agents)
  - 503 cases provide guidance when a canonical operation is underspecified (e.g., missing primary executor)
- Merge/restack endpoints:
  - accept requests and create merge run records
  - return appropriate status codes for invalid requests

## Suggested approach

- Use `httpx.AsyncClient` against the ASGI app object (no sockets).
- Use the scenario DB fixture(s) from T-1 for persistence.
- Where server behavior depends on repo state, use scenario repo fixtures too.

## Acceptance Criteria

- `uv run pytest -k api` passes.
- Tests exercise at least one 500-path and assert request id behavior (header and/or JSON detail).

