---
id: T-4
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-4-api-integration
---

# T-4 API integration tests (ASGI + DB)

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
  - websocket/event payloads are JSON-serializable (regression guard for datetime serialization issues)
  - 409 cases return actionable guidance (e.g., affected running agents)
  - 503 cases provide guidance when a canonical operation is underspecified (e.g., missing primary executor)
- Merge/restack endpoints:
  - accept requests and create merge run records
  - return appropriate status codes for invalid requests

## Proposed tests (names + intent)

- `test_unhandled_exception_returns_500_with_request_id_header_and_detail()`
  - Assert: `x-request-id` header is present and `detail` includes the request id.
- `test_event_payloads_are_json_serializable()`
  - Assert: event endpoints / websocket event shapes do not include raw `datetime` objects.
  - (If this fits better in T-5, keep it there, but ensure it’s covered somewhere.)
- `test_merge_returns_409_with_running_agents_prefix_contract()`
  - Assert: 409 response preserves the `RUNNING_AGENTS:` prefix contract so the UI can parse it reliably.
- `test_canonical_merge_without_primary_executor_returns_503_guidance()`
  - Assert: error message points the user to the correct remediation (attach a primary executor / start daemon).

## Suggested approach

- Use `httpx.AsyncClient` against the ASGI app object (no sockets).
- Use the scenario DB fixture(s) from T-1 for persistence.
- Where server behavior depends on repo state, use scenario repo fixtures too.

## Acceptance Criteria

- `uv run pytest -k api` passes.
- Tests exercise at least one 500-path and assert request id behavior (header and/or JSON detail).
