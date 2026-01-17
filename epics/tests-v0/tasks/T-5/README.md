---
rn:
  node:
    branch: rn/tests-v0/T-5-daemon-ws
  parent: T-1
  after: []
---

# T-5 Daemon/WS protocol + runtime integration tests

## Goal

Add tests that validate the server ↔ daemon coordination layer:

- presence + attachment semantics
- command enqueue + delivery
- event emission invariants (at least minimal ordering/consistency)

## Behaviors to validate

- When a daemon “attaches” to a repo instance, the server reflects it in status endpoints.
- Commands created by server actions are persisted and delivered to the correct host key.
- Merge-run events emitted by daemon execution result in correct merge run state transitions.

## Guardrails / what not to test

- This task should validate **coordination contracts** (routing, delivery, state transitions), not UI rendering or git semantics.
- Prefer asserting on **stable contracts** (payload shape, state progression) over timing-sensitive ordering unless there is an explicit ordering invariant.

## Suggested approach

Depending on what’s easiest/stablest in the current codebase:

- Prefer: drive the daemon registry with a **fake in-process connection** and assert messages (less flaky than a websocket client).
- Acceptable: an in-process websocket client if it’s already straightforward and stable in this repo.

The primary goal is to validate the coordination behavior, not the underlying websocket library.

## Proposed tests (names + intent)

- `test_daemon_attach_updates_repo_executor_status()`
  - Assert: once attached, repo/host shows up in repo-executor status payloads.
- `test_server_routes_commands_to_attached_host_key()`
  - Assert: a server-initiated command destined for host K is delivered only to the connection for K.
- `test_merge_run_events_progress_merge_run_state_happy_path()`
  - Emit a minimal set of merge-run events and assert the server persists the expected status progression.
- `test_ws_event_stream_payload_is_json_serializable()`
  - Regression test for datetime serialization: ensure event payloads are dumped with `mode="json"` (or equivalent) so `send_json` never sees raw datetimes.

## Acceptance Criteria

- At least one test asserts that server-created daemon commands are delivered to the correct attached host key.
- At least one test asserts merge-run state progression driven by daemon-emitted events (happy path).
