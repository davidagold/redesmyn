# T-5 Daemon/WS protocol + runtime integration tests

## Metadata

```yaml
id: T-5
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-5-daemon-ws
```

## Goal

Add tests that validate the server ↔ daemon coordination layer:

- presence + attachment semantics
- command enqueue + delivery
- event emission invariants (at least minimal ordering/consistency)

## Behaviors to validate

- When a daemon “attaches” to a repo instance, the server reflects it in status endpoints.
- Commands created by server actions are persisted and delivered to the correct host key.
- Merge-run events emitted by daemon execution result in correct merge run state transitions.

## Suggested approach

Depending on what’s easiest/stablest in the current codebase:

- In-process WS client (preferred if straightforward), or
- Directly drive the `DaemonConnectionRegistry` with a fake connection object and assert messages.

The primary goal is to validate the coordination behavior, not the underlying websocket library.

## Acceptance Criteria

- At least one test asserts that server-created daemon commands are delivered to the correct attached host key.
- At least one test asserts merge-run state progression driven by daemon-emitted events (happy path).

