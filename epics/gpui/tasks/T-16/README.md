---
epic: gpui
branch:
  suggested: rn/gpui/T-16-control-plane-skeleton
rn:
  node:
    branch: rn/gpui/T-16-control-plane-skeleton
  parent: T-12
---

# T-16 Headless control plane service skeleton (Domain 2)

## Problem

We want a desktop app whose UI is just one client of a **headless control plane**.

If we implement the control plane as “UI-owned state” (or as an in-process singleton with ad-hoc APIs), we will:

- reintroduce tight coupling,
- make remote/alternate clients painful,
- and undermine AI-first testability.

We also must preserve the primary boundary:

- control plane does not assume repo filesystem access and does not run repo-local actions directly.

## Goal

Create the Rust control plane as a proper service with:

- a clean in-process API for embedding (desktop app),
- a transport boundary for out-of-process clients (`rn` via Unix socket),
- and a transport boundary to the daemon (in-proc now, network later).

This ticket establishes the runtime + structure; it does not yet implement full business logic.

## Requirements

### 1) Crate structure

In the Rust workspace, establish:

- `redesmyn_control_plane` (library): the application core and service wiring.
- `redesmyn-server` (binary): a thin host that starts the service for development/headless mode.

Rules:

- `redesmyn_control_plane` must not depend on daemon-only crates (git/process execution).
- All daemon interaction must go through the transport trait defined in Domain 1/0 (T-7/T-11).

### 2) Process model

Define a minimal, explicit lifecycle:

- `ControlPlane::start(...) -> ControlPlaneHandle`
- `ControlPlaneHandle::shutdown()`

It must support:

- embedded mode (desktop app starts/stops it),
- headless mode (server binary runs it),
- and test mode (integration tests start it with in-memory/temporary DB and mock transports).

### 3) Configuration + paths

Use the typed config layer (T-5):

- DB path (SQLite via `sqlx`; split DB strategy)
- UDS path for client API server (T-12)
- optional: log dir/state dir paths

Keep defaults sensible for macOS/Linux; do not depend on repo working directory.

Split DB note:

- The control plane’s default DB path is the Rust DB: `<repo>/.redesmyn/redesmyn_rust.sqlite3`.
- The legacy DB (`<repo>/.redesmyn/redesmyn.sqlite3`) must not be migrated/written by Rust.
- Legacy → Rust import is a separate flow (see T-67).

### 4) Concurrency model

Use a simple async runtime strategy:

- `tokio` runtime (expected via `sqlx` + UDS).
- Do not build a bespoke task scheduler.
- Centralize background tasks behind a small, explicit manager (spawn + graceful shutdown).

### 5) “No silent actions” hook

Even at skeleton stage, establish the pattern:

- every mutation is represented as a **command** with a `command_id`,
- the command state machine is observable (even if initially stubbed),
- and the client protocol can subscribe to updates (T-12/T-15).

## Acceptance criteria

- There is a runnable control plane binary (`redesmyn-server`) that starts and cleanly shuts down.
- The service is embeddable: desktop app can start it in-process without reaching into internals.
- The codebase structure makes it hard to violate the daemon/control-plane boundary (dependency graph).
- A minimal integration test starts the control plane, connects a client, and performs a `Health/Status` request (method can be stubbed initially).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on Domain 1 client protocol contract (T-12) for the control plane “client-facing” surface.
- Depends on Domain 0 storage scaffolding (T-6) and config/logging conventions (T-4, T-5).

## Reference implementation (today; for behavior orientation only)

- Control plane (Python today):
  - `redesmyn/api.py` (FastAPI app; lifespan wiring; mounts dashboard assets; exposes REST `/v1/*`, UI WS `/v1/ws`, and daemon WS `/v1/daemon/ws`).
  - `redesmyn/cli.py` (`rn server run` and `rn debug dev` process model).
  - `redesmyn/settings.py` (runtime config: host/port, runner mode, feature toggles).
  - `justfile` (`just dev` and related workflows).
- Client surfaces (Python/TS today):
  - `openapi/openapi.json` (current REST surface used by the dashboard).
  - `dashboard/src/api.ts` (HTTP client for `/v1/*`).
  - `dashboard/src/hooks/useEventStream.ts` (WS client for `/v1/ws`).
- Test harness (Python today):
  - `tests/scenarios/scenario.py` (in-process ASGI app + temp repo/DB wiring).
  - `tests/test_api_integration.py` (API behavior expectations).
  - `tests/test_smoke_integration.py` (smoke coverage).
