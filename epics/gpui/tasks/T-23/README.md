---
epic: gpui
branch:
  suggested: rn/gpui/T-23-daemon-skeleton
rn:
  parent: T-11
---

# T-23 Daemon runtime skeleton (service, lifecycle, control-plane connection) (Domain 3)

## Problem

The daemon is the repo executor. It owns:

- repo-local git/worktrees,
- agent/session process lifecycle (later domain),
- telemetry/observation,
- and execution of control-plane-issued commands.

We must preserve the primary invariant:

- the control plane never assumes repo filesystem access and never runs repo-local actions directly.

Even in the embedded desktop app, the daemon must be reachable only through the same transport abstraction used for remote daemons.

## Goal

Implement the Rust daemon runtime skeleton as a real service with:

- a clean lifecycle (start/shutdown),
- robust outbound connection management to the control plane (handshake, reconnect),
- clear capability reporting,
- and the structural seams needed for later worktree/git/session subsystems.

This ticket is about runtime structure and correctness, not about implementing every command.

## Requirements

### 1) Crate structure

Establish:

- `redesmyn_daemon` (library): runtime core and service wiring.
- `redesmyn-daemon` (binary): thin host for running the daemon headlessly (dev/CI).

Rules:

- `redesmyn_daemon` must not depend on control-plane storage crates.
- The daemon uses the daemon ↔ control plane stream protocol (T-11) via the transport/codec layer (T-7/T-10).

### 2) Lifecycle API

Provide an explicit lifecycle, suitable for embedding:

- `Daemon::start(config, control_plane_stream) -> DaemonHandle`
- `DaemonHandle::shutdown()`

The handle must ensure:

- background tasks are owned and cancelable,
- shutdown is graceful (flush final updates if possible),
- and resources are released deterministically (important for tests).

### 3) Connection management

Implement a connection manager that:

- connects outbound to the control plane stream,
- performs handshake (hello/ack, version validation),
- reconnects with bounded exponential backoff,
- and reports connection state changes as internal events for observability.

Requirements:

- Major protocol mismatch is a hard failure (do not reconnect-loop forever).
- Minor mismatch is accepted (log once, continue).

### 4) Capability model

Define a typed `DaemonCapabilities` model (small, explicit set), e.g.:

- supports_repo_execution
- supports_worktrees
- supports_git_observation
- supports_session_exec
- supports_session_attach (tmux)
- supports_artifacts

The daemon advertises capabilities in hello.

### 5) Repo attachment seam

The daemon must expose a clean internal API for “repo attachment”:

- attach/detach by stable repo identity (workspace_id + repo_id),
- map to local filesystem via the repo registry (T-24),
- and manage per-repo background tasks (observation, command execution).

This ticket only needs the skeleton and state machine; the registry implementation lands in T-24.

### 6) Observability

Use `tracing` conventions (T-4):

- stable span fields: host_id, host_instance_id, workspace_id/repo_id when scoped.
- do not log large payloads by default.

## Acceptance criteria

- A headless daemon binary starts and connects to a control plane stream endpoint (initially via in-proc test harness or a stub server).
- Handshake + reconnect logic is correct and testable (deterministic tests; no sleeps required for correctness).
- Lifecycle is embeddable: desktop app can start/stop the daemon module without reaching into internals.
- Code structure makes it hard to violate the daemon/control-plane boundary.

## Dependencies / sequencing

- Depends on Domain 1 daemon stream contract (T-11) and schema pipeline (T-10).
- Depends on logging conventions (T-4) and config layer (T-5) once implemented.

