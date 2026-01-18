---
epic: gpui
branch:
  suggested: rn/gpui/T-39-daemon-app-server-agent
rn:
  parent: T-35
---

# T-39 Daemon app-server agent runtime skeleton (Domain 4)

## Problem

We expect to extend beyond “shell” and “exec” agents into **app-server-style agents** that:

- run as a long-lived server,
- expose a typed control surface (requests/responses),
- and stream structured session events (tools, messages, artifacts).

If we do not establish a seam for this early, we will contort the exec-runtime into something it isn’t and make later integration costly.

## Goal

Create the daemon-side **app-server agent runtime skeleton**:

- typed interfaces,
- lifecycle management,
- and event bridging into `SessionEvent` (T-14),

without fully implementing a specific external protocol yet.

## Requirements

### 1) Agent runtime taxonomy

Introduce an explicit runtime kind (names illustrative):

- `AgentRuntimeKind::ShellTmux`
- `AgentRuntimeKind::Exec`
- `AgentRuntimeKind::AppServer`

This is separate from agent brand/kind (Codex/Claude/etc).

### 2) Minimal skeleton capabilities

Define the minimal interfaces needed for future work:

- start/shutdown of the app-server process,
- connect/reconnect semantics (if the server is contacted over a socket),
- request/response dispatch for “send message” and “interrupt” equivalents,
- stream of structured events for:
  - messages,
  - tool invocations/results,
  - artifacts.

### 3) Transport constraints

- Keep all repo-local filesystem knowledge inside the daemon.
- Ensure the control plane sees only stable ids and typed protocol messages.

### 4) Testability

- Provide a deterministic “fake app server agent” fixture interface for tests (no network flakiness).

## Acceptance criteria

- There is a compile-time-checked skeleton with clear interfaces and lifecycle.
- A deterministic test can:
  - start the skeleton with a fake server implementation,
  - exchange one request/response,
  - and receive one structured event.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Builds on the exec-session supervisor conventions (T-35) for lifecycle + backpressure patterns.
- Uses session event and artifact contracts (T-14).

## Reference implementation (today; for motivation orientation only)

- Motivation (external; not implemented in Redesmyn today):
  - `codex-rs/app-server` (linked by project notes as the direction for “app-server” agents).

