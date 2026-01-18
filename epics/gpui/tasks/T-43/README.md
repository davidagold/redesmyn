---
epic: gpui
branch:
  suggested: rn/gpui/T-43-gpui-desktop-bootstrap
rn:
  parent: T-16
---

# T-43 GPUI desktop app bootstrap + lifecycle (embed control plane + daemon modules) (Domain 5)

## Problem

We are replacing the web UI with a **standalone desktop app** built with **Rust + GPUI**.

This desktop app must:

- embed the **control plane** and (for now) a “modularly embedded” **daemon** in one process,
- while ensuring the application logic **does not assume colocation** (remote daemons later),
- and provide a stable host for the rest of the native UI work (graph, sessions, diff).

If we treat the desktop UI as “the control plane”, we will undermine remote-daemon readiness and AI-first testability.

## Goal

Create a runnable `redesmyn_desktop` GPUI app that:

- starts/stops cleanly,
- embeds the headless control plane service (Domain 2) and a modular daemon (Domain 3) through the same transport seams used for remote,
- and renders an initial UI shell (placeholders are fine; the layout is implemented in later tickets).

## Requirements

### 1) Crate structure and dependency rules

In the Rust workspace:

- `redesmyn_desktop` (binary) is the GPUI host app.
- Prefer a shared UI crate (e.g. `redesmyn_ui`) for reusable views/components, so later UI work can be parallelized without constant merge conflicts.

Boundary rules:

- `redesmyn_desktop` may depend on:
  - control plane client/service crate(s),
  - and UI crates,
  - but must not import daemon “internals” directly (only through the transport trait).

### 2) GPUI integration

- Add GPUI as a dependency in a pinned, reproducible way (document the pin/update strategy).
- The app must run on macOS + Linux (Windows not required).

### 3) Process/lifecycle model

Define an explicit lifecycle:

- `DesktopApp::start() -> DesktopHandle`
- `DesktopHandle::shutdown()`

Shutdown must:

- stop background tasks,
- gracefully stop embedded control plane and daemon modules,
- and flush final logs/events best-effort.

### 4) Embedded services (no colocated assumptions)

Implement embedding through the same seams intended for remote mode:

- Control plane communicates to the daemon via the typed transport trait (T-7/T-11).
- The desktop UI communicates to the control plane via a client interface; in embedded mode this can be in-proc, but it must preserve the same semantics as the client API (T-12).

### 5) Logging + config wiring

- Use the shared typed config layer (T-5) and tracing conventions (T-4).
- Default state dirs must be sensible for macOS + Linux and must not assume repo CWD.

### 6) Minimal UI surface

Render a placeholder root view that proves:

- window creation works,
- and the UI thread is connected to an application state model (even if stubbed).

No design work is required here; that’s in T-44..T-46.

## Acceptance criteria

- `cargo run -p redesmyn_desktop` starts a window and exits cleanly.
- The embedded control plane can be started/stopped from the desktop app without reaching into control plane internals.
- Dependency graph preserves daemon/control-plane separation by construction.
- There is a documented “how to run the desktop app” section (dev flags, logs, config paths).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on control plane service skeleton (T-16).
- Should coordinate with daemon skeleton (T-23) for embedding hooks and transports.

## Reference implementation (today; for behavior orientation only)

- Web UI host (today):
  - `dashboard/src/components/layout/RootLayout.tsx` (layout host; currently includes a sidebar).
  - `dashboard/src/components/layout/Sidebar.tsx` (navigation sidebar we are removing in the desktop port).
- Control plane host (Python today):
  - `redesmyn/api.py` (FastAPI app: REST + WS; serves the dashboard).
  - `redesmyn/cli.py` (process model / dev workflows).

