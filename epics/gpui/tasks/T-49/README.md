---
epic: gpui
branch:
  suggested: rn/gpui/T-49-command-palette-skeleton
rn:
  parent: T-44
---

# T-49 Command palette skeleton (upgrade; not required for initial port) (Domain 5)

## Problem

We are removing the persistent navigation sidebar.

We still need a fast way to access:

- global actions (theme toggle, refresh),
- navigation-like actions (open epic by slug),
- and developer tools (wiretap, open logs, etc.).

A command palette is a natural “desktop-native” upgrade.

## Goal

Add a minimal command palette skeleton that can evolve over time without blocking the port.

This ticket is an upgrade and is **not required** for the initial “port parity” milestone, but we want it planned and ready.

## Requirements

### 1) Invocation

- Keyboard shortcut (e.g. Cmd/Ctrl+K) and a small UI entrypoint (optional).
- Must be accessible and not interfere with text inputs.

### 2) Command model

Define a typed command registry:

- `id`, `title`, `keywords`, `enabled/disabled reason`,
- action handler (async),
- optional grouping (Navigation / Actions / Developer).

### 3) Minimal initial commands

Include:

- Toggle theme
- Refresh graph (if epic selected)
- Toggle/collapse left session pane
- Open epic… (search/select)

### 4) No silent actions

- Running a command must produce immediate visible feedback.
- Prevent duplicate invocations while a command is in flight.

## Acceptance criteria

- The command palette opens, filters, and runs at least the minimal command set.
- Commands obey the shared in-flight/error conventions (T-44).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on UI foundations (T-44).
