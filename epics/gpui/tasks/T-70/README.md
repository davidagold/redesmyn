---
epic: gpui
branch:
  suggested: rn/gpui/T-70-session-viewer-fixture-mode
rn:
  node:
    branch: rn/gpui/T-70-session-viewer-fixture-mode
  parent: T-59
  after:
    - T-15
---

# T-70 Desktop fixture mode: seeded SessionView DB + demo event injection (Domain 7)

## Problem

The GPUI session viewer (T-59) currently requires:

- a valid `SessionId` (manual input), and
- a populated Rust control-plane DB (`session_events`).

In typical dev runs, the Rust DB is empty, which makes it hard to:

- visually verify SessionView behaviors (history load, pagination, live append, “new messages” indicator, scroll anchoring), and
- let AI use T-15-style loops (“do action → wait → check”) while building the session viewer and UI driver surfaces.

We should **not** point the GPUI app at the legacy Python DB (`.redesmyn/redesmyn.sqlite3`); the schema is different and Rust config rejects that path.

## Goal

Add a deterministic **desktop fixture mode** that:

- uses an isolated Rust DB seeded with a realistic session timeline, and
- provides an explicit action that appends a demo event to the seeded session so live subscriptions can be validated on demand.

This enables quick manual visual checks and provides a foundation for later UI-driver tests (T-66).

## Requirements

### 1) Fixture mode toggle

Add a dev-only way to enable fixture mode, e.g.:

- env: `REDESMYN_RUST__DESKTOP__FIXTURE=session_viewer`, or
- CLI flag: `--fixture session_viewer`.

Fixture mode must not affect normal dev runs unless explicitly enabled.

### 2) Isolated fixture DB

When fixture mode is enabled:

- Use a separate state directory / DB path (e.g. under a temp dir or `.redesmyn/fixtures/...`).
- Run Rust migrations as usual.
- Never read/write the legacy Python DB.

### 3) Seed deterministic session data

Seed at least one deterministic `SessionId` with enough events to exercise:

- initial history load (first page),
- backwards pagination (`Load older` multiple times),
- and “at bottom” vs “not at bottom” behavior.

Guidelines:

- Use `scope_kind = 'none'` unless repo/epic/task scope is required for a near-term feature.
- Ensure `created_at_ms` ordering is deterministic and stable across runs.

### 4) Make fixture session discoverable / auto-loadable

Avoid forcing manual copy/paste of session ids in fixture mode.

Options (pick one):

- Auto-fill and auto-load via the existing `REDESMYN_SESSION_VIEWER_SESSION_ID` pathway, or
- Add a fixture-only “Fixture sessions” picker in the sessions pane.

### 5) Demo event injection (action-driven live updates)

Provide a fixture-only, local-only action that appends a new event to the seeded session so `SubscribeSessionEvents` can be validated without sleeps.

Implementation options (pick one):

- SessionView fixture-only button: “Emit demo message” (user/assistant), which calls a fixture helper on the control plane to append a `session_events` row.
- Or, a fixture-only menu item / hotkey that does the same.

### 6) Observability

Add minimal tracing/logging around:

- fixture DB path selection,
- seed completion (session ids + counts),
- demo event emission.

Avoid noisy per-tick logs.

## Acceptance criteria

- With fixture mode enabled, the desktop app launches and `SessionView` shows a non-empty feed immediately.
- `Load older` works at least twice (seeded data exceeds one page).
- Triggering “Emit demo message” appends a new timeline item via the live subscription path.
- No accidental writes to the legacy Python DB.

## Dependencies / sequencing

- Depends on: T-59 (SessionView scaffold + session events history/subscription surfaces).
- Informs/unblocks: T-66 (session viewer AI-testability) by providing a deterministic fixture environment.

