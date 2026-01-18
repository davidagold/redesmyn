---
epic: gpui
branch:
  suggested: rn/gpui/T-48-gpui-ui-driver
rn:
  parent: T-45
  after:
    - T-15
---

# T-48 Desktop UI driver + semantic UI snapshot (AI-first testability; local-only) (Domain 5)

## Problem

The app must be easy for AI to test:

- AI must be able to trigger actions deterministically.
- AI must be able to assert both underlying state and what the GUI is showing.

GPUI does not come with the same browser automation affordances as the web dashboard, so we must build explicit automation surfaces early.

## Goal

Implement the desktop **UI driver** and **semantic UI snapshot** surfaces (as specified in T-15) for the GPUI desktop app.

This is local-only and intended for:

- CI,
- AI-driven development,
- and debugging.

## Requirements

### 1) Driver transport

Expose a local-only automation surface:

- UDS is acceptable, or an in-proc test harness API if it provides equivalent capabilities.

Security model:

- local-only; rely on filesystem permissions for v0.

### 2) High-level actions (no coordinate clicks)

Support actions like:

- open/select epic
- collapse/expand left session pane
- create chat session
- close chat session
- pin/unpin chat session to current epic
- trigger refresh
- (later) select task in graph (stub acceptable until Domain 6)

Actions must map to the same underlying command/query flows as real UI actions.

### 3) Semantic UI snapshot

Provide a stable snapshot representation containing (at minimum):

- selected epic id/slug (or null)
- left pane: visible/collapsed + width
- current primary view (graph index vs epic workspace)
- visible in-flight actions (to enforce “no silent actions”)
- visible error callouts (actionable text)

### 4) Deterministic waiting

Integrate with waiting primitives (T-15) so tests can:

- wait for “idle”,
- wait for command completion,
- wait for a UI snapshot predicate.

### 5) Failure artifacts (optional but recommended)

On test failure, allow capturing:

- a UI snapshot dump,
- and (optionally) a screenshot.

Prefer semantic assertions for most tests.

## Acceptance criteria

- A test can:
  1) launch the desktop app,
  2) select an epic,
  3) trigger refresh,
  4) query UI snapshot and assert in-flight → complete transitions.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on the testability contract (T-15).
- Depends on the desktop layout shell (T-45) for stable UI primitives to drive.

## Reference implementation (today; for behavior orientation only)

- Web e2e testing today:
  - `tests/e2e/test_ui_happy_path.py` (Playwright).
