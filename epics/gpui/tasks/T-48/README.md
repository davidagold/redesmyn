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

Architecture note: follow the GPUI state architecture policy in `epics/gpui/README.md` §3.7 (Presentation Model; explicit + testable). The UI driver must trigger high-level actions that flow through the same presentation model reducers and command paths as real UI interactions, and semantic snapshots must reflect that explicit state.

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

### 5) Artifacts: semantic snapshots + screenshots (required)

Self-testing must produce **both**:

- **Semantic UI snapshots** (machine-readable; stable) for assertions and debug diffs.
- **Pixel screenshots** (PNG) for human / LLM visual inspection.

On test failure (and optionally at each checkpoint), capture:

- a UI snapshot dump,
- and a screenshot.

Prefer semantic assertions for most tests.

#### Screenshot capture surface

Expose screenshot capture via the UI driver (transport per §1), e.g.:

- `CaptureScreenshot { window?: primary|all, include_decorations?: bool } -> { png_path | png_bytes }`

Notes:

- Prefer writing to a caller-provided artifacts directory (see below) rather than returning large blobs over the driver transport.
- Fixed window size is required for stability (see Determinism).

#### Artifacts convention

Define and document a stable artifacts convention so CI and agents can always find outputs.

Minimum:

- Support a single `REDESMYN_TEST_ARTIFACTS_DIR` (or equivalent) to route outputs.
- For each checkpoint, write:
  - `ui_snapshot_<label>.json`
  - `screenshot_<label>.png`

The convention must be deterministic and collision-resistant across parallel test runs.

#### Determinism requirements (for stable screenshots)

To keep screenshots useful and non-flaky, fixture/test mode must enforce:

- deterministic window size (config-driven),
- deterministic theme (e.g. dark/light pinned; no system theme),
- no time-based animations affecting visual diffs (disable or freeze where feasible),
- and stable text rendering inputs (font selection pinned where possible).

### 6) Documentation (required)

Add a short developer-facing guide that explains:

- how to run UI-driver tests locally,
- what artifacts are produced and where,
- and how to reproduce a failing run from artifacts.

Guide: `epics/gpui/tasks/T-48/DEV_GUIDE.md`.

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
