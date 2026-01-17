---
epic: gpui
branch:
  suggested: rn/gpui/T-15-ai-testability
rn:
  parent: T-10
---

# T-15 AI-first testability surfaces (actions + model + GUI) (Domain 1)

## Requirement (new)

The new application must be easy for AI to test:

- AI must be able to **trigger actions** deterministically.
- AI must be able to **assess results**, both:
  - in terms of the underlying objects/state the actions operate on, and
  - in terms of what the GUI is showing.

This is not “nice to have”: it is a core product/development requirement.

## Problem

Even with a clean headless control plane and typed protocols, GUI testing can become:

- flaky (timing, animation, async event ordering),
- opaque (no stable way to assert “what the UI means”),
- and expensive (pixel diffs only).

We also want to avoid repeating the “local mode” mistake:

- tests should not bypass the daemon/control-plane boundary in ways that hide real integration bugs.

## Goal

Define and implement the minimal **automation and introspection surfaces** needed so AI (and CI) can:

1. drive core workflows through the same public contracts as real clients, and
2. assert both semantic UI state and underlying model state without brittle pixel-level tests.

This ticket defines the testability contract and the core protocol surfaces that enable it.

## Design approach

### A) “Everything is scriptable”

- All user actions are available through the headless control plane API (T-12).
- The daemon is reachable only via the control plane (T-11); no direct daemon APIs are required.

### B) “Semantic UI snapshot” is the primary GUI assertion

Provide a machine-readable, stable representation of what the UI is showing.

Pixel screenshots remain useful, but semantic assertions must be first-class.

### C) Determinism + waiting primitives

Testing requires reliable “wait until X is true” primitives to avoid sleeps and race conditions.

## Requirements

### 1) Stable, queryable model state

Ensure the control plane exposes query surfaces sufficient to validate outcomes:

- graph state (tasks, edges/topology, merge readiness, merge runs)
- command lifecycle state (accepted/running/blocked/resumable/succeeded/failed/canceled)
- session event streams (T-14)
- artifact references and metadata (T-14)

These are accessed via the client protocol (T-12), not by poking at internal structs.

### 2) “Wait until” primitives (non-flaky)

Add explicit waiting primitives to the client protocol so tests can block on conditions:

- `WaitForCommand { command_id, terminal_states, timeout }`
- `WaitForEvent { filter, timeout }`
- `WaitForIdle { scope?, timeout }` (meaning: no pending command updates/events for a quiescence window)

Rules:

- Waiting is event-driven, not polling-heavy.
- Timeouts produce structured errors.

### 3) Desktop UI automation driver (semantic, not pixel-only)

Define a local-only “UI driver” interface (transport choice is an implementation detail; UDS is acceptable) that supports:

- driving navigation/actions at a high level (not coordinate clicks), e.g.:
  - open epic
  - select task
  - trigger “merge”
  - open session view
  - open diff view
- querying a **semantic UI snapshot**:
  - current route/view
  - selected epic/task/edge ids
  - visible panels and their primary states (details open, config open, etc.)
  - visible progress indicators (what actions are in flight)
  - key callouts/errors currently displayed

Important:

- The UI driver must be able to assert “no silent actions”: if a command is issued, the UI snapshot must reflect an in-flight indicator until completion.
- UI driver APIs must be stable and use our ULID/newtype identifiers where applicable.

### 4) Screenshot capture (secondary)

Support capturing screenshots for:

- debugging failed tests,
- occasional golden UI regression checks.

Prefer semantic snapshot assertions for most tests.

### 5) Accessibility as a test surface

Even if we use a custom UI driver, the UI must expose:

- stable labels/names for key controls,
- roles where applicable,

so OS-level automation and assistive tech remain viable and we are not locked into a bespoke driver forever.

### 6) Fixture modes

Define test fixture modes that make AI-driven tests cheap to run:

- **Mock daemon mode**: a deterministic daemon simulator that emits telemetry/events from fixtures without touching a real git repo.
- **Real repo mode**: a temporary repo/worktree fixture for “true integration” tests.

Tests should be able to choose between:

- fast deterministic mode (mock daemon) for most UI tests, and
- slower integration mode (real repo) for a smaller set.

## Acceptance criteria

- There is a written contract (proto + docs) for:
  - waiting primitives,
  - semantic UI snapshots,
  - and high-level UI driver actions.
- A test can:
  1) issue a command via the client protocol,
  2) wait for completion via a wait primitive,
  3) query model state to validate the result, and
  4) query UI snapshot to validate the GUI reflects the result.

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-10/README.md` (schema/codegen).
- Builds on:
  - `epics/gpui/tasks/T-11/README.md` (daemon stream contract),
  - `epics/gpui/tasks/T-12/README.md` (client protocol over UDS),
  - `epics/gpui/tasks/T-14/README.md` (session events + artifacts contract).

