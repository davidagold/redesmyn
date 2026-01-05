# Tests v0 Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **Tests v0** epic: intent, v0 spec, invariants, and key decisions. Keep it current.

## Metadata

```yaml
slug: tests-v0
name: Tests v0
root_branch: main
linear:
  project_id: null
```

## 1) Vision

Build a testing framework that makes Redesmyn safe to evolve quickly:

- A fast, readable **behavior-first** test suite that validates integration/user behavior end-to-end.
- Targeted unit tests for core backend mechanics where they provide leverage.
- Test code that is “AI-friendly”: predictable structure, strong fixtures/scenarios, minimal incidental complexity.

## 2) Requirements (v0)

### 2.1 Test style and organization

- Prefer **module-level** tests to avoid indentation creep.
- Test function names should read like a spec: `test_<expected_behavior>`.
- Use classes only when they provide real organizational benefits (e.g., shared parametrization).
- Prefer fixtures and “scenario” bundles over hand-rolled setup inside tests.

### 2.2 Testing pyramid (v0)

v0 should include coverage at each layer:

1. **Backend integration** (primary): API + DB + projections + daemon protocol (in-process).
2. **Backend unit tests** (selective): mechanics whose correctness is hard to infer from integration tests alone.
3. **Thin Playwright “happy path”** (must-have): one end-to-end UI flow to ensure the system stays wired.

### 2.3 Reliability goals

- Deterministic tests: no reliance on real network services.
- Temporary resources (db files, git repos, ports) should be isolated per test or per scenario.
- Explicit timeouts for anything that can hang.
- Tests should be runnable locally without special setup beyond repo dependencies.

## 3) Proposed stack (v0)

### 3.1 Python backend tests

- `pytest`
- `pytest-asyncio` (async tests)
- `httpx` (ASGI integration client)
- `pytest-timeout` (hang protection)

### 3.2 Git mechanics

- Use a temporary on-disk git repository and invoke real `git` commands (matches production semantics).

### 3.3 UI end-to-end (thin)

- Playwright for a single “happy path” that starts the system and drives the UI.

We intentionally keep the Playwright footprint small in v0; most behavioral coverage should remain in backend integration tests.

Note: this repo uses pytest `--strict-markers`. If we add Playwright tests, they should run under an explicit marker (e.g. `e2e`) and be runnable separately from the default suite.

## 4) What we want to validate (core behaviors)

These are the “must not break” behaviors to cover across tasks:

- **Effective base** selection (e.g., child branch creation ignores merged parents).
- **Merge / restack mechanics**:
  - plan building is correct (spine vs descendants, merge_then_restack)
  - execution updates merge run status/events correctly
  - resume after conflict revalidates and continues safely
- **API behavior**:
  - validation and status transitions are correct
  - error surfaces are consistent (e.g., 409 for running agents, 503 guidance when canonical executor missing)
  - 500s include request ids and are logged
- **Event/WS payload robustness**:
  - event payloads remain JSON-serializable (regression guard for datetime serialization issues)
- **Daemon / WS runtime**:
  - command enqueue/delivery, presence/attachment, event ordering invariants
- **Projections**:
  - graph/projection output reflects repo + events correctly
- **CLI integration**:
  - `rn sync` and `rn shell` behaviors match expectations
  - `rn merge` / `rn restack` plan + confirmation behavior is correct
- **UI wiring**:
  - a minimal end-to-end UI flow succeeds (Playwright happy path)

## 5) Task map

The implementation is split by domain so multiple agents can work independently. Each task should lean heavily on the shared test harness/scenario utilities in T-1.

- `epics/tests-v0/tasks/T-1/README.md`: Test harness + fixtures + scenario bundles (trunk).
- `epics/tests-v0/tasks/T-2/README.md`: DB/migrations + schema invariants.
- `epics/tests-v0/tasks/T-3/README.md`: Git mechanics (unit + integration).
- `epics/tests-v0/tasks/T-4/README.md`: API integration tests.
- `epics/tests-v0/tasks/T-5/README.md`: Daemon/WS protocol + runtime integration tests.
- `epics/tests-v0/tasks/T-6/README.md`: Projections/graph integration tests.
- `epics/tests-v0/tasks/T-7/README.md`: CLI integration tests.
- `epics/tests-v0/tasks/T-8/README.md`: Playwright thin happy path (end-to-end wiring).

## 6) Notes on doc conventions

Many existing task docs in this repo include a “Brief (local)” section because `rn sync` can generate that heading as a stable, human-owned section (never overwritten by sync).

For this epic, **do not treat “Brief” as a brevity constraint**: write as much detail as needed to make tasks implementable without extra context.
