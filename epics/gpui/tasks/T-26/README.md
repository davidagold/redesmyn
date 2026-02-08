---
epic: gpui
branch:
  suggested: rn/gpui/T-26-git-backend-abstraction
rn:
  node:
    branch: rn/gpui/T-26-git-backend-abstraction
  parent: T-23
---

# T-26 Git backend abstraction (CLI-first now, swappable later) (Domain 3)

## Problem

Git operations are the substrate of Redesmyn. We need:

- correctness and compatibility with real-world repos,
- good performance (“zippy” CLI and smooth UI),
- and a maintainable abstraction that can evolve.

There are multiple implementation options:

- shelling out to `git` (max compatibility, straightforward, good enough perf for many tasks),
- using a Rust git library (potentially faster, more structured, but higher risk/complexity).

If we choose too early without a seam, we risk:

- lock-in to a poor fit,
- or over-engineering.

## Goal

Define a `GitBackend` abstraction and implement a default backend that shells out to `git` CLI.

Design explicitly so we can later swap to a Rust-native backend (e.g., gitoxide) without rewriting higher-level logic.

## Requirements

### 1) Backend trait

Define a minimal, composable trait surface that supports:

- ref queries (resolve branch/head sha)
- commit graph queries needed for projections (merge base, ancestry checks)
- worktree operations (delegated to T-27, but the backend supports required primitives)
- safe execution helpers (command invocation, timeouts, cancellation best-effort)

### 2) CLI backend implementation

Implement `GitCliBackend`:

- uses subprocess invocation with explicit working directory,
- captures stdout/stderr,
- maps errors into typed error categories (T-3),
- supports cancellation best-effort (signal/kill) for long operations.

### 3) Output parsing strategy

Prefer stable, machine-readable git output forms:

- `--porcelain`, `--format=...`, etc.

No fragile parsing of human-oriented output.

### 4) Performance hygiene

- Avoid calling git repeatedly in tight loops; support batched queries where feasible.
- Cache read-only derived data within a single observation tick (T-28) when safe.

### 5) Testability

- Provide tests using temporary git repos created during tests.
- Tests validate correctness of key primitives (merge base, is-ancestor, ref resolution).

## Acceptance criteria

- A stable `GitBackend` abstraction exists and higher-level code depends on it, not on ad-hoc subprocess calls.
- The default CLI backend is correct, well-typed, and test-covered.
- The design clearly documents what would be required to implement a Rust-native backend later.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on daemon skeleton (T-23) only for wiring; can be developed largely independently.
- Used by worktree management (T-27), observation/projections (T-28), and merge/restack (T-29/T-30).

## Reference implementation (today; git execution orientation only)

- Git subprocess wrappers (Python today):
  - `redesmyn/git_subprocess.py` (invocation helpers).
  - `redesmyn/repo.py` (git helpers like ancestry checks, in-progress op detection, worktree helpers).
  - `redesmyn/git_proxy.py` (git proxying rules used by `rn git …`).
- Higher-level git mechanics (Python today):
  - `redesmyn/git_mechanics_v0.py` (merge/restack planning + execution; consumes repo/worktree helpers).
- Tests (Python today):
  - `tests/test_git_mechanics_planning.py`
  - `tests/test_git_mechanics_execution.py`
  - `tests/scenarios/scenario.py` and `tests/scenarios/seeds/git.py` (real git repo/worktree fixtures).
