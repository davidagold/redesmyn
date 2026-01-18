---
epic: gpui
branch:
  suggested: rn/gpui/T-1-rust-workspace-foundations
rn:
  parent: null
---

# T-1 Rust workspace bootstrap + crate boundaries (Domain 0)

## Context

We are porting Redesmyn to **Rust + GPUI** while preserving the **daemon ↔ control-plane separation** as the primary architectural boundary.

During the port, we will keep a split codebase (existing Python/TS stays in place) so the Rust implementation can be developed incrementally and validated against current behavior.

The base branch for this epic is `gpui` so work can land incrementally without disturbing `main`.

## Goal

Establish a Rust workspace skeleton that:

- maximizes safe parallelism (many agents can work in different crates with minimal merge conflicts),
- enforces the control-plane/daemon boundary by construction (dependency rules),
- is idiomatic Rust (small crates, explicit interfaces),
- sets us up for long-term performance (typed IDs, protocol-first design).

## Proposed workspace layout

Create a Rust workspace under `rust/` with crates in `rust/crates/`.

We use Rust **2024 edition** unless it introduces GPUI/tooling issues; if it does, fall back to 2021 with a short note explaining why.

### Crate map (initial cut)

These crates are intentionally small and boundary-oriented; they are not a “final architecture”, but they are designed to minimize coupling and enable parallel work.

#### `redesmyn_ids`

- ULID-backed newtypes for identifiers (e.g., `RepoId`, `TaskId`, `HostId`, `RunId`, etc.).
- Serde + `sqlx` integration.
- No I/O.

#### `redesmyn_domain`

- Core domain models + enums + invariants (tasks, graph topology, statuses, desired state).
- Pure logic only: no DB, no git, no network.

#### `redesmyn_protocol`

- Versioned, strongly-typed message schemas for:
  - daemon ↔ control plane
  - UI/CLI ↔ control plane
- Message envelopes: idempotency keys, timestamps, resync semantics.

#### `redesmyn_storage`

- Control-plane persistence with `sqlx`: event log + projections + queries.
- Depends on `redesmyn_domain` and `redesmyn_protocol`.
- Must not depend on git/daemon crates.

#### `redesmyn_git`

- Repo-executor primitives: worktrees, safe git operations, planning/execution, observation inputs.
- Depends on `redesmyn_domain` + `redesmyn_protocol`.
- Must not depend on `redesmyn_storage`.

#### `redesmyn_exec`

- Daemon-side primitives: process/session lifecycle, exec streaming, artifacts.
- Depends on `redesmyn_domain` + `redesmyn_protocol`.

#### `redesmyn_logging`

- `tracing` + `tracing_subscriber` setup and shared span conventions.
- Shared logging policy (metadata vs payload sampling/redaction).
- Used by all binaries and most libraries.

#### `redesmyn_config`

- Typed config structs and layered config loading (defaults → TOML → env).
- Clear separation between control-plane settings, daemon settings, and desktop settings.

#### `redesmyn_transport` (+ optional `redesmyn_codec`)

- Typed transport traits used by control plane to communicate with daemons.
- In-proc transport for desktop embedding.
- Optional split:
  - `redesmyn_transport`: traits + in-proc channel transport
  - `redesmyn_codec`: framing + protobuf/json codecs for network transports

#### `redesmyn_control_plane` (lib) + `redesmyn-server` (bin)

- Control plane API layer: serves the desktop UI + `rn`.
- Routes commands to daemon via a **transport trait** (no daemon internals).
- Must not depend on `redesmyn_git`.

#### `redesmyn_daemon` (lib) + `redesmyn-daemon` (bin)

- Daemon runtime: repo attachment, leases, telemetry loop, command execution.
- Must not depend on `redesmyn_storage`.

#### `rn` (bin)

- Fast CLI that talks to the control plane.
- Shares types with the rest of the workspace.

#### `redesmyn_desktop` (bin)

- GPUI desktop app embedding:
  - control plane, and
  - a modular daemon
- Embedding is through the same transport interface used for remote daemons; no special-case “if embedded then call daemon functions”.

## Requirements

### 1) Workspace scaffolding

- Add `rust/Cargo.toml` workspace and `rust/crates/*` crate skeletons with minimal `lib.rs`/`main.rs`.
- Add a pinned `rust-toolchain.toml` (stable toolchain version pinned explicitly).
- Add minimal docs describing how to build/check Rust workspace.

### 2) Dependency boundary enforcement

Enforce these rules by crate dependencies (and add a short doc explaining the intent):

- Control plane does not import git/worktree logic.
- Daemon does not import control-plane storage logic.
- Domain and protocol crates are dependency roots for most of the system.

### 3) Build/lint ergonomics

- Add `just` commands (or equivalent) that do not disrupt existing flows:
  - `just rust::check`
  - `just rust::test`
  - `just rust::fmt`
  - `just rust::clippy`

### 4) Split-codebase friendliness

- Do not delete or restructure the existing Python/TS code.
- Keep Rust additions contained to `rust/` and docs/epics updates.

## Acceptance criteria

- `cd rust && cargo check` succeeds on macOS.
- Workspace builds without requiring GPUI code yet (desktop crate can be stubbed if needed).
- Crate boundaries exist and the dependency graph makes the daemon/control-plane split explicit and hard to violate.
- A short “how to work in this workspace” note exists (paths, commands, boundary rules).

## Notes / decisions (resolved here)

- Rust edition: **2024** (fallback only if GPUI/tooling forces it).
- Workspace structure: `rust/` + `rust/crates/`.
- ID strategy is ULID/newtypes everywhere (implemented in T-2).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- This ticket is a prerequisite for all other Domain 0 tickets.
- After this lands, Domain 0 tickets are intended to be runnable in parallel (one task per branch/worktree) with minimal merge conflicts by working primarily within their dedicated crates.

## Reference implementation (today; for behavior orientation only)

- Repo layout (today):
  - `redesmyn/` (Python control plane + daemon)
  - `dashboard/` (React web UI)
  - `openapi/openapi.json` (OpenAPI surface consumed by the dashboard)
  - `tests/` (integration + e2e coverage)
- Developer workflows (today):
  - `justfile` (entrypoints such as `just dev`)
  - `redesmyn/cli.py` (Python `rn` commands)
