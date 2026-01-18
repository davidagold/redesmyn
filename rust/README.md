# Rust workspace (GPUI epic)

This repository keeps the legacy Python/TS codebase in place while we build a Rust + GPUI implementation under `rust/`.

## Quick start

- `just rust::check`
- `just rust::test`
- `just rust::fmt`
- `just rust::clippy`

Or directly:

- `cd rust && cargo check`

## `rn-rs` (Rust CLI)

During the port, the existing Python CLI remains `rn`. The Rust port’s CLI binary is named `rn-rs` so both can coexist on `PATH`.

Examples:

- `cd rust && cargo run -p rn -- --help`
- `cd rust && cargo run -p rn -- doctor`
- `cd rust && cargo run -p rn -- version`
- `cd rust && cargo run -p rn -- bench startup`

Conventions:

- Structured output: `--output human|json` (default: `human`).
- Exit codes (aligned with the shared T-3 error categories):
  - `0`: success
  - `1`: internal/unexpected
  - `2`: invalid request / CLI usage
  - `3`: not found
  - `4`: conflict
  - `5`: unauthorized
  - `6`: unavailable
  - `130`: interrupted (Ctrl-C)

Install to `PATH`:

- `cd rust && cargo install --path crates/rn --bin rn-rs`

## Crate boundaries (by construction)

Dependency rules (enforced by `Cargo.toml` edges):

- The control plane (`redesmyn_control_plane`) does **not** depend on repo/git execution crates (e.g. `redesmyn_git`).
- The daemon (`redesmyn_daemon`) does **not** depend on control-plane storage crates (e.g. `redesmyn_storage`).
- `redesmyn_domain` and `redesmyn_protocol` are dependency roots for most crates.

Crates live under `rust/crates/*` and are intentionally small to minimize merge conflicts across parallel Domain 0 tasks.
