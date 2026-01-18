# Rust workspace (GPUI epic)

This repository keeps the legacy Python/TS codebase in place while we build a Rust + GPUI implementation under `rust/`.

## Quick start

- `just rust::check`
- `just rust::test`
- `just rust::fmt`
- `just rust::clippy`

Or directly:

- `cd rust && cargo check`

## Crate boundaries (by construction)

Dependency rules (enforced by `Cargo.toml` edges):

- The control plane (`redesmyn_control_plane`) does **not** depend on repo/git execution crates (e.g. `redesmyn_git`).
- The daemon (`redesmyn_daemon`) does **not** depend on control-plane storage crates (e.g. `redesmyn_storage`).
- `redesmyn_domain` and `redesmyn_protocol` are dependency roots for most crates.

Crates live under `rust/crates/*` and are intentionally small to minimize merge conflicts across parallel Domain 0 tasks.
