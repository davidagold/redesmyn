# Rust error/result conventions

This workspace uses a small, stable error taxonomy so errors can cross boundaries (daemon ↔ control plane ↔ clients) and still be actionable for humans and machines.

## Taxonomy

Use `redesmyn_protocol::ErrorCategory` everywhere for cross-boundary categorization:

- `invalid_request` → bad input / misuse
- `not_found` → missing resource
- `conflict` → state conflict / precondition failure
- `unauthorized` → missing/invalid auth
- `unavailable` → dependency down / retryable
- `internal` → unexpected bug / invariant violation

## Library crates

- Library crates (`redesmyn_*`) return typed `Result<T, E>` where `E` is a `thiserror` enum.
- Do not use `panic!` for normal control flow.

## Protocol/API mapping

Cross-boundary failures should be converted into `redesmyn_protocol::ErrorEnvelope`:

- `category`: stable enum (`ErrorCategory`)
- `message`: user-actionable and non-noisy (no stack traces)
- `detail`: optional small structured info for debugging/UX

The stable HTTP mapping is `ErrorEnvelope::http_status()`.

## CLI/binary mapping

Cross-boundary errors map to stable exit codes via `ErrorEnvelope::exit_code()`:

- `internal` → 1
- `invalid_request` → 2
- `not_found` → 3
- `conflict` → 4
- `unauthorized` → 5
- `unavailable` → 6

## Example (T-3)

- Library error: `redesmyn_control_plane::error::ControlPlaneError`
- Protocol envelope: `redesmyn_protocol::ErrorEnvelope`
- Binary rendering + exit code: `redesmyn-server demo-error T-404`

