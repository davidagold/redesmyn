## Redesmyn Desktop (GPUI)

### Run

From the repo root:

- `cd rust && cargo run -p redesmyn_desktop`

This uses `rust/rust-toolchain.toml` to pin the toolchain.

### Logs

- `REDESMYN_LOG=info` (or any `tracing_subscriber` filter)
- `REDESMYN_LOG_FORMAT=pretty|json`
- `REDESMYN_LOG_PAYLOADS=off|redact|sample[:N]|full`

Example:

- `REDESMYN_LOG=redesmyn_desktop=debug,redesmyn_control_plane=info cd rust && cargo run -p redesmyn_desktop`

### Config

Config is loaded via `redesmyn_config` (layered defaults → TOML → env overrides).

- `$XDG_CONFIG_HOME/redesmyn/config.toml` (or `~/.config/redesmyn/config.toml`)
- `<repo>/.redesmyn/config.toml`

Rust config lives under `[rust]` in the shared TOML file.

Agent session palette defaults (repo/global config):

- `[ui].show_unreachable_agent_sessions = true` to include sessions that cannot be opened from Cmd-S.

### GPUI pinning

GPUI is pinned in the workspace `rust/Cargo.toml` and locked in `rust/Cargo.lock`.
Update by bumping the `gpui` version in `rust/Cargo.toml`, then regenerating the lockfile.
