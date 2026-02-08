//! Typed config layer (GPUI epic, Domain 0).
//!
//! This crate implements a coherent configuration story for the Rust workspace:
//!
//! - **Typed** config models (control plane / daemon / desktop).
//! - **Layered** loading with stable precedence:
//!   1. compiled defaults
//!   2. config file(s) (TOML)
//!   3. environment overrides
//! - **Early validation** with actionable error messages.
//!
//! ## Split-codebase (legacy Python) coexistence
//!
//! The existing Python implementation uses:
//!
//! - `.env` files and flat env vars like `REDESMYN_DB_PATH`
//! - TOML config at:
//!   - `$XDG_CONFIG_HOME/redesmyn/config.toml` (or `~/.config/redesmyn/config.toml`)
//!   - `<repo>/.redesmyn/config.toml`
//!
//! To avoid key collisions, **Rust config lives under a `[rust]` namespace** in the
//! same `config.toml` files. The Python loaders ignore unknown keys, so both worlds
//! can share a single TOML file during the port.
//!
//! ## Environment variable naming
//!
//! Env overrides use the prefix `REDESMYN_` and `__` as a nesting separator:
//!
//! - `REDESMYN_RUST__CONTROL_PLANE__API__BIND=127.0.0.1:9234`
//! - `REDESMYN_RUST__PROFILE=dev`
//!
//! ## Example `config.toml`
//!
//! ```toml
//! [rust]
//! profile = "dev"
//!
//! [rust.control_plane.api]
//! bind = "127.0.0.1:9234"
//!
//! [rust.control_plane.auth]
//! daemon_token = "dev"
//! ```

mod error;
mod load;
mod model;
mod paths;
mod secret;

pub use crate::error::{LoadConfigError, ValidationError};
pub use crate::load::{ConfigFiles, DotenvMode, LoadConfigOptions, load_rust_config};
pub use crate::model::{
    ConfigProfile, ControlPlaneApiConfig, ControlPlaneAuthConfig, ControlPlaneConfig,
    ControlPlaneDbConfig, DaemonConfig, DesktopConfig, DesktopFixtureMode, ExecutorConfig,
    RustConfig, SandboxConfig, SandboxNetworkMode, SandboxType, WindowConfig,
};
pub use crate::paths::{
    discover_repo_root_from, global_config_path, legacy_db_path, repo_config_path, repo_state_dir,
    rust_db_path,
};
pub use crate::secret::SecretString;
