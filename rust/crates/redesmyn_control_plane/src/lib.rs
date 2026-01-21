//! Control plane API layer.
//!
//! This crate intentionally does not depend on repo execution crates like `redesmyn_git`.

pub mod client_api;
pub mod control_plane;
pub mod demo;
pub mod error;
pub mod event_log;

pub use control_plane::ControlPlane;
