//! Control plane API layer.
//!
//! This crate intentionally does not depend on repo execution crates like `redesmyn_git`.

mod command;
mod task_manager;

pub mod client_api;
pub mod demo;
pub mod error;
pub mod event_log;

mod control_plane;

pub use control_plane::{
    ControlPlane, ControlPlaneDb, ControlPlaneHandle, ControlPlaneInitError, ControlPlaneStartError,
    ControlPlaneStartOptions,
};
