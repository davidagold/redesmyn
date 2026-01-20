//! Control plane API layer.
//!
//! This crate intentionally does not depend on repo execution crates like `redesmyn_git`.

mod command;
mod control_plane;
mod daemon_link;
mod event_log;
mod task_manager;

pub mod client_api;
pub mod demo;
pub mod error;

pub use control_plane::{
    ControlPlane, ControlPlaneDb, ControlPlaneHandle, ControlPlaneStartError,
    ControlPlaneStartOptions,
};

pub use daemon_link::DaemonLinkHandle;
