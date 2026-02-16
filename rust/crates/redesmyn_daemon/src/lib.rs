//! Daemon runtime: service lifecycle, control-plane connection management, repo attachment.
//!
//! This crate intentionally does not depend on control-plane storage crates like `redesmyn_storage`.

#![forbid(unsafe_code)]

mod agent_driver;
mod backoff;
mod command_router;
mod control_plane;
mod host_identity;
mod merge_planner;
mod repo;
mod runtime;
mod state_dir;

pub use crate::control_plane::{ConnectionState, ControlPlaneConnector};
pub use crate::host_identity::HostIdentity;
pub use crate::repo::{
    AttachedRepoRoots, FileRepoRegistry, RepoAttachError, RepoDetachError, RepoRegisterError,
    RepoRegistrationRequest, RepoRegistry, RepoRegistryError,
};
pub use crate::merge_planner::{
    MergePlan, MergePlanRequest, MergeRestackCommandKind, MergeRestackPlanner, PlanBlockedError,
    PlanBlocker, PlanBlockerCode, PlanError, PlanPolicyInput, PlanPolicyMetadata, PlanScope,
    PlanStep, PlanStepKind, PlannerContext, PlannerTaskGraph, PlannerTaskNode, RepoInstanceStatus,
    RepoPrimaryStatus, RestackPlan, RestackPlanRequest, WorktreeHealth,
};
pub use crate::runtime::{Daemon, DaemonHandle, DaemonRuntimeConfig};
pub use redesmyn_protocol::daemon::DaemonCapabilities;
