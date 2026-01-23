//! Repo-executor primitives: worktrees and safe git operations.
//!
//! This crate defines a small, semantic [`GitBackend`] trait used by daemon-side
//! subsystems (worktrees, observation, merge/restack) so higher-level logic does
//! not depend on ad-hoc subprocess invocation.
//!
//! The default implementation is [`GitCliBackend`], which shells out to the `git`
//! CLI for maximum real-world compatibility.

#![forbid(unsafe_code)]

mod backend;
mod cli;
mod error;
mod types;

pub use backend::{GitBackend, GitRunOptions, GitWorktreeAddOptions};
pub use cli::GitCliBackend;
pub use error::GitError;
pub use types::{
    GitOid, GitRefName, GitRevision, GitWorktree, GitWorktreeTarget, ParseGitOidError,
};

use std::{future::Future, pin::Pin};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
