use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use redesmyn_git::GitCliBackend;
use redesmyn_ids::{HostId, RepoId, WorkspaceId};
use redesmyn_protocol::RepoScope;
use tempfile::TempDir;
use tokio::sync::watch;

use super::{
    AttachedRepoRoots, FileRepoRegistry, GitRepoIdentityResolver, RepoAttachError,
    RepoAttachmentManager, RepoRegistrationRequest,
};
use crate::host_identity::HostIdentity;

fn run_git(repo_root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .expect("run git command");

    assert!(
        output.status.success(),
        "git command failed: args={args:?}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn init_repo(temp: &TempDir, remote_url: &str) -> std::path::PathBuf {
    let repo_root = temp.path().join("repo");
    std::fs::create_dir_all(&repo_root).expect("create repo root");

    run_git(&repo_root, &["init"]);
    run_git(&repo_root, &["config", "user.email", "test@example.com"]);
    run_git(&repo_root, &["config", "user.name", "Test User"]);
    run_git(&repo_root, &["remote", "add", "origin", remote_url]);

    std::fs::write(repo_root.join("README.md"), "hello\n").expect("write README");
    run_git(&repo_root, &["add", "README.md"]);
    run_git(&repo_root, &["commit", "-m", "initial"]);

    repo_root
}

fn test_scope() -> RepoScope {
    RepoScope::new(WorkspaceId::new(), RepoId::new())
}

fn identity_resolver() -> Arc<GitRepoIdentityResolver> {
    Arc::new(GitRepoIdentityResolver::new(Arc::new(GitCliBackend::new())))
}

#[tokio::test]
async fn register_attach_and_detach_are_idempotent() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo_root = init_repo(&temp, "https://github.com/example/repo-a.git");
    let expected_repo_root = std::fs::canonicalize(&repo_root).expect("canonicalize repo root");

    let scope = test_scope();
    let registry = Arc::new(FileRepoRegistry::new(temp.path().join("registry")));
    let attached_roots = AttachedRepoRoots::default();
    let identity = HostIdentity::new(HostId::new());
    let mut manager = RepoAttachmentManager::new(
        identity,
        registry,
        identity_resolver(),
        attached_roots.clone(),
    );

    manager
        .register_repo(RepoRegistrationRequest {
            scope,
            repo_root: repo_root.clone(),
            display_name: Some("repo-a".to_string()),
            trusted: Some(true),
        })
        .await
        .expect("register repo");

    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    manager
        .attach(scope, shutdown_rx.clone())
        .await
        .expect("first attach");
    manager
        .attach(scope, shutdown_rx)
        .await
        .expect("second attach is idempotent");

    assert_eq!(attached_roots.resolve(scope), Some(expected_repo_root));

    assert!(manager.detach(scope).expect("first detach"));
    assert!(!manager.detach(scope).expect("second detach is idempotent"));

    assert!(attached_roots.resolve(scope).is_none());
}

#[tokio::test]
async fn attach_detects_identity_mismatch_after_registration() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo_root = init_repo(&temp, "https://github.com/example/repo-a.git");

    let scope = test_scope();
    let registry = Arc::new(FileRepoRegistry::new(temp.path().join("registry")));
    let attached_roots = AttachedRepoRoots::default();

    let manager_identity = HostIdentity::new(HostId::new());
    let mut manager = RepoAttachmentManager::new(
        manager_identity,
        registry,
        identity_resolver(),
        attached_roots,
    );

    manager
        .register_repo(RepoRegistrationRequest {
            scope,
            repo_root: repo_root.clone(),
            display_name: None,
            trusted: None,
        })
        .await
        .expect("register repo");

    run_git(
        &repo_root,
        &[
            "remote",
            "set-url",
            "origin",
            "https://github.com/example/repo-b.git",
        ],
    );

    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let err = manager
        .attach(scope, shutdown_rx)
        .await
        .expect_err("attach should fail on identity mismatch");

    match err {
        RepoAttachError::IdentityMismatch { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn second_attach_reports_repo_instance_busy_with_owner_metadata() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo_root = init_repo(&temp, "https://github.com/example/repo-a.git");

    let scope = test_scope();
    let registry = Arc::new(FileRepoRegistry::new(temp.path().join("registry")));

    let attached_roots_a = AttachedRepoRoots::default();
    let mut manager_a = RepoAttachmentManager::new(
        HostIdentity::new(HostId::new()),
        registry.clone(),
        identity_resolver(),
        attached_roots_a,
    );

    manager_a
        .register_repo(RepoRegistrationRequest {
            scope,
            repo_root,
            display_name: None,
            trusted: None,
        })
        .await
        .expect("register repo");

    let (_shutdown_tx_a, shutdown_rx_a) = watch::channel(false);
    manager_a
        .attach(scope, shutdown_rx_a)
        .await
        .expect("attach A");

    let attached_roots_b = AttachedRepoRoots::default();
    let mut manager_b = RepoAttachmentManager::new(
        HostIdentity::new(HostId::new()),
        registry,
        identity_resolver(),
        attached_roots_b,
    );

    let (_shutdown_tx_b, shutdown_rx_b) = watch::channel(false);
    let err = manager_b
        .attach(scope, shutdown_rx_b)
        .await
        .expect_err("attach B should fail while A holds lock");

    match err {
        RepoAttachError::RepoInstanceBusy { lock_path, owner } => {
            assert!(lock_path.ends_with("redesmyn.repo_instance.lock"));
            assert!(owner.is_some(), "expected owner metadata for busy lock");
            let envelope =
                RepoAttachError::RepoInstanceBusy { lock_path, owner }.as_error_envelope();
            let detail = envelope.detail.expect("busy detail");
            assert_eq!(
                detail.get("conflict_code").map(String::as_str),
                Some("repo_instance_busy")
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
