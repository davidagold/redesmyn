use std::path::Path;

use redesmyn_git::{GitBackend, GitCliBackend, GitRevision, GitRunOptions, GitWorktreeTarget};
use tempfile::TempDir;

fn git(repo_root: &Path, args: &[&str]) -> std::process::Output {
    let output = std::process::Command::new("git")
        .arg("--no-pager")
        .arg("-c")
        .arg("color.ui=false")
        .args(args)
        .current_dir(repo_root)
        .output()
        .expect("git subprocess should run");

    if !output.status.success() {
        panic!(
            "git {:?} failed: status={:?} stdout={} stderr={}",
            args,
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    output
}

fn git_stdout(repo_root: &Path, args: &[&str]) -> String {
    String::from_utf8(git(repo_root, args).stdout)
        .expect("stdout utf-8")
        .trim()
        .to_owned()
}

fn setup_repo() -> (TempDir, String) {
    let dir = TempDir::new().expect("tempdir");
    let repo_root = dir.path();

    git(repo_root, &["init"]);
    // Avoid relying on global user config in CI.
    git(repo_root, &["config", "user.email", "test@example.com"]);
    git(repo_root, &["config", "user.name", "Redesmyn Test"]);

    // Ensure the branch name is deterministic even if git defaults to `master`.
    git(repo_root, &["checkout", "-b", "main"]);

    std::fs::write(repo_root.join("a.txt"), "a\n").expect("write");
    git(repo_root, &["add", "a.txt"]);
    git(repo_root, &["commit", "-m", "base"]);
    let base = git_stdout(repo_root, &["rev-parse", "HEAD"]);

    (dir, base)
}

#[tokio::test]
async fn cli_backend_resolve_merge_base_and_ancestry() {
    let (dir, base) = setup_repo();
    let repo_root = dir.path();

    git(repo_root, &["checkout", "-b", "feature"]);
    std::fs::write(repo_root.join("f.txt"), "f\n").expect("write");
    git(repo_root, &["add", "f.txt"]);
    git(repo_root, &["commit", "-m", "feature"]);

    git(repo_root, &["checkout", "main"]);
    std::fs::write(repo_root.join("m.txt"), "m\n").expect("write");
    git(repo_root, &["add", "m.txt"]);
    git(repo_root, &["commit", "-m", "main"]);

    let backend = GitCliBackend::new();
    let main = backend
        .resolve_commit(
            repo_root,
            &GitRevision::new("main").unwrap(),
            GitRunOptions::default(),
        )
        .await
        .unwrap();
    let feature = backend
        .resolve_commit(
            repo_root,
            &GitRevision::new("feature").unwrap(),
            GitRunOptions::default(),
        )
        .await
        .unwrap();

    let merge_base = backend
        .merge_base(repo_root, &main, &feature, GitRunOptions::default())
        .await
        .unwrap()
        .expect("merge base");
    assert_eq!(merge_base.as_str(), base);

    assert!(
        backend
            .is_ancestor(repo_root, &merge_base, &main, GitRunOptions::default())
            .await
            .unwrap()
    );
    assert!(
        backend
            .is_ancestor(repo_root, &merge_base, &feature, GitRunOptions::default())
            .await
            .unwrap()
    );
    assert!(
        !backend
            .is_ancestor(repo_root, &main, &feature, GitRunOptions::default())
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn cli_backend_batch_resolve_handles_missing() {
    let (dir, _base) = setup_repo();
    let repo_root = dir.path();

    git(repo_root, &["checkout", "-b", "feature"]);
    std::fs::write(repo_root.join("f.txt"), "f\n").expect("write");
    git(repo_root, &["add", "f.txt"]);
    git(repo_root, &["commit", "-m", "feature"]);
    git(repo_root, &["checkout", "main"]);

    let backend = GitCliBackend::new();
    let revs = vec![
        GitRevision::new("main").unwrap(),
        GitRevision::new("feature").unwrap(),
        GitRevision::new("no_such_branch").unwrap(),
    ];

    let resolved = backend
        .resolve_commits(repo_root, &revs, GitRunOptions::default())
        .await
        .unwrap();

    assert_eq!(resolved.len(), 3);
    assert!(resolved[0].is_some());
    assert!(resolved[1].is_some());
    assert!(resolved[2].is_none());
}

#[tokio::test]
async fn cli_backend_cancellation_short_circuits() {
    let (dir, _base) = setup_repo();
    let repo_root = dir.path();

    let backend = GitCliBackend::new();
    let (_tx, rx) = tokio::sync::watch::channel(true);

    let err = backend
        .resolve_commit(
            repo_root,
            &GitRevision::new("HEAD").unwrap(),
            GitRunOptions::default().with_cancel(rx),
        )
        .await
        .expect_err("should cancel");

    assert!(matches!(err, redesmyn_git::GitError::Cancelled { .. }));
}

#[tokio::test]
async fn cli_backend_worktree_list_parses_porcelain() {
    let (dir, _base) = setup_repo();
    let repo_root = dir.path();

    let backend = GitCliBackend::new();
    let worktrees = backend
        .worktree_list(repo_root, GitRunOptions::default())
        .await
        .unwrap();

    assert!(!worktrees.is_empty());
    assert!(worktrees.iter().any(|wt| wt.path == repo_root));

    // Ensure the public enum is usable (T-27 uses this for worktree creation).
    let _ = GitWorktreeTarget::Head;
}
