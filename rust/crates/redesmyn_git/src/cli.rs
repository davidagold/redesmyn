use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use redesmyn_logging::tracing;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::backend::{GitRunOptions, GitWorktreeAddOptions};
use crate::error::GitError;
use crate::types::{GitOid, GitRefName, GitRevision, GitWorktree, GitWorktreeTarget};
use crate::{BoxFuture, GitBackend};

#[derive(Debug, Clone)]
pub struct GitCliBackend {
    git_exe: PathBuf,
}

impl Default for GitCliBackend {
    fn default() -> Self {
        Self {
            git_exe: PathBuf::from("git"),
        }
    }
}

impl GitCliBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_git_exe(mut self, git_exe: impl Into<PathBuf>) -> Self {
        self.git_exe = git_exe.into();
        self
    }

    async fn run_git(
        &self,
        op: &'static str,
        repo_root: &Path,
        args: &[&str],
        stdin: Option<&[u8]>,
        options: GitRunOptions,
    ) -> Result<GitCommandOutput, GitError> {
        let mut cancel = options.cancel;
        if let Some(rx) = cancel.as_ref() {
            if *rx.borrow() {
                return Err(GitError::Cancelled { op });
            }
        }

        let span = tracing::debug_span!(
            "git.cli",
            op,
            repo_root = %repo_root.display(),
        );
        let _guard = span.enter();

        let start = Instant::now();

        let mut cmd = Command::new(&self.git_exe);
        cmd.arg("--no-pager");
        cmd.arg("-c");
        cmd.arg("color.ui=false");
        cmd.env("GIT_TERMINAL_PROMPT", "0");
        cmd.current_dir(repo_root);
        cmd.stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true);
        cmd.args(args);

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(GitError::GitNotFound);
            }
            Err(err) => return Err(GitError::SpawnGit { source: err }),
        };

        if let Some(stdin_bytes) = stdin {
            let mut child_stdin = child.stdin.take().ok_or_else(|| GitError::Io {
                source: std::io::Error::new(std::io::ErrorKind::Other, "missing stdin pipe"),
            })?;
            child_stdin
                .write_all(stdin_bytes)
                .await
                .map_err(|err| GitError::Io { source: err })?;
        }

        let stdout = child.stdout.take().ok_or_else(|| GitError::Io {
            source: std::io::Error::new(std::io::ErrorKind::Other, "missing stdout pipe"),
        })?;
        let stderr = child.stderr.take().ok_or_else(|| GitError::Io {
            source: std::io::Error::new(std::io::ErrorKind::Other, "missing stderr pipe"),
        })?;

        let stdout_task = tokio::spawn(read_to_end(stdout));
        let stderr_task = tokio::spawn(read_to_end(stderr));

        let outcome = match (cancel.as_mut(), options.timeout) {
            (Some(cancel_rx), Some(timeout)) => tokio::select! {
                status = child.wait() => WaitOutcome::Exited(status.map_err(|err| GitError::Io { source: err })?),
                _ = wait_for_cancel(cancel_rx) => WaitOutcome::Cancelled,
                _ = tokio::time::sleep(timeout) => WaitOutcome::TimedOut(timeout),
            },
            (Some(cancel_rx), None) => tokio::select! {
                status = child.wait() => WaitOutcome::Exited(status.map_err(|err| GitError::Io { source: err })?),
                _ = wait_for_cancel(cancel_rx) => WaitOutcome::Cancelled,
            },
            (None, Some(timeout)) => tokio::select! {
                status = child.wait() => WaitOutcome::Exited(status.map_err(|err| GitError::Io { source: err })?),
                _ = tokio::time::sleep(timeout) => WaitOutcome::TimedOut(timeout),
            },
            (None, None) => WaitOutcome::Exited(
                child
                    .wait()
                    .await
                    .map_err(|err| GitError::Io { source: err })?,
            ),
        };

        match &outcome {
            WaitOutcome::Exited(_) => {}
            WaitOutcome::Cancelled | WaitOutcome::TimedOut(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
        };

        let stdout = stdout_task
            .await
            .map_err(|err| GitError::Io {
                source: std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("stdout join failed: {err}"),
                ),
            })?
            .map_err(|err| GitError::Io { source: err })?;
        let stderr = stderr_task
            .await
            .map_err(|err| GitError::Io {
                source: std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("stderr join failed: {err}"),
                ),
            })?
            .map_err(|err| GitError::Io { source: err })?;

        match outcome {
            WaitOutcome::Exited(status) => {
                let elapsed = start.elapsed();
                if !status.success() {
                    tracing::warn!(
                        exit_code = ?status.code(),
                        elapsed_ms = elapsed.as_millis(),
                        stderr_len = stderr.len(),
                        "git command failed"
                    );
                }
                Ok(GitCommandOutput {
                    status,
                    stdout,
                    stderr,
                })
            }
            WaitOutcome::Cancelled => {
                tracing::warn!(
                    elapsed_ms = start.elapsed().as_millis(),
                    "git command cancelled"
                );
                Err(GitError::Cancelled { op })
            }
            WaitOutcome::TimedOut(timeout) => {
                tracing::warn!(
                    timeout_ms = timeout.as_millis(),
                    elapsed_ms = start.elapsed().as_millis(),
                    "git command timed out"
                );
                Err(GitError::Timeout { op, timeout })
            }
        }
    }
}

enum WaitOutcome {
    Exited(std::process::ExitStatus),
    Cancelled,
    TimedOut(Duration),
}

async fn wait_for_cancel(cancel_rx: &mut tokio::sync::watch::Receiver<bool>) {
    loop {
        if *cancel_rx.borrow() {
            return;
        }
        if cancel_rx.changed().await.is_err() {
            return;
        }
    }
}

async fn read_to_end(
    mut reader: impl tokio::io::AsyncRead + Unpin + Send + 'static,
) -> Result<Vec<u8>, std::io::Error> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).await?;
    Ok(buf)
}

struct GitCommandOutput {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn parse_oid_line(op: &'static str, stdout: &[u8]) -> Result<Option<GitOid>, GitError> {
    let s = std::str::from_utf8(stdout).map_err(|err| GitError::Parse {
        op,
        reason: format!("stdout is not utf-8: {err}"),
    })?;
    let line = s.trim();
    if line.is_empty() {
        return Ok(None);
    }
    GitOid::parse(line).map(Some).map_err(|_| GitError::Parse {
        op,
        reason: format!("unexpected oid: {line}"),
    })
}

fn parse_worktree_list(stdout: &[u8]) -> Result<Vec<GitWorktree>, GitError> {
    let s = std::str::from_utf8(stdout).map_err(|err| GitError::Parse {
        op: "worktree_list",
        reason: format!("stdout is not utf-8: {err}"),
    })?;

    let mut out = Vec::new();
    let mut current_path: Option<PathBuf> = None;
    let mut current_head: Option<GitOid> = None;
    let mut current_branch: Option<GitRefName> = None;
    let mut current_detached = false;

    for raw in s.lines() {
        let line = raw.trim_end();
        if line.is_empty() {
            flush_worktree(
                &mut out,
                &mut current_path,
                &mut current_head,
                &mut current_branch,
                &mut current_detached,
            )?;
            continue;
        }

        if let Some(rest) = line.strip_prefix("worktree ") {
            flush_worktree(
                &mut out,
                &mut current_path,
                &mut current_head,
                &mut current_branch,
                &mut current_detached,
            )?;
            current_path = Some(PathBuf::from(rest));
            continue;
        }

        if let Some(rest) = line.strip_prefix("HEAD ") {
            current_head = Some(GitOid::parse(rest).map_err(|_| GitError::Parse {
                op: "worktree_list",
                reason: format!("invalid HEAD oid: {rest}"),
            })?);
            continue;
        }

        if let Some(rest) = line.strip_prefix("branch ") {
            current_branch = Some(GitRefName::new(rest).map_err(|_| GitError::Parse {
                op: "worktree_list",
                reason: format!("invalid branch ref: {rest}"),
            })?);
            continue;
        }

        if line == "detached" {
            current_detached = true;
            continue;
        }

        // Forward compatible: ignore unrecognized keys like `locked`, `prunable`, etc.
    }

    flush_worktree(
        &mut out,
        &mut current_path,
        &mut current_head,
        &mut current_branch,
        &mut current_detached,
    )?;
    Ok(out)
}

fn flush_worktree(
    out: &mut Vec<GitWorktree>,
    current_path: &mut Option<PathBuf>,
    current_head: &mut Option<GitOid>,
    current_branch: &mut Option<GitRefName>,
    current_detached: &mut bool,
) -> Result<(), GitError> {
    if current_path.is_none() && current_head.is_none() {
        return Ok(());
    }

    let path = current_path.take().ok_or_else(|| GitError::Parse {
        op: "worktree_list",
        reason: "missing worktree path".to_owned(),
    })?;

    let head = current_head.take().ok_or_else(|| GitError::Parse {
        op: "worktree_list",
        reason: "missing worktree HEAD".to_owned(),
    })?;

    let wt = GitWorktree {
        path,
        head,
        branch: current_branch.take(),
        detached: *current_detached,
    };
    *current_detached = false;
    out.push(wt);
    Ok(())
}

impl GitBackend for GitCliBackend {
    fn resolve_commits<'a>(
        &'a self,
        repo_root: &'a Path,
        revs: &'a [GitRevision],
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<Vec<Option<GitOid>>, GitError>> {
        Box::pin(async move {
            if revs.is_empty() {
                return Ok(Vec::new());
            }

            let mut input = String::new();
            for rev in revs {
                input.push_str(rev.as_str());
                input.push_str("^{commit}\n");
            }

            let output = self
                .run_git(
                    "resolve_commits",
                    repo_root,
                    &["cat-file", "--batch-check=%(objectname) %(objecttype)"],
                    Some(input.as_bytes()),
                    options,
                )
                .await?;

            if !output.status.success() {
                let exit_code = output.status.code().unwrap_or(-1);
                return Err(GitError::CommandFailed {
                    op: "resolve_commits",
                    exit_code,
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                });
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut out = Vec::with_capacity(revs.len());
            for line in stdout.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                if line.ends_with(" missing") {
                    out.push(None);
                    continue;
                }

                let mut parts = line.split_whitespace();
                let oid = parts.next().ok_or_else(|| GitError::Parse {
                    op: "resolve_commits",
                    reason: format!("unexpected output line: {line}"),
                })?;
                let kind = parts.next().ok_or_else(|| GitError::Parse {
                    op: "resolve_commits",
                    reason: format!("unexpected output line: {line}"),
                })?;
                if kind != "commit" {
                    out.push(None);
                    continue;
                }

                out.push(Some(GitOid::parse(oid).map_err(|_| GitError::Parse {
                    op: "resolve_commits",
                    reason: format!("invalid oid in output: {oid}"),
                })?));
            }

            if out.len() != revs.len() {
                return Err(GitError::Parse {
                    op: "resolve_commits",
                    reason: format!("expected {} lines of output, got {}", revs.len(), out.len()),
                });
            }

            Ok(out)
        })
    }

    fn merge_base<'a>(
        &'a self,
        repo_root: &'a Path,
        a: &'a GitOid,
        b: &'a GitOid,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<Option<GitOid>, GitError>> {
        Box::pin(async move {
            let output = self
                .run_git(
                    "merge_base",
                    repo_root,
                    &["merge-base", a.as_str(), b.as_str()],
                    None,
                    options,
                )
                .await?;

            match output.status.code().unwrap_or(-1) {
                0 => parse_oid_line("merge_base", &output.stdout),
                1 => Ok(None),
                exit_code => Err(GitError::CommandFailed {
                    op: "merge_base",
                    exit_code,
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                }),
            }
        })
    }

    fn is_ancestor<'a>(
        &'a self,
        repo_root: &'a Path,
        ancestor: &'a GitOid,
        descendant: &'a GitOid,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<bool, GitError>> {
        Box::pin(async move {
            let output = self
                .run_git(
                    "is_ancestor",
                    repo_root,
                    &[
                        "merge-base",
                        "--is-ancestor",
                        ancestor.as_str(),
                        descendant.as_str(),
                    ],
                    None,
                    options,
                )
                .await?;

            match output.status.code().unwrap_or(-1) {
                0 => Ok(true),
                1 => Ok(false),
                exit_code => Err(GitError::CommandFailed {
                    op: "is_ancestor",
                    exit_code,
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                }),
            }
        })
    }

    fn worktree_list<'a>(
        &'a self,
        repo_root: &'a Path,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<Vec<GitWorktree>, GitError>> {
        Box::pin(async move {
            let output = self
                .run_git(
                    "worktree_list",
                    repo_root,
                    &["worktree", "list", "--porcelain"],
                    None,
                    options,
                )
                .await?;

            if !output.status.success() {
                let exit_code = output.status.code().unwrap_or(-1);
                return Err(GitError::CommandFailed {
                    op: "worktree_list",
                    exit_code,
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                });
            }

            parse_worktree_list(&output.stdout)
        })
    }

    fn worktree_add<'a>(
        &'a self,
        repo_root: &'a Path,
        path: &'a Path,
        target: &'a GitWorktreeTarget,
        add_options: GitWorktreeAddOptions,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<(), GitError>> {
        Box::pin(async move {
            let mut args: Vec<&str> = vec!["worktree", "add"];
            if add_options.force {
                args.push("--force");
            }
            if add_options.detach {
                args.push("--detach");
            }
            if !add_options.checkout {
                args.push("--no-checkout");
            }
            if let Some(branch) = add_options.new_branch.as_deref() {
                if add_options.reset_branch {
                    args.push("-B");
                } else {
                    args.push("-b");
                }
                args.push(branch);
            }

            let path_str = path.to_str().ok_or_else(|| GitError::Parse {
                op: "worktree_add",
                reason: "worktree path is not utf-8".to_owned(),
            })?;
            args.push(path_str);

            match target {
                GitWorktreeTarget::Head => {}
                GitWorktreeTarget::Revision(rev) => {
                    // Prevent option-like revs (e.g. `--help`) from being parsed as flags.
                    args.push("--");
                    args.push(rev.as_str());
                }
            }

            let output = self
                .run_git("worktree_add", repo_root, &args, None, options)
                .await?;

            if !output.status.success() {
                let exit_code = output.status.code().unwrap_or(-1);
                return Err(GitError::CommandFailed {
                    op: "worktree_add",
                    exit_code,
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                });
            }
            Ok(())
        })
    }

    fn worktree_remove<'a>(
        &'a self,
        repo_root: &'a Path,
        path: &'a Path,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<(), GitError>> {
        Box::pin(async move {
            let path_str = path.to_str().ok_or_else(|| GitError::Parse {
                op: "worktree_remove",
                reason: "worktree path is not utf-8".to_owned(),
            })?;

            let output = self
                .run_git(
                    "worktree_remove",
                    repo_root,
                    &["worktree", "remove", "--force", "--", path_str],
                    None,
                    options,
                )
                .await?;

            if !output.status.success() {
                let exit_code = output.status.code().unwrap_or(-1);
                return Err(GitError::CommandFailed {
                    op: "worktree_remove",
                    exit_code,
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                });
            }
            Ok(())
        })
    }

    fn worktree_prune<'a>(
        &'a self,
        repo_root: &'a Path,
        options: GitRunOptions,
    ) -> BoxFuture<'a, Result<(), GitError>> {
        Box::pin(async move {
            let output = self
                .run_git(
                    "worktree_prune",
                    repo_root,
                    &["worktree", "prune"],
                    None,
                    options,
                )
                .await?;

            if !output.status.success() {
                let exit_code = output.status.code().unwrap_or(-1);
                return Err(GitError::CommandFailed {
                    op: "worktree_prune",
                    exit_code,
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                });
            }
            Ok(())
        })
    }
}
