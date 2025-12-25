from __future__ import annotations

import subprocess
from pathlib import Path


class NotAGitRepositoryError(RuntimeError):
    pass


def _run_git(
    args: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )


def find_repo_root(*, cwd: Path | None = None) -> Path:
    proc = _run_git(["worktree", "list", "--porcelain"], cwd=cwd)
    if proc.returncode == 0:
        candidates: list[Path] = []
        for line in proc.stdout.splitlines():
            if line.startswith("worktree "):
                candidates.append(Path(line.removeprefix("worktree ").strip()))

        for path in candidates:
            if (path / ".git").is_dir():
                return path

        if candidates:
            return candidates[0]

    fallback = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if fallback.returncode != 0:
        raise NotAGitRepositoryError(fallback.stderr.strip() or "Not a git repository")
    return Path(fallback.stdout.strip())


def worktree_root(*, cwd: Path | None = None) -> Path:
    proc = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if proc.returncode != 0:
        raise NotAGitRepositoryError(proc.stderr.strip() or "Not a git repository")
    return Path(proc.stdout.strip())


def current_branch(*, cwd: Path | None = None) -> str:
    proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    if proc.returncode != 0:
        return "HEAD"
    return proc.stdout.strip()


def default_branch(repo_root: Path) -> str:
    origin_head = _run_git(["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=repo_root)
    if origin_head.returncode == 0:
        return origin_head.stdout.strip().removeprefix("refs/remotes/origin/")

    head = _run_git(["symbolic-ref", "HEAD"], cwd=repo_root)
    if head.returncode == 0:
        return head.stdout.strip().removeprefix("refs/heads/")

    return "main"


def branch_exists(repo_root: Path, branch_name: str) -> bool:
    proc = _run_git(
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"], cwd=repo_root
    )
    return proc.returncode == 0


class GitCommandError(RuntimeError):
    pass


def git_worktree_add(
    repo_root: Path,
    *,
    worktree_path: Path,
    branch_name: str,
    base_ref: str,
) -> None:
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    if branch_exists(repo_root, branch_name):
        args = ["worktree", "add", str(worktree_path), branch_name]
    else:
        args = ["worktree", "add", "-b", branch_name, str(worktree_path), base_ref]

    proc = _run_git(args, cwd=repo_root)
    if proc.returncode != 0:
        raise GitCommandError(proc.stderr.strip() or f"git {' '.join(args)} failed")
