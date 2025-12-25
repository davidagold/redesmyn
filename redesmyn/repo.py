from __future__ import annotations

import subprocess
from pathlib import Path


class NotAGitRepositoryError(RuntimeError):
    pass


def _run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )


def find_repo_root(*, cwd: Path | None = None) -> Path:
    proc = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if proc.returncode != 0:
        raise NotAGitRepositoryError(proc.stderr.strip() or "Not a git repository")
    return Path(proc.stdout.strip())


def current_branch(repo_root: Path) -> str:
    proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
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

