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
    return worktree_root(cwd=cwd)


def canonical_repo_root(*, cwd: Path | None = None) -> Path:
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

    return worktree_root(cwd=cwd)


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


def git_merge_base(repo_root: Path, ref_a: str, ref_b: str) -> str | None:
    proc = _run_git(["merge-base", ref_a, ref_b], cwd=repo_root)
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def git_rev_list(
    repo_root: Path,
    ref: str,
    *,
    first_parent: bool = True,
    max_count: int | None = None,
) -> list[str]:
    args = ["rev-list"]
    if first_parent:
        args.append("--first-parent")
    if max_count is not None:
        args.extend(["-n", str(max_count)])
    args.append(ref)
    proc = _run_git(args, cwd=repo_root)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def git_commit_info(
    repo_root: Path, shas: list[str]
) -> dict[str, dict[str, str | None]]:
    unique: list[str] = []
    seen: set[str] = set()
    for sha in shas:
        if not sha or sha in seen:
            continue
        seen.add(sha)
        unique.append(sha)

    if not unique:
        return {}

    proc = _run_git(
        ["show", "-s", "--format=%H%x1f%an%x1f%ae%x1f%aI", *unique], cwd=repo_root
    )
    if proc.returncode != 0:
        return {}

    info: dict[str, dict[str, str | None]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        sha, author_name, author_email, authored_at = parts
        info[sha] = {
            "author_name": author_name or None,
            "author_email": author_email or None,
            "authored_at": authored_at or None,
        }
    return info


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
