from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict


class NotAGitRepositoryError(RuntimeError):
    pass


class CommitInfo(TypedDict):
    author_name: str | None
    author_email: str | None
    authored_at: datetime | None
    committer_name: str | None
    committer_email: str | None
    committed_at: datetime | None
    title: str | None
    message: str | None


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


def git_rename_current_branch(worktree_path: Path, *, new_name: str) -> None:
    if not new_name or new_name == "HEAD":
        raise ValueError(f"Invalid branch name: {new_name!r}")
    proc = _run_git(["branch", "-m", new_name], cwd=worktree_path)
    if proc.returncode != 0:
        output = (proc.stdout + "\n" + proc.stderr).strip()
        raise GitCommandError(output or f"git branch -m {new_name} failed")


def git_merge_base(repo_root: Path, ref_a: str, ref_b: str) -> str | None:
    proc = _run_git(["merge-base", ref_a, ref_b], cwd=repo_root)
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def git_is_ancestor(repo_root: Path, ancestor_ref: str, descendant_ref: str) -> bool:
    proc = _run_git(
        ["merge-base", "--is-ancestor", ancestor_ref, descendant_ref], cwd=repo_root
    )
    return proc.returncode == 0


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


def git_rev_parse(repo_root: Path, ref: str) -> str | None:
    proc = _run_git(["rev-parse", "--verify", ref], cwd=repo_root)
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def git_rev_list_range(
    repo_root: Path,
    rev_range: str,
    *,
    first_parent: bool = True,
    max_count: int | None = None,
    reverse: bool = False,
) -> list[str]:
    args = ["rev-list"]
    if first_parent:
        args.append("--first-parent")
    if reverse:
        args.append("--reverse")
    if max_count is not None:
        args.extend(["-n", str(max_count)])
    args.append(rev_range)
    proc = _run_git(args, cwd=repo_root)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def git_rev_list_count(
    repo_root: Path, rev_range: str, *, first_parent: bool = True
) -> int:
    args = ["rev-list"]
    if first_parent:
        args.append("--first-parent")
    args.extend(["--count", rev_range])
    proc = _run_git(args, cwd=repo_root)
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.strip() or "0")
    except ValueError:
        return 0


def git_for_each_ref(repo_root: Path, *, prefix: str = "refs/heads") -> dict[str, str]:
    proc = _run_git(
        ["for-each-ref", prefix, "--format=%(refname) %(objectname)"], cwd=repo_root
    )
    if proc.returncode != 0:
        return {}

    refs: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            refname, sha = line.split(" ", 1)
        except ValueError:
            continue
        refs[refname.strip()] = sha.strip()
    return refs


def git_commit_info(repo_root: Path, shas: list[str]) -> dict[str, CommitInfo]:
    """
    Return basic commit metadata keyed by sha.

    `git show` emits ISO 8601; we parse it to `datetime` so API response types
    remain accurate.
    """

    def parse_iso_dt(value: str) -> datetime | None:
        try:
            # `git` usually emits "+00:00", but tolerate "Z" if it appears.
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

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
        [
            "show",
            "-s",
            "--format=%H%x1f%an%x1f%ae%x1f%aI%x1f%cn%x1f%ce%x1f%cI%x1f%s%x1f%b%x1e",
            *unique,
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        return {}

    info: dict[str, CommitInfo] = {}
    for record in proc.stdout.split("\x1e"):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split("\x1f")
        if len(parts) != 9:
            continue
        (
            sha,
            author_name,
            author_email,
            authored_at,
            committer_name,
            committer_email,
            committed_at,
            title,
            message,
        ) = parts
        commit_info: CommitInfo = {
            "author_name": author_name or None,
            "author_email": author_email or None,
            "authored_at": parse_iso_dt(authored_at) if authored_at else None,
            "committer_name": committer_name or None,
            "committer_email": committer_email or None,
            "committed_at": parse_iso_dt(committed_at) if committed_at else None,
            "title": title or None,
            "message": message.rstrip("\n") or None,
        }
        info[sha] = commit_info
    return info


class GitCommandError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GitWorktreeEntry:
    path: Path
    branch_ref: str | None
    head: str | None


def git_worktree_list(repo_root: Path) -> list[GitWorktreeEntry]:
    proc = _run_git(["worktree", "list", "--porcelain"], cwd=repo_root)
    if proc.returncode != 0:
        raise GitCommandError(proc.stderr.strip() or "git worktree list failed")

    entries: list[GitWorktreeEntry] = []
    current_path: Path | None = None
    current_branch: str | None = None
    current_head: str | None = None

    def flush() -> None:
        nonlocal current_path, current_branch, current_head
        if current_path is None:
            return
        entries.append(
            GitWorktreeEntry(
                path=current_path,
                branch_ref=current_branch,
                head=current_head,
            )
        )
        current_path = None
        current_branch = None
        current_head = None

    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("worktree "):
            flush()
            current_path = Path(line.removeprefix("worktree ").strip())
            continue
        if line.startswith("branch "):
            current_branch = line.removeprefix("branch ").strip() or None
            continue
        if line.startswith("HEAD "):
            current_head = line.removeprefix("HEAD ").strip() or None
            continue

    flush()
    return entries


def git_worktree_path_for_branch(repo_root: Path, branch_name: str) -> Path | None:
    branch_ref = f"refs/heads/{branch_name}"
    for entry in git_worktree_list(repo_root):
        if entry.branch_ref == branch_ref:
            return entry.path
    return None


def git_status_porcelain(worktree_path: Path) -> str:
    proc = _run_git(["status", "--porcelain=v1"], cwd=worktree_path)
    if proc.returncode != 0:
        raise GitCommandError(proc.stderr.strip() or "git status failed")
    return proc.stdout


def git_has_in_progress_operation(worktree_path: Path) -> bool:
    """Return True when the worktree is mid-mutation (rebase/merge/cherry-pick/revert)."""

    def git_path(name: str) -> Path | None:
        proc = _run_git(["rev-parse", "--git-path", name], cwd=worktree_path)
        if proc.returncode != 0:
            return None
        raw = proc.stdout.strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = worktree_path / path
        return path

    candidates = (
        "rebase-apply",
        "rebase-merge",
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
    )
    for name in candidates:
        path = git_path(name)
        if path is not None and path.exists():
            return True
    return False


def git_rebase(worktree_path: Path, upstream_ref: str) -> None:
    proc = _run_git(["rebase", upstream_ref], cwd=worktree_path)
    if proc.returncode != 0:
        output = (proc.stdout + "\n" + proc.stderr).strip()
        raise GitCommandError(output or f"git rebase {upstream_ref} failed")


def git_rebase_update_refs(worktree_path: Path, upstream_ref: str) -> None:
    proc = _run_git(["rebase", "--update-refs", upstream_ref], cwd=worktree_path)
    if proc.returncode != 0:
        output = (proc.stdout + "\n" + proc.stderr).strip()
        raise GitCommandError(
            output or f"git rebase --update-refs {upstream_ref} failed"
        )


def git_merge_ff_only(worktree_path: Path, ref: str) -> None:
    proc = _run_git(["merge", "--ff-only", ref], cwd=worktree_path)
    if proc.returncode != 0:
        output = (proc.stdout + "\n" + proc.stderr).strip()
        raise GitCommandError(output or f"git merge --ff-only {ref} failed")


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
