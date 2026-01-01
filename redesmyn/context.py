from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from redesmyn.repo import canonical_repo_root, worktree_root


@dataclass(frozen=True, slots=True)
class RepoContext:
    repo_root: Path
    worktree_root: Path
    state_dir: Path
    db_path: Path


def build_repo_context(
    *,
    repo_root: Path,
    worktree_root: Path | None = None,
    state_dir_name: str = ".redesmyn",
    db_filename: str = "redesmyn.sqlite3",
    db_path: Path | None = None,
) -> RepoContext:
    worktree = worktree_root or repo_root
    state_dir = repo_root / state_dir_name
    resolved_db_path = db_path or (state_dir / db_filename)
    return RepoContext(
        repo_root=repo_root,
        worktree_root=worktree,
        state_dir=state_dir,
        db_path=resolved_db_path,
    )


def get_repo_context(cwd: Path | None = None) -> RepoContext:
    worktree = worktree_root(cwd=cwd)
    repo_root = canonical_repo_root(cwd=cwd)
    return build_repo_context(repo_root=repo_root, worktree_root=worktree)
