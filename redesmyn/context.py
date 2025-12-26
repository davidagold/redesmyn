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


def get_repo_context(cwd: Path | None = None) -> RepoContext:
    worktree = worktree_root(cwd=cwd)
    repo_root = canonical_repo_root(cwd=cwd)
    state_dir = repo_root / ".redesmyn"
    db_path = state_dir / "redesmyn.sqlite3"
    return RepoContext(
        repo_root=repo_root,
        worktree_root=worktree,
        state_dir=state_dir,
        db_path=db_path,
    )
