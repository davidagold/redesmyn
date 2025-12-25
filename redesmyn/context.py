from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from redesmyn.repo import find_repo_root


@dataclass(frozen=True, slots=True)
class RepoContext:
    repo_root: Path
    state_dir: Path
    db_path: Path


def get_repo_context(cwd: Path | None = None) -> RepoContext:
    repo_root = find_repo_root(cwd=cwd)
    state_dir = repo_root / ".redesmyn"
    db_path = state_dir / "redesmyn.sqlite3"
    return RepoContext(repo_root=repo_root, state_dir=state_dir, db_path=db_path)

