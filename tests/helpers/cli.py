from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    title: str
    branch: str | None = None
    stacked_on: str | None = None


@dataclass(frozen=True, slots=True)
class DbTaskRow:
    id: int
    branch_name: str | None
    parent_task_id: int | None
    state: str
    worktree_path: str | None


def _noninteractive_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    env.setdefault("GIT_CONFIG_GLOBAL", os.devnull)
    env.setdefault("GCM_INTERACTIVE", "never")
    return env


def run_rn(
    repo_root: Path,
    args: list[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    # Prefer the current repo code (editable install in the test venv) while
    # still running the CLI with `cwd=repo_root` (a temp git repo without a
    # `pyproject.toml`, so `uv run rn ...` isn't suitable here).
    cmd = [
        sys.executable,
        "-c",
        "from redesmyn.cli import main; main()",
        *args,
    ]
    merged_env = _noninteractive_env()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
        env=merged_env,
        timeout=timeout_s,
    )


def write_docs(*, repo_root: Path, epic_slug: str, tasks: list[TaskSpec]) -> None:
    epic_dir = repo_root / "epics" / epic_slug
    tasks_dir = epic_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    epic_readme = epic_dir / "README.md"
    epic_readme.write_text(
        "\n".join(
            [
                "# CLI Epic",
                "",
                "## Metadata",
                "```yaml",
                f"slug: {epic_slug}",
                "name: CLI Epic",
                "root_branch: main",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for task in tasks:
        task_dir = tasks_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        stacked_on_line = (
            f"stacked_on: {task.stacked_on}"
            if task.stacked_on is not None
            else "stacked_on:"
        )
        node_lines: list[str] = []
        if task.branch is not None:
            node_lines = ["node:", f"  branch: {task.branch}"]
        (task_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {task.title}",
                    "",
                    "## Metadata",
                    "```yaml",
                    f"id: {task.task_id}",
                    stacked_on_line,
                    *node_lines,
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(".redesmyn/\n", encoding="utf-8")


def db_path(repo_root: Path) -> Path:
    return repo_root / ".redesmyn" / "redesmyn.sqlite3"


def db_task_row(db_path: Path, *, local_path: str) -> DbTaskRow:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, branch_name, parent_task_id, state, worktree_path "
            "FROM tasks WHERE local_path = ?",
            (local_path,),
        ).fetchone()
        assert row is not None
        parent_task_id = row["parent_task_id"]
        return DbTaskRow(
            id=int(row["id"]),
            branch_name=(
                str(row["branch_name"]) if row["branch_name"] is not None else None
            ),
            parent_task_id=int(parent_task_id) if parent_task_id is not None else None,
            state=str(row["state"]),
            worktree_path=(
                str(row["worktree_path"]) if row["worktree_path"] is not None else None
            ),
        )


def db_update_task_state(db_path: Path, *, task_id: int, state: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE tasks SET state = ? WHERE id = ?", (state, task_id))
        conn.commit()


def write_fake_shell(path: Path) -> None:
    path.write_text(
        '#!/bin/sh\necho "fake-shell"\nexit 0\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
