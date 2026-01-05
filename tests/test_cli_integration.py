from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.scenarios.scenario import ScenarioRepo


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


def _run_rn(
    repo_root: Path,
    args: list[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    rn_path = shutil.which("rn")
    if rn_path is None:
        candidate = Path(sys.executable).with_name("rn")
        if candidate.exists():
            rn_path = str(candidate)
        else:
            raise RuntimeError("rn executable not found on PATH")

    cmd = [rn_path, *args]
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


def _write_docs(*, repo_root: Path, epic_slug: str, tasks: list[TaskSpec]) -> None:
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
        node_lines = []
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


def _db_path(repo_root: Path) -> Path:
    return repo_root / ".redesmyn" / "redesmyn.sqlite3"


def _db_task_row(db_path: Path, *, local_path: str) -> DbTaskRow:
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


def _db_update_task_state(db_path: Path, *, task_id: int, state: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE tasks SET state = ? WHERE id = ?", (state, task_id))
        conn.commit()


def _write_fake_shell(path: Path) -> None:
    path.write_text(
        '#!/bin/sh\necho "fake-shell"\nexit 0\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.integration
def test_sync_from_local_imports_tasks_and_sets_parent_links_without_creating_branches(
    tmp_path: Path,
) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    _write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[
            TaskSpec(task_id="T-1", title="Parent", branch="task-1"),
            TaskSpec(task_id="T-2", title="Child", branch="task-2", stacked_on="T-1"),
        ],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = _run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = _run_rn(repo.repo_root, ["sync", "--from", "local", "--no-create-branches"])
    assert proc.returncode == 0, proc.stderr

    db_path = _db_path(repo.repo_root)
    parent = _db_task_row(db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md")
    child = _db_task_row(db_path, local_path=f"epics/{epic_slug}/tasks/T-2/README.md")

    assert parent.branch_name is None
    assert child.branch_name is None
    assert child.parent_task_id == parent.id


@pytest.mark.integration
def test_shell_print_outputs_worktree_path_and_exits_zero(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    _write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task", branch="task-1")],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = _run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = _run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    db_path = _db_path(repo.repo_root)
    task = _db_task_row(db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md")
    task_id = task.id

    shell_proc = _run_rn(
        repo.repo_root, ["shell", "--task-id", str(task_id), "--print"]
    )
    assert shell_proc.returncode == 0, shell_proc.stderr
    worktree_path = Path(shell_proc.stdout.strip())
    assert worktree_path == repo.repo_root / ".redesmyn" / "worktrees" / "task-1"
    assert worktree_path.exists()
    assert (
        repo.git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree_path) == "task-1"
    )


@pytest.mark.integration
def test_shell_no_create_errors_when_worktree_missing(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    _write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task", branch="task-1")],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = _run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = _run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    db_path = _db_path(repo.repo_root)
    task = _db_task_row(db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md")
    task_id = task.id

    shell_proc = _run_rn(
        repo.repo_root,
        ["shell", "--task-id", str(task_id), "--no-create", "--print"],
    )
    assert shell_proc.returncode == 2
    assert "worktree does not exist" in shell_proc.stderr


@pytest.mark.integration
def test_shell_refuses_nesting_by_default_and_allows_nested(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    _write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task", branch="task-1")],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = _run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = _run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    db_path = _db_path(repo.repo_root)
    task = _db_task_row(db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md")
    task_id = task.id

    created = _run_rn(repo.repo_root, ["shell", "--task-id", str(task_id), "--print"])
    assert created.returncode == 0, created.stderr

    fake_shell = tmp_path / "fake_shell.sh"
    _write_fake_shell(fake_shell)

    refused = _run_rn(
        repo.repo_root,
        ["shell", "--task-id", str(task_id), "--no-create"],
        env={"RN_PARENT_CWD": "/tmp", "SHELL": str(fake_shell)},
        timeout_s=10.0,
    )
    assert refused.returncode == 2
    assert "already in an `rn shell` subshell" in refused.stderr

    allowed = _run_rn(
        repo.repo_root,
        ["shell", "--task-id", str(task_id), "--no-create", "--nested"],
        env={"RN_PARENT_CWD": "/tmp", "SHELL": str(fake_shell)},
        timeout_s=10.0,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "fake-shell" in allowed.stdout


def _seed_stack_for_merge_or_restack(
    *, repo: ScenarioRepo, epic_slug: str
) -> tuple[int, int]:
    _write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[
            TaskSpec(task_id="T-1", title="Parent", branch="task-1"),
            TaskSpec(task_id="T-2", title="Child", branch="task-2", stacked_on="T-1"),
        ],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = _run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = _run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    db_path = _db_path(repo.repo_root)
    parent = _db_task_row(db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md")
    child = _db_task_row(db_path, local_path=f"epics/{epic_slug}/tasks/T-2/README.md")
    return parent.id, child.id


@pytest.mark.integration
def test_merge_prompts_for_confirmation_and_supports_yes_flag(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    parent_id, child_id = _seed_stack_for_merge_or_restack(
        repo=repo, epic_slug=epic_slug
    )

    repo.git(["checkout", "-b", "task-1"], cwd=repo.repo_root)
    (repo.repo_root / "parent.txt").write_text("parent\n", encoding="utf-8")
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "parent change"], cwd=repo.repo_root)
    repo.git(["checkout", "main"], cwd=repo.repo_root)
    repo.git(["merge", "--ff-only", "task-1"], cwd=repo.repo_root)

    shell_proc = _run_rn(
        repo.repo_root, ["shell", "--task-id", str(child_id), "--print"]
    )
    assert shell_proc.returncode == 0, shell_proc.stderr
    child_worktree = Path(shell_proc.stdout.strip())

    (child_worktree / "child.txt").write_text("child\n", encoding="utf-8")
    repo.git(["add", "-A"], cwd=child_worktree)
    repo.git(["commit", "-m", "child change"], cwd=child_worktree)

    db_path = _db_path(repo.repo_root)
    _db_update_task_state(db_path, task_id=parent_id, state="done")
    _db_update_task_state(db_path, task_id=child_id, state="in_progress")

    refused = _run_rn(
        repo.repo_root,
        ["merge", "--task", str(child_id), "--force"],
        input_text="n\n",
        timeout_s=10.0,
    )
    combined_refused = refused.stdout + refused.stderr
    assert refused.returncode == 1, combined_refused
    assert "Merge plan:" in combined_refused
    assert "Proceed with merge?" in combined_refused

    allowed = _run_rn(
        repo.repo_root,
        ["merge", "--task", str(child_id), "--force", "-y"],
        timeout_s=20.0,
    )
    combined_allowed = allowed.stdout + allowed.stderr
    assert allowed.returncode == 0, combined_allowed
    assert "Merged into" in combined_allowed


@pytest.mark.integration
def test_restack_prompts_for_confirmation_and_supports_yes_flag(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    parent_id, child_id = _seed_stack_for_merge_or_restack(
        repo=repo, epic_slug=epic_slug
    )

    repo.git(["checkout", "-b", "task-1"], cwd=repo.repo_root)
    (repo.repo_root / "parent.txt").write_text("parent\n", encoding="utf-8")
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "parent change"], cwd=repo.repo_root)
    repo.git(["checkout", "main"], cwd=repo.repo_root)
    repo.git(["merge", "--ff-only", "task-1"], cwd=repo.repo_root)

    shell_proc = _run_rn(
        repo.repo_root, ["shell", "--task-id", str(child_id), "--print"]
    )
    assert shell_proc.returncode == 0, shell_proc.stderr
    child_worktree = Path(shell_proc.stdout.strip())

    (child_worktree / "child.txt").write_text("child\n", encoding="utf-8")
    repo.git(["add", "-A"], cwd=child_worktree)
    repo.git(["commit", "-m", "child change"], cwd=child_worktree)

    (repo.repo_root / "main.txt").write_text("main\n", encoding="utf-8")
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "main change"], cwd=repo.repo_root)

    db_path = _db_path(repo.repo_root)
    _db_update_task_state(db_path, task_id=parent_id, state="done")
    _db_update_task_state(db_path, task_id=child_id, state="in_progress")

    refused = _run_rn(
        repo.repo_root,
        ["restack", "--task", str(child_id)],
        input_text="n\n",
        timeout_s=10.0,
    )
    combined_refused = refused.stdout + refused.stderr
    assert refused.returncode == 1, combined_refused
    assert "Restack plan:" in combined_refused
    assert "Proceed with restack?" in combined_refused

    allowed = _run_rn(
        repo.repo_root,
        ["restack", "--task", str(child_id), "-y"],
        timeout_s=20.0,
    )
    combined_allowed = allowed.stdout + allowed.stderr
    assert allowed.returncode == 0, combined_allowed
    assert "Restacked." in combined_allowed
