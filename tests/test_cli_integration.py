from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.cli import (
    TaskSpec,
    db_path,
    db_task_row,
    db_update_task_state,
    run_rn,
    write_docs,
    write_fake_shell,
)
from tests.scenarios.scenario import ScenarioRepo


@pytest.mark.integration
def test_sync_from_local_imports_tasks_and_sets_parent_links_without_creating_branches(
    tmp_path: Path,
) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[
            TaskSpec(task_id="T-1", title="Parent", branch="task-1"),
            TaskSpec(task_id="T-2", title="Child", branch="task-2", stacked_on="T-1"),
        ],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = run_rn(repo.repo_root, ["sync", "--from", "local", "--no-create-branches"])
    assert proc.returncode == 0, proc.stderr

    epic_db_path = db_path(repo.repo_root)
    parent = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md"
    )
    child = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-2/README.md"
    )

    assert parent.branch_name is None
    assert child.branch_name is None
    assert child.parent_task_id == parent.id


@pytest.mark.integration
def test_shell_print_outputs_worktree_path_and_exits_zero(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task", branch="task-1")],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    epic_db_path = db_path(repo.repo_root)
    task = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md"
    )
    task_id = task.id

    shell_proc = run_rn(repo.repo_root, ["shell", "--task-id", str(task_id), "--print"])
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
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task", branch="task-1")],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    epic_db_path = db_path(repo.repo_root)
    task = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md"
    )
    task_id = task.id

    shell_proc = run_rn(
        repo.repo_root,
        ["shell", "--task-id", str(task_id), "--no-create", "--print"],
    )
    assert shell_proc.returncode == 2
    assert "worktree does not exist" in shell_proc.stderr


@pytest.mark.integration
def test_shell_refuses_nesting_by_default_and_allows_nested(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task", branch="task-1")],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    epic_db_path = db_path(repo.repo_root)
    task = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md"
    )
    task_id = task.id

    created = run_rn(repo.repo_root, ["shell", "--task-id", str(task_id), "--print"])
    assert created.returncode == 0, created.stderr

    fake_shell = tmp_path / "fake_shell.sh"
    write_fake_shell(fake_shell)

    refused = run_rn(
        repo.repo_root,
        ["shell", "--task-id", str(task_id), "--no-create"],
        env={"RN_PARENT_CWD": "/tmp", "SHELL": str(fake_shell)},
        timeout_s=10.0,
    )
    assert refused.returncode == 2
    assert "already in an `rn shell` subshell" in refused.stderr

    allowed = run_rn(
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
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[
            TaskSpec(task_id="T-1", title="Parent", branch="task-1"),
            TaskSpec(task_id="T-2", title="Child", branch="task-2", stacked_on="T-1"),
        ],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    epic_db_path = db_path(repo.repo_root)
    parent = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md"
    )
    child = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-2/README.md"
    )
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

    shell_proc = run_rn(
        repo.repo_root, ["shell", "--task-id", str(child_id), "--print"]
    )
    assert shell_proc.returncode == 0, shell_proc.stderr
    child_worktree = Path(shell_proc.stdout.strip())

    (child_worktree / "child.txt").write_text("child\n", encoding="utf-8")
    repo.git(["add", "-A"], cwd=child_worktree)
    repo.git(["commit", "-m", "child change"], cwd=child_worktree)

    epic_db_path = db_path(repo.repo_root)
    db_update_task_state(epic_db_path, task_id=parent_id, state="done")
    db_update_task_state(epic_db_path, task_id=child_id, state="in_progress")

    refused = run_rn(
        repo.repo_root,
        ["merge", "--task", str(child_id), "--force"],
        input_text="n\n",
        timeout_s=10.0,
    )
    combined_refused = refused.stdout + refused.stderr
    assert refused.returncode == 1, combined_refused
    assert "Merge plan:" in combined_refused
    assert "Proceed with merge?" in combined_refused

    allowed = run_rn(
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

    shell_proc = run_rn(
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

    epic_db_path = db_path(repo.repo_root)
    db_update_task_state(epic_db_path, task_id=parent_id, state="done")
    db_update_task_state(epic_db_path, task_id=child_id, state="in_progress")

    refused = run_rn(
        repo.repo_root,
        ["restack", "--task", str(child_id)],
        input_text="n\n",
        timeout_s=10.0,
    )
    combined_refused = refused.stdout + refused.stderr
    assert refused.returncode == 1, combined_refused
    assert "Restack plan:" in combined_refused
    assert "Proceed with restack?" in combined_refused

    allowed = run_rn(
        repo.repo_root,
        ["restack", "--task", str(child_id), "-y"],
        timeout_s=20.0,
    )
    combined_allowed = allowed.stdout + allowed.stderr
    assert allowed.returncode == 0, combined_allowed
    assert "Restacked." in combined_allowed
