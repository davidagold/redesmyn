from __future__ import annotations

from pathlib import Path

import pytest
import sqlite3

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
            TaskSpec(task_id="T-1", title="Parent"),
            TaskSpec(task_id="T-2", title="Child", parent="T-1"),
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
        tasks=[TaskSpec(task_id="T-1", title="Task")],
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
    assert task.branch_name == "rn/cli-epic/T-1-task"

    shell_proc = run_rn(repo.repo_root, ["shell", "--task-id", str(task_id), "--print"])
    assert shell_proc.returncode == 0, shell_proc.stderr
    worktree_path = Path(shell_proc.stdout.strip())
    assert (
        worktree_path
        == repo.repo_root / ".redesmyn" / "worktrees" / "rn" / "cli-epic" / "T-1-task"
    )
    assert worktree_path.exists()
    assert (
        repo.git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree_path)
        == "rn/cli-epic/T-1-task"
    )


@pytest.mark.integration
def test_merge_run_cancel_local_marks_run_canceled(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task")],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "add epic docs"], cwd=repo.repo_root)

    init_proc = run_rn(repo.repo_root, ["init"])
    assert init_proc.returncode == 0, init_proc.stderr

    proc = run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc.returncode == 0, proc.stderr

    epic_db_path = db_path(repo.repo_root)
    local_path = f"epics/{epic_slug}/tasks/T-1/README.md"

    with sqlite3.connect(epic_db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, epic_id, branch_name FROM tasks WHERE local_path = ?",
            (local_path,),
        ).fetchone()
        assert row is not None
        task_id = int(row["id"])
        epic_id = int(row["epic_id"])
        branch_name = str(row["branch_name"])
        run_id = "cli-test-run"

        conn.execute(
            "INSERT INTO merge_runs ("
            "run_id, epic_id, requested_task_id, status, scope, allow_running, force, canonical, plan, "
            "blocked_task_id, blocked_step_index, blocked_step_kind, blocked_branch_name, blocked_worktree_path, blocked_error"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                epic_id,
                task_id,
                "blocked",
                "spine",
                0,
                0,
                1,
                "{}",
                task_id,
                0,
                "rebase",
                branch_name,
                None,
                "conflict",
            ),
        )
        conn.commit()

    cancel_proc = run_rn(
        repo.repo_root,
        ["merge-run", "cancel", run_id, "--local", "--yes"],
    )
    assert cancel_proc.returncode == 0, cancel_proc.stderr
    assert f"Merge run canceled: {run_id}" in cancel_proc.stdout

    with sqlite3.connect(epic_db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status, blocked_step_kind, blocked_branch_name, blocked_worktree_path, blocked_error "
            "FROM merge_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "canceled"
        assert row["blocked_step_kind"] is None
        assert row["blocked_branch_name"] is None
        assert row["blocked_worktree_path"] is None
        assert row["blocked_error"] is None


@pytest.mark.integration
def test_sync_from_local_does_not_rename_worktree_branch_when_title_changes(
    tmp_path: Path,
) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task")],
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
    original_branch = task.branch_name
    assert original_branch == "rn/cli-epic/T-1-task"

    shell_proc = run_rn(repo.repo_root, ["shell", "--task-id", str(task_id), "--print"])
    assert shell_proc.returncode == 0, shell_proc.stderr
    worktree_path = Path(shell_proc.stdout.strip())
    assert (
        repo.git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree_path)
        == original_branch
    )

    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Renamed Task")],
    )
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "rename task"], cwd=repo.repo_root)

    proc2 = run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc2.returncode == 0, proc2.stderr

    updated = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md"
    )
    assert updated.branch_name == original_branch
    assert (
        repo.git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree_path)
        == original_branch
    )


@pytest.mark.integration
def test_sync_from_local_repairs_worktree_branch_mismatch_without_db_change(
    tmp_path: Path,
) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task")],
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
    assert task.branch_name == "rn/cli-epic/T-1-task"

    shell_proc = run_rn(repo.repo_root, ["shell", "--task-id", str(task.id), "--print"])
    assert shell_proc.returncode == 0, shell_proc.stderr
    worktree_path = Path(shell_proc.stdout.strip())

    repo.git(["branch", "-m", "mismatched"], cwd=worktree_path)
    assert (
        repo.git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree_path)
        == "mismatched"
    )

    proc2 = run_rn(repo.repo_root, ["sync", "--from", "local"])
    assert proc2.returncode == 0, proc2.stderr

    task2 = db_task_row(
        epic_db_path, local_path=f"epics/{epic_slug}/tasks/T-1/README.md"
    )
    assert task2.branch_name == "rn/cli-epic/T-1-task"
    assert (
        repo.git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree_path)
        == "rn/cli-epic/T-1-task"
    )


@pytest.mark.integration
def test_shell_no_create_errors_when_worktree_missing(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[TaskSpec(task_id="T-1", title="Task")],
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
        tasks=[TaskSpec(task_id="T-1", title="Task")],
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
) -> tuple[int, int, str, str]:
    write_docs(
        repo_root=repo.repo_root,
        epic_slug=epic_slug,
        tasks=[
            TaskSpec(task_id="T-1", title="Parent"),
            TaskSpec(task_id="T-2", title="Child", parent="T-1"),
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
    assert parent.branch_name is not None
    assert child.branch_name is not None
    return parent.id, child.id, parent.branch_name, child.branch_name


@pytest.mark.integration
def test_merge_prompts_for_confirmation_and_supports_yes_flag(tmp_path: Path) -> None:
    repo = ScenarioRepo.init(tmp_path)
    epic_slug = "cli-epic"
    parent_id, child_id, parent_branch, _child_branch = (
        _seed_stack_for_merge_or_restack(repo=repo, epic_slug=epic_slug)
    )

    repo.git(["checkout", "-b", parent_branch], cwd=repo.repo_root)
    (repo.repo_root / "parent.txt").write_text("parent\n", encoding="utf-8")
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "parent change"], cwd=repo.repo_root)
    repo.git(["checkout", "main"], cwd=repo.repo_root)
    repo.git(["merge", "--ff-only", parent_branch], cwd=repo.repo_root)

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
    parent_id, child_id, parent_branch, _child_branch = (
        _seed_stack_for_merge_or_restack(repo=repo, epic_slug=epic_slug)
    )

    repo.git(["checkout", "-b", parent_branch], cwd=repo.repo_root)
    (repo.repo_root / "parent.txt").write_text("parent\n", encoding="utf-8")
    repo.git(["add", "-A"], cwd=repo.repo_root)
    repo.git(["commit", "-m", "parent change"], cwd=repo.repo_root)
    repo.git(["checkout", "main"], cwd=repo.repo_root)
    repo.git(["merge", "--ff-only", parent_branch], cwd=repo.repo_root)

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
