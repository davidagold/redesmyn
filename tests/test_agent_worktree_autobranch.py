from __future__ import annotations

import pytest
from sqlalchemy import select

from redesmyn.agent_runtime import ensure_task_worktree
from redesmyn.db import Epic, Repository, Task
from redesmyn.domain.enums import TaskState

from tests.scenarios.scenario import Scenario


async def test_ensure_task_worktree_creates_branch_when_missing(
    scenario: Scenario,
) -> None:
    async with scenario.db.session() as session:
        repo_row = await session.scalar(
            select(Repository).where(
                Repository.repo_root == str(scenario.ctx.repo_root)
            )
        )
        assert repo_row is not None

        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug="test-epic",
            root_branch="main",
        )
        session.add(epic)
        await session.flush()

        task = Task(
            epic_id=epic.id,
            title="T-1 Example task",
            branch_name=None,
            worktree_path=None,
            state=TaskState.Todo,
        )
        session.add(task)
        await session.flush()

        worktree_path = await ensure_task_worktree(
            session, scenario.ctx, task=task, epic=epic
        )
        await session.commit()

    assert task.branch_name is not None
    assert task.branch_name.startswith("rn/test-epic/")
    assert task.worktree_path == str(worktree_path)
    assert worktree_path.exists()

    scenario.repo.git(
        [
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{task.branch_name}",
        ]
    )


async def test_ensure_task_worktree_errors_when_prefix_branch_exists(
    scenario: Scenario,
) -> None:
    scenario.repo.git(["branch", "rn/test-epic"])

    async with scenario.db.session() as session:
        repo_row = await session.scalar(
            select(Repository).where(
                Repository.repo_root == str(scenario.ctx.repo_root)
            )
        )
        assert repo_row is not None

        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug="test-epic",
            root_branch="main",
        )
        session.add(epic)
        await session.flush()

        task = Task(
            epic_id=epic.id,
            title="T-1 Example task",
            branch_name=None,
            worktree_path=None,
            state=TaskState.Todo,
        )
        session.add(task)
        await session.flush()

        with pytest.raises(RuntimeError) as excinfo:
            await ensure_task_worktree(session, scenario.ctx, task=task, epic=epic)
        await session.commit()

    assert task.branch_name is None
    assert task.worktree_path is None
    message = str(excinfo.value)
    assert "a branch exists at prefix 'rn/test-epic'" in message
    assert "Cannot create branch 'rn/test-epic/T-1-example-task'" in message
