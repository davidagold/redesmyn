from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from redesmyn.db import Agent, AgentSession, Epic, MergeRun, Repository, Task
from redesmyn.domain.enums import AgentStatus, MergeRunStatus, TaskState

from tests.scenarios.scenario import Scenario


@dataclass(frozen=True, slots=True)
class SeededSpine:
    epic_id: int
    parent_task_id: int
    child_task_id: int
    parent_branch: str
    child_branch: str
    parent_worktree: Path
    child_worktree: Path


async def seed_merged_parent(scenario: Scenario) -> SeededSpine:
    repo = scenario.repo

    parent_branch = "task-parent"
    child_branch = "task-child"

    parent_wt = repo.create_worktree(branch_name=parent_branch)
    repo.commit_file(
        worktree_path=parent_wt,
        relpath="parent.txt",
        content="parent\n",
        message="parent change",
    )
    repo.fast_forward_main(from_branch=parent_branch)

    child_wt = repo.create_worktree(branch_name=child_branch, from_ref="main")
    repo.commit_file(
        worktree_path=child_wt,
        relpath="child.txt",
        content="child\n",
        message="child change",
    )

    async with scenario.db.session() as session:
        repo_row = await session.scalar(
            select(Repository).where(
                Repository.repo_root == str(scenario.ctx.repo_root)
            )
        )
        if repo_row is None:
            raise RuntimeError("Scenario repository row missing")

        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug="test-epic",
            root_branch="main",
        )
        session.add(epic)
        await session.flush()

        parent = Task(
            epic_id=epic.id,
            title="Parent",
            branch_name=parent_branch,
            worktree_path=str(parent_wt),
            state=TaskState.Done,
        )
        session.add(parent)
        await session.flush()

        child = Task(
            epic_id=epic.id,
            title="Child",
            branch_name=child_branch,
            worktree_path=str(child_wt),
            parent_task_id=parent.id,
            state=TaskState.InProgress,
            merge_ready_at=datetime.now(UTC),
        )
        session.add(child)
        await session.commit()

        return SeededSpine(
            epic_id=epic.id,
            parent_task_id=parent.id,
            child_task_id=child.id,
            parent_branch=parent_branch,
            child_branch=child_branch,
            parent_worktree=parent_wt,
            child_worktree=child_wt,
        )


async def seed_running_agent(scenario: Scenario) -> SeededSpine:
    seeded = await seed_merged_parent(scenario)
    async with scenario.db.session() as session:
        agent = Agent(display_name="test-agent", status=AgentStatus.Running)
        session.add(agent)
        await session.flush()
        session.add(
            AgentSession(
                agent_id=agent.id,
                task_id=seeded.child_task_id,
                status=AgentStatus.Running,
            )
        )
        await session.commit()
    return seeded


async def seed_conflicted_merge_run(scenario: Scenario) -> SeededSpine:
    seeded = await seed_merged_parent(scenario)

    # Create a real rebase conflict to leave the child worktree in a blocked state.
    scenario.repo.git(["checkout", "main"], cwd=scenario.ctx.repo_root)
    scenario.repo.commit_file(
        worktree_path=scenario.ctx.repo_root,
        relpath="conflict.txt",
        content="main\n",
        message="main conflict",
    )
    scenario.repo.commit_file(
        worktree_path=seeded.child_worktree,
        relpath="conflict.txt",
        content="child\n",
        message="child conflict",
    )
    scenario.repo.try_rebase(worktree_path=seeded.child_worktree, upstream_ref="main")

    async with scenario.db.session() as session:
        run = MergeRun(
            run_id=uuid4().hex,
            epic_id=seeded.epic_id,
            requested_task_id=seeded.child_task_id,
            canonical=True,
            status=MergeRunStatus.Blocked,
            blocked_step_index=0,
            blocked_step_kind="rebase",
            blocked_task_id=seeded.child_task_id,
            blocked_branch_name=seeded.child_branch,
            blocked_worktree_path=str(seeded.child_worktree),
            blocked_error="rebase conflict",
        )
        session.add(run)
        await session.commit()
    return seeded
