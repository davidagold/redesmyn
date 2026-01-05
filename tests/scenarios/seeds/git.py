from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.db import AgentSession, Epic, MergeRun, Repository, Task
from redesmyn.domain.enums import AgentStatus, MergeRunStatus, TaskState

from tests.scenarios.scenario import Scenario


def _unique_suffix() -> str:
    return uuid4().hex[:8]


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{_unique_suffix()}"


@dataclass(frozen=True, slots=True)
class SeededSpine:
    epic_id: int
    parent_task_id: int
    child_task_id: int
    parent_branch: str
    child_branch: str
    parent_worktree: Path
    child_worktree: Path


@dataclass(frozen=True, slots=True)
class SeededSpineWithDescendant:
    epic_id: int
    parent_task_id: int
    child_task_id: int
    descendant_task_id: int
    parent_branch: str
    child_branch: str
    descendant_branch: str
    parent_worktree: Path
    child_worktree: Path
    descendant_worktree: Path


@dataclass(frozen=True, slots=True)
class SeededThreeTaskChain:
    epic_id: int
    parent_task_id: int
    child_task_id: int
    grandchild_task_id: int
    parent_branch: str
    child_branch: str
    grandchild_branch: str
    parent_worktree: Path
    child_worktree: Path
    grandchild_worktree: Path


async def _require_repository_row(
    session: AsyncSession, scenario: Scenario
) -> Repository:
    repo_row = await session.scalar(
        select(Repository).where(Repository.repo_root == str(scenario.ctx.repo_root))
    )
    if repo_row is None:
        raise RuntimeError("Scenario repository row missing")
    return repo_row


async def seed_merged_parent(scenario: Scenario) -> SeededSpine:
    """Seed a 2-task spine where the parent is marked done and merged into base."""
    repo = scenario.repo

    parent_branch = _unique_name("task-parent")
    child_branch = _unique_name("task-child")

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
        repo_row = await _require_repository_row(session, scenario)

        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug=_unique_name("test-epic"),
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
        session.add(
            AgentSession(
                task_id=seeded.child_task_id,
                status=AgentStatus.Running,
            )
        )
        await session.commit()
    return seeded


async def seed_conflicted_merge_run(scenario: Scenario) -> SeededSpine:
    """Seed a blocked rebase in the child worktree + a MergeRun row marked blocked."""
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


async def seed_active_spine_with_descendant_worktree(
    scenario: Scenario, *, descendant_state: TaskState = TaskState.Todo
) -> SeededSpineWithDescendant:
    """Seed a 2-task active spine + one descendant that already has a worktree."""
    repo = scenario.repo

    parent_branch = _unique_name("task-parent")
    child_branch = _unique_name("task-child")
    descendant_branch = _unique_name("task-descendant")

    parent_wt = repo.create_worktree(branch_name=parent_branch)
    repo.commit_file(
        worktree_path=parent_wt,
        relpath="parent.txt",
        content="parent\n",
        message="parent change",
    )

    child_wt = repo.create_worktree(branch_name=child_branch, from_ref=parent_branch)
    repo.commit_file(
        worktree_path=child_wt,
        relpath="child.txt",
        content="child\n",
        message="child change",
    )

    descendant_wt = repo.create_worktree(
        branch_name=descendant_branch, from_ref=child_branch
    )
    repo.commit_file(
        worktree_path=descendant_wt,
        relpath="descendant.txt",
        content="descendant\n",
        message="descendant change",
    )

    async with scenario.db.session() as session:
        repo_row = await _require_repository_row(session, scenario)
        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug=_unique_name("test-epic"),
            root_branch="main",
        )
        session.add(epic)
        await session.flush()

        parent = Task(
            epic_id=epic.id,
            title="Parent",
            branch_name=parent_branch,
            worktree_path=str(parent_wt),
            state=TaskState.InProgress,
            merge_ready_at=datetime.now(UTC),
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
        await session.flush()

        descendant = Task(
            epic_id=epic.id,
            title="Descendant",
            branch_name=descendant_branch,
            worktree_path=str(descendant_wt),
            parent_task_id=child.id,
            state=descendant_state,
        )
        session.add(descendant)
        await session.commit()

        return SeededSpineWithDescendant(
            epic_id=epic.id,
            parent_task_id=parent.id,
            child_task_id=child.id,
            descendant_task_id=descendant.id,
            parent_branch=parent_branch,
            child_branch=child_branch,
            descendant_branch=descendant_branch,
            parent_worktree=parent_wt,
            child_worktree=child_wt,
            descendant_worktree=descendant_wt,
        )


async def seed_three_task_chain_in_progress(scenario: Scenario) -> SeededThreeTaskChain:
    """Seed a 3-task chain on branches with worktrees; all tasks InProgress."""
    repo = scenario.repo

    parent_branch = _unique_name("task-parent")
    child_branch = _unique_name("task-child")
    grandchild_branch = _unique_name("task-grandchild")

    parent_wt = repo.create_worktree(branch_name=parent_branch)
    repo.commit_file(
        worktree_path=parent_wt,
        relpath="parent.txt",
        content="parent\n",
        message="parent change",
    )

    child_wt = repo.create_worktree(branch_name=child_branch, from_ref=parent_branch)
    repo.commit_file(
        worktree_path=child_wt,
        relpath="child.txt",
        content="child\n",
        message="child change",
    )

    grandchild_wt = repo.create_worktree(
        branch_name=grandchild_branch, from_ref=child_branch
    )
    repo.commit_file(
        worktree_path=grandchild_wt,
        relpath="grandchild.txt",
        content="grandchild\n",
        message="grandchild change",
    )

    async with scenario.db.session() as session:
        repo_row = await _require_repository_row(session, scenario)
        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug=_unique_name("test-epic"),
            root_branch="main",
        )
        session.add(epic)
        await session.flush()

        parent = Task(
            epic_id=epic.id,
            title="Parent",
            branch_name=parent_branch,
            worktree_path=str(parent_wt),
            state=TaskState.InProgress,
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
        )
        session.add(child)
        await session.flush()

        grandchild = Task(
            epic_id=epic.id,
            title="Grandchild",
            branch_name=grandchild_branch,
            worktree_path=str(grandchild_wt),
            parent_task_id=child.id,
            state=TaskState.InProgress,
        )
        session.add(grandchild)
        await session.commit()

        return SeededThreeTaskChain(
            epic_id=epic.id,
            parent_task_id=parent.id,
            child_task_id=child.id,
            grandchild_task_id=grandchild.id,
            parent_branch=parent_branch,
            child_branch=child_branch,
            grandchild_branch=grandchild_branch,
            parent_worktree=parent_wt,
            child_worktree=child_wt,
            grandchild_worktree=grandchild_wt,
        )


async def seed_active_spine_with_mid_plan_rebase_conflict(
    scenario: Scenario,
) -> SeededSpine:
    """Seed an active 2-task spine where child rebase conflicts mid-plan."""
    repo = scenario.repo

    repo.commit_file(
        worktree_path=scenario.ctx.repo_root,
        relpath="conflict.txt",
        content="base\n",
        message="base conflict seed",
    )

    parent_branch = _unique_name("task-parent")
    child_branch = _unique_name("task-child")

    parent_wt = repo.create_worktree(branch_name=parent_branch, from_ref="main")
    repo.commit_file(
        worktree_path=parent_wt,
        relpath="conflict.txt",
        content="parent-1\n",
        message="parent conflict 1",
    )

    child_wt = repo.create_worktree(branch_name=child_branch, from_ref=parent_branch)
    repo.commit_file(
        worktree_path=child_wt,
        relpath="conflict.txt",
        content="child\n",
        message="child conflict",
    )

    repo.commit_file(
        worktree_path=parent_wt,
        relpath="conflict.txt",
        content="parent-2\n",
        message="parent conflict 2",
    )

    async with scenario.db.session() as session:
        repo_row = await _require_repository_row(session, scenario)
        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug=_unique_name("test-epic"),
            root_branch="main",
        )
        session.add(epic)
        await session.flush()

        parent = Task(
            epic_id=epic.id,
            title="Parent",
            branch_name=parent_branch,
            worktree_path=str(parent_wt),
            state=TaskState.InProgress,
            merge_ready_at=datetime.now(UTC),
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
