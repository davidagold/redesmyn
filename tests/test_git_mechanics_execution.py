from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select

from redesmyn.db import Task
from redesmyn.domain.enums import TaskState
from redesmyn.git_mechanics_v0 import (
    MergePlanError,
    MergeRunStepUpdate,
    build_merge_cascade_plan,
    build_restack_plan,
    execute_merge_cascade_plan,
    execute_restack_plan,
)
from redesmyn.repo import (
    GitCommandError,
    git_has_in_progress_operation,
    git_is_ancestor,
)

from tests.scenarios.scenario import Scenario
from tests.scenarios.variants import (
    seed_active_spine_with_mid_plan_rebase_conflict,
    seed_conflicted_merge_run,
    seed_three_task_chain_in_progress,
)


@pytest.mark.integration
async def test_execute_merge_conflict_marks_blocked_and_can_resume(
    scenario: Scenario,
) -> None:
    seeded = await seed_active_spine_with_mid_plan_rebase_conflict(scenario)

    plan = await build_merge_cascade_plan(
        ctx=scenario.ctx,
        sessionmaker=scenario.db.sessionmaker,
        task_id=seeded.child_task_id,
        run_id=uuid4().hex,
        scope="spine",
        restack_mode="strict",
        force=False,
    )

    updates: list[MergeRunStepUpdate] = []

    async def update_run(update: MergeRunStepUpdate) -> None:
        updates.append(update)

    with pytest.raises(GitCommandError):
        await execute_merge_cascade_plan(
            ctx=scenario.ctx,
            sessionmaker=scenario.db.sessionmaker,
            plan=plan,
            allow_running=True,
            update_run=update_run,
        )

    failed = next(u for u in updates if u.phase == "failed")
    assert failed.blocked is True
    assert failed.step.kind == "rebase"
    assert failed.step.branch_name == seeded.child_branch
    assert plan.steps[failed.step_index] == failed.step
    assert git_has_in_progress_operation(seeded.child_worktree)

    with pytest.raises(MergePlanError) as excinfo:
        await build_merge_cascade_plan(
            ctx=scenario.ctx,
            sessionmaker=scenario.db.sessionmaker,
            task_id=seeded.child_task_id,
            run_id=uuid4().hex,
            scope="spine",
            restack_mode="strict",
            force=False,
        )
    message = str(excinfo.value)
    assert (
        "in-progress git operation" in message
        or f"Worktree is not on {seeded.child_branch}" in message
    )

    # Manual resolution, simulating user intervention between "blocked" and "resume".
    conflict_path = seeded.child_worktree / "conflict.txt"
    conflict_path.write_text("resolved\n", encoding="utf-8")
    scenario.repo.git(["add", "conflict.txt"], cwd=seeded.child_worktree)
    scenario.repo.git(
        ["-c", "core.editor=true", "rebase", "--continue"], cwd=seeded.child_worktree
    )
    assert not git_has_in_progress_operation(seeded.child_worktree)

    resumed_plan = await build_merge_cascade_plan(
        ctx=scenario.ctx,
        sessionmaker=scenario.db.sessionmaker,
        task_id=seeded.child_task_id,
        run_id=plan.run_id,
        scope="spine",
        restack_mode="strict",
        force=False,
    )
    await execute_merge_cascade_plan(
        ctx=scenario.ctx,
        sessionmaker=scenario.db.sessionmaker,
        plan=resumed_plan,
        allow_running=True,
        start_at_step_index=failed.step_index,
    )

    assert git_is_ancestor(scenario.ctx.repo_root, seeded.child_branch, "main")

    async with scenario.db.session() as session:
        rows = list(
            await session.scalars(
                select(Task).where(
                    Task.id.in_([seeded.parent_task_id, seeded.child_task_id])
                )
            )
        )
        by_id = {t.id: t for t in rows}
        assert by_id[seeded.parent_task_id].state == TaskState.Done
        assert by_id[seeded.child_task_id].state == TaskState.Done
        assert by_id[seeded.parent_task_id].merge_ready_at is None
        assert by_id[seeded.child_task_id].merge_ready_at is None


@pytest.mark.integration
async def test_resume_revalidates_dirty_worktrees_before_continuing(
    scenario: Scenario,
) -> None:
    seeded = await seed_conflicted_merge_run(scenario)
    assert git_has_in_progress_operation(seeded.child_worktree)

    with pytest.raises(MergePlanError) as excinfo:
        await build_merge_cascade_plan(
            ctx=scenario.ctx,
            sessionmaker=scenario.db.sessionmaker,
            task_id=seeded.child_task_id,
            run_id=uuid4().hex,
            scope="spine",
            restack_mode="strict",
            force=False,
        )

    message = str(excinfo.value)
    assert (
        "in-progress git operation" in message
        or f"Worktree is not on {seeded.child_branch}" in message
    )


@pytest.mark.integration
async def test_execute_restack_rebases_target_and_descendants(
    scenario: Scenario,
) -> None:
    seeded = await seed_three_task_chain_in_progress(scenario)

    scenario.repo.commit_file(
        worktree_path=scenario.ctx.repo_root,
        relpath="base.txt",
        content="new base\n",
        message="base update",
    )
    main_head = scenario.repo.git(["rev-parse", "HEAD"], cwd=scenario.ctx.repo_root)

    plan = await build_restack_plan(
        ctx=scenario.ctx,
        sessionmaker=scenario.db.sessionmaker,
        task_id=seeded.parent_task_id,
        run_id=uuid4().hex,
        scope="descendants",
    )
    await execute_restack_plan(
        ctx=scenario.ctx,
        sessionmaker=scenario.db.sessionmaker,
        plan=plan,
        allow_running=True,
    )

    assert git_is_ancestor(scenario.ctx.repo_root, main_head, seeded.parent_branch)
    assert git_is_ancestor(scenario.ctx.repo_root, main_head, seeded.child_branch)
    assert git_is_ancestor(scenario.ctx.repo_root, main_head, seeded.grandchild_branch)
