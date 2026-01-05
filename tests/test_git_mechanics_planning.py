from __future__ import annotations

from uuid import uuid4

import pytest

from redesmyn.git_mechanics_v0 import (
    MergeBlockedByRunningAgents,
    MergePlanError,
    build_merge_cascade_plan,
    execute_merge_cascade_plan,
)

from tests.scenarios.scenario import Scenario
from tests.scenarios.variants import (
    seed_active_spine_with_descendant_worktree,
    seed_merged_parent,
    seed_running_agent,
)


@pytest.mark.integration
async def test_merge_plan_uses_epic_base_when_parent_is_merged(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)

    plan = await build_merge_cascade_plan(
        ctx=scenario.ctx,
        sessionmaker=scenario.db.sessionmaker,
        task_id=seeded.child_task_id,
        run_id=uuid4().hex,
        scope="spine",
        restack_mode="strict",
        force=False,
    )

    child = plan.tasks[seeded.child_task_id]
    assert child.upstream_ref == "main"
    assert child.upstream_ref != seeded.parent_branch


@pytest.mark.integration
async def test_merge_plan_requires_base_worktree_present(scenario: Scenario) -> None:
    seeded = await seed_merged_parent(scenario)

    scenario.repo.git(["checkout", "-b", "off-main"], cwd=scenario.ctx.repo_root)

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
    assert "No worktree has" in message
    assert "main" in message


@pytest.mark.integration
async def test_merge_plan_collects_running_agents_and_execution_blocks_without_allow_running(
    scenario: Scenario,
) -> None:
    seeded = await seed_running_agent(scenario)

    plan = await build_merge_cascade_plan(
        ctx=scenario.ctx,
        sessionmaker=scenario.db.sessionmaker,
        task_id=seeded.child_task_id,
        run_id=uuid4().hex,
        scope="spine",
        restack_mode="strict",
        force=False,
    )
    assert plan.running_agents

    with pytest.raises(MergeBlockedByRunningAgents):
        await execute_merge_cascade_plan(
            ctx=scenario.ctx,
            sessionmaker=scenario.db.sessionmaker,
            plan=plan,
            allow_running=False,
        )


@pytest.mark.integration
async def test_execute_merge_then_restack_orders_steps_correctly(
    scenario: Scenario,
) -> None:
    seeded = await seed_active_spine_with_descendant_worktree(scenario)

    plan = await build_merge_cascade_plan(
        ctx=scenario.ctx,
        sessionmaker=scenario.db.sessionmaker,
        task_id=seeded.child_task_id,
        run_id=uuid4().hex,
        scope="descendants",
        restack_mode="merge_then_restack",
        force=False,
    )

    actual = [(step.kind, step.branch_name) for step in plan.steps]
    assert actual == [
        ("rebase", seeded.parent_branch),
        ("rebase", seeded.child_branch),
        ("merge_ff", seeded.parent_branch),
        ("merge_ff", seeded.child_branch),
        ("rebase", seeded.descendant_branch),
    ]
