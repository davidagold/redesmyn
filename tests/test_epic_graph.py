from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from redesmyn.db import (
    Agent,
    AgentSession,
    GitTrunkTimelineByInstance,
    MergeRun,
    RepoExecutorLease,
    Repository,
)
from redesmyn.domain.enums import AgentStatus, MergeRunStatus
from redesmyn.schemas.core import EpicGraphResponse

from tests.scenarios.scenario import Scenario
from tests.scenarios.variants import seed_merged_parent


@pytest.mark.integration
async def test_epic_graph_includes_expected_task_nodes_and_parent_links(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)

    response = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert response.status_code == 200
    graph = EpicGraphResponse.model_validate(response.json())

    tasks_by_id = {task.id: task for task in graph.tasks}
    assert set(tasks_by_id.keys()) == {seeded.parent_task_id, seeded.child_task_id}
    assert tasks_by_id[seeded.parent_task_id].parent_task_id is None
    assert tasks_by_id[seeded.child_task_id].parent_task_id == seeded.parent_task_id


@pytest.mark.integration
async def test_epic_graph_includes_agent_session_overlay_fields(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)

    async with scenario.db.session() as session:
        agent_1 = Agent(display_name="agent-1", status=AgentStatus.Running)
        agent_2 = Agent(display_name="agent-2", status=AgentStatus.Error)
        session.add_all([agent_1, agent_2])
        await session.flush()

        session.add_all(
            [
                AgentSession(
                    agent_id=agent_1.id,
                    task_id=seeded.child_task_id,
                    status=AgentStatus.Running,
                    started_at=datetime.now(UTC),
                    ended_at=datetime.now(UTC),
                ),
                AgentSession(
                    agent_id=agent_2.id,
                    task_id=seeded.child_task_id,
                    status=AgentStatus.Error,
                    started_at=datetime.now(UTC),
                ),
            ]
        )
        await session.commit()

    response = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert response.status_code == 200
    graph = EpicGraphResponse.model_validate(response.json())

    assert len(graph.agent_sessions) == 1
    latest = graph.agent_sessions[0]
    assert latest.task_id == seeded.child_task_id
    assert latest.status == AgentStatus.Error
    assert latest.agent_name == "agent-2"


@pytest.mark.integration
async def test_epic_graph_includes_merge_run_overlay_fields(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)

    blocked_run_id = "run-blocked"
    async with scenario.db.session() as session:
        session.add_all(
            [
                MergeRun(
                    run_id=blocked_run_id,
                    epic_id=seeded.epic_id,
                    requested_task_id=seeded.child_task_id,
                    canonical=True,
                    status=MergeRunStatus.Blocked,
                    blocked_step_index=0,
                    blocked_step_kind="rebase",
                    blocked_task_id=seeded.child_task_id,
                    blocked_branch_name=seeded.child_branch,
                    blocked_worktree_path=str(seeded.child_worktree),
                    blocked_error="blocked for test",
                ),
                MergeRun(
                    run_id="run-succeeded",
                    epic_id=seeded.epic_id,
                    requested_task_id=seeded.child_task_id,
                    canonical=True,
                    status=MergeRunStatus.Succeeded,
                ),
                MergeRun(
                    run_id="run-non-canonical",
                    epic_id=seeded.epic_id,
                    requested_task_id=seeded.child_task_id,
                    canonical=False,
                    status=MergeRunStatus.Running,
                ),
            ]
        )
        await session.commit()

    response = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert response.status_code == 200
    graph = EpicGraphResponse.model_validate(response.json())

    assert [run.run_id for run in graph.merge_runs] == [blocked_run_id]
    blocked = graph.merge_runs[0]
    assert blocked.status == MergeRunStatus.Blocked
    assert blocked.blocked_task_id == seeded.child_task_id
    assert blocked.blocked_branch_name == seeded.child_branch


@pytest.mark.integration
async def test_epic_graph_includes_trunk_timeline_when_available(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)

    async with scenario.db.session() as session:
        repo = await session.scalar(
            select(Repository).where(
                Repository.repo_root == str(scenario.ctx.repo_root)
            )
        )
        assert repo is not None

        lease = await session.get(RepoExecutorLease, (repo.workspace_id, repo.repo_id))
        if lease is None:
            session.add(
                RepoExecutorLease(
                    workspace_id=repo.workspace_id,
                    repo_id=repo.repo_id,
                    host_key=scenario.host_key,
                    lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
                )
            )
        else:
            lease.host_key = scenario.host_key
            lease.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5)

        session.add(
            GitTrunkTimelineByInstance(
                epic_id=seeded.epic_id,
                host_key=scenario.host_key,
                data={
                    "base_sha": "a" * 40,
                    "base_commit": {"sha": "a" * 40, "title": "base"},
                    "commits_before": [],
                    "commits_after": [{"sha": "b" * 40, "title": "after"}],
                    "has_more_before": False,
                    "has_more_after": False,
                },
            )
        )
        await session.commit()

    response = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert response.status_code == 200
    graph = EpicGraphResponse.model_validate(response.json())

    assert graph.trunk is not None
    assert graph.trunk.base_sha == "a" * 40
    assert [c.sha for c in graph.trunk.commits_after] == ["b" * 40]
