from __future__ import annotations

import pytest
from sqlalchemy import delete, select

from redesmyn.db import RepoExecutorLease, Repository
from redesmyn.repo_identity import RepoKey
from redesmyn.schemas.core import EpicGraphResponse

from tests.scenarios.scenario import Scenario
from tests.scenarios.variants import seed_merged_parent


@pytest.mark.integration
async def test_epic_graph_reacquires_primary_lease_in_local_mode(
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
        repo_key = RepoKey(workspace_id=repo.workspace_id, repo_id=repo.repo_id)
        await session.execute(
            delete(RepoExecutorLease).where(
                RepoExecutorLease.workspace_id == repo_key.workspace_id,
                RepoExecutorLease.repo_id == repo_key.repo_id,
            )
        )
        await session.commit()

    response = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert response.status_code == 200
    graph = EpicGraphResponse.model_validate(response.json())
    assert graph.repo_executor is not None
    assert graph.repo_executor.primary_host_key == scenario.host_key
