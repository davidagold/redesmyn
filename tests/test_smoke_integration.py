from __future__ import annotations

import pytest
from sqlalchemy import select

from redesmyn.db import Repository

from tests.scenarios.scenario import Scenario


@pytest.mark.integration
async def test_scenario_smoke_healthz_and_repo_initialized(scenario: Scenario) -> None:
    response = await scenario.app.client.get("/v1/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    async with scenario.db.session() as session:
        repo = await session.scalar(
            select(Repository).where(
                Repository.repo_root == str(scenario.ctx.repo_root)
            )
        )
        assert repo is not None
