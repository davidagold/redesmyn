from __future__ import annotations

import pytest

from tests.scenarios.scenario import Scenario


@pytest.mark.integration
async def test_dashboard_assets_are_served_without_repo_dashboard_dir(
    scenario: Scenario,
) -> None:
    response = await scenario.app.client.get("/")
    assert response.status_code == 200
    assert "<title>Redesmyn</title>" in response.text
    assert 'id="root"' in response.text

    spa_response = await scenario.app.client.get(
        "/graph/some-epic", follow_redirects=True
    )
    assert spa_response.status_code == 200
    assert "<title>Redesmyn</title>" in spa_response.text
