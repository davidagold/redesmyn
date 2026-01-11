from __future__ import annotations

import re

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

    assets_js_paths = re.findall(r'(/assets/[^"\']+\.js)', response.text)
    assert assets_js_paths, "Expected dashboard HTML to reference /assets/*.js"

    assets_js_response = await scenario.app.client.get(assets_js_paths[0])
    assert assets_js_response.status_code == 200
    assert "text/html" not in assets_js_response.headers.get("content-type", "")

    spa_response = await scenario.app.client.get(
        "/graph/some-epic", follow_redirects=True
    )
    assert spa_response.status_code == 200
    assert "<title>Redesmyn</title>" in spa_response.text

    status_response = await scenario.app.client.get("/v1/status")
    assert status_response.status_code == 200
    assert status_response.headers["content-type"].startswith("application/json")
    assert "<title>Redesmyn</title>" not in status_response.text
