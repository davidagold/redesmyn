from __future__ import annotations

from pathlib import Path

import pytest

from redesmyn.orchestration_config import repo_config_path
from tests.scenarios.scenario import Scenario


@pytest.mark.integration
async def test_github_config_round_trips_via_api(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    resp = await scenario.app.client.get("/v1/github/config")
    resp.raise_for_status()
    assert resp.json()["autoForcePush"] is False

    resp = await scenario.app.client.post(
        "/v1/github/config",
        json={"autoForcePush": True},
    )
    resp.raise_for_status()
    assert resp.json()["autoForcePush"] is True

    resp = await scenario.app.client.get("/v1/github/config")
    resp.raise_for_status()
    assert resp.json()["autoForcePush"] is True

    raw = repo_config_path(scenario.ctx).read_text(encoding="utf-8")
    assert "github" in raw
    assert "auto_force_push = true" in raw


@pytest.mark.integration
async def test_github_status_includes_auto_force_push_when_disconnected(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    await scenario.app.client.post(
        "/v1/github/config",
        json={"autoForcePush": True},
    )
    resp = await scenario.app.client.get("/v1/github/status")
    resp.raise_for_status()
    payload = resp.json()
    assert payload["connected"] is False
    assert payload["autoForcePush"] is True
