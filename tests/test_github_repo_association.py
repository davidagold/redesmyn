from __future__ import annotations

import pytest
from sqlalchemy import select

from redesmyn.db import Epic, Repository
from redesmyn.integrations.github_repo import parse_github_repo_ref


@pytest.mark.unit
def test_parse_github_repo_ref_supports_common_forms() -> None:
    assert parse_github_repo_ref("owner/repo") == (
        parse_github_repo_ref("https://github.com/owner/repo.git")
    )

    ssh = parse_github_repo_ref("git@github.com:owner/repo.git")
    assert ssh is not None
    assert ssh.host == "github.com"
    assert ssh.owner == "owner"
    assert ssh.repo == "repo"

    https = parse_github_repo_ref("https://github.com/owner/repo.git")
    assert https is not None
    assert https.host == "github.com"
    assert https.owner == "owner"
    assert https.repo == "repo"


@pytest.mark.integration
async def test_epic_github_repo_endpoint_detects_and_overrides(scenario) -> None:
    scenario.repo.git(["remote", "add", "origin", "git@github.com:acme/widgets.git"])

    async with scenario.db.session() as session:
        repo = await session.scalar(select(Repository).limit(1))
        assert repo is not None
        epic = Epic(
            repository_id=repo.id,
            name="GitHub Epic",
            slug="github-epic",
            root_branch="main",
            linear_project_id=None,
            github_repo_host=None,
            github_repo_owner=None,
            github_repo_name=None,
        )
        session.add(epic)
        await session.commit()

    response = await scenario.app.client.get("/v1/epics/github-epic/github/repo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is None
    assert payload["detected"] == {
        "host": "github.com",
        "owner": "acme",
        "repo": "widgets",
    }
    assert payload["effective"] == payload["detected"]

    response = await scenario.app.client.patch(
        "/v1/epics/github-epic/github/repo",
        json={"repo": "other-org/other-repo"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] == {
        "host": "github.com",
        "owner": "other-org",
        "repo": "other-repo",
    }
    assert payload["effective"] == payload["configured"]
