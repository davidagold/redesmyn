from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

import redesmyn.api as api_module
from redesmyn.db import Epic, Repository, Task
from redesmyn.integrations.github_credentials import GitHubCredentials
from redesmyn.integrations.github_pr import GitHubPullRequestInfo, GitHubPullRequestRef
from redesmyn.integrations import github_pr_actions
from tests.scenarios.scenario import Scenario
from tests.scenarios.seeds.git import seed_merged_parent


@dataclass(frozen=True, slots=True)
class _DummyGithubStore:
    creds: GitHubCredentials

    def get(self) -> GitHubCredentials | None:
        return self.creds

    def set(self, credentials: GitHubCredentials) -> None:  # pragma: no cover
        raise AssertionError("not expected")

    def clear(self) -> None:  # pragma: no cover
        raise AssertionError("not expected")


@pytest.mark.integration
async def test_open_task_github_pr_pushes_and_creates(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    seeded = await seed_merged_parent(scenario)

    async with scenario.db.session() as session:
        epic_row = await session.get(Epic, seeded.epic_id)
        assert epic_row is not None
        epic_row.github_repo_host = "github.com"
        epic_row.github_repo_owner = "octo"
        epic_row.github_repo_name = "repo"
        session.add(epic_row)
        await session.commit()

    store = _DummyGithubStore(
        creds=GitHubCredentials(
            access_token="test-token",
            token_type="Bearer",
            connected_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )
    monkeypatch.setattr(api_module, "default_github_credential_store", lambda: store)

    pushes: list[tuple[str, bool]] = []

    def fake_git_push(
        _repo_root: Path,
        *,
        remote: str,
        branch_name: str,
        set_upstream: bool = False,
        force_with_lease: bool = False,
    ) -> None:
        assert remote == "origin"
        assert set_upstream is True
        pushes.append((branch_name, force_with_lease))

    monkeypatch.setattr(github_pr_actions, "git_push", fake_git_push)

    async def fake_detect_pull_request_for_branch(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        github_pr_actions,
        "detect_pull_request_for_branch",
        fake_detect_pull_request_for_branch,
    )

    async def fake_create_pull_request(
        *,
        owner: str,
        repo: str,
        title: str,
        body: str | None,
        head_branch: str,
        base_branch: str,
        access_token: str,
    ) -> GitHubPullRequestInfo:
        assert owner == "octo"
        assert repo == "repo"
        assert access_token == "test-token"
        assert title == "Child"
        assert body is None
        assert head_branch == seeded.child_branch
        assert base_branch == "main"
        return GitHubPullRequestInfo(
            ref=GitHubPullRequestRef(owner="octo", repo="repo", number=123),
            url="https://github.com/octo/repo/pull/123",
        )

    monkeypatch.setattr(
        github_pr_actions, "create_pull_request", fake_create_pull_request
    )

    resp = await scenario.app.client.post(
        f"/v1/tasks/{seeded.child_task_id}/github/pr/open"
    )
    resp.raise_for_status()

    payload = resp.json()
    assert payload["prId"] == "octo/repo#123"
    assert payload["url"] == "https://github.com/octo/repo/pull/123"
    assert payload["task"]["id"] == seeded.child_task_id
    assert payload["task"]["githubPrId"] == "octo/repo#123"

    assert pushes == [(seeded.child_branch, False)]

    async with scenario.db.session() as session:
        task_row = await session.get(Task, seeded.child_task_id)
        assert task_row is not None
        assert task_row.github_pr_id == "octo/repo#123"


@pytest.mark.integration
async def test_open_task_github_pr_falls_back_when_base_branch_missing_locally(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    async with scenario.db.session() as session:
        repo_row = await session.scalar(
            select(Repository).where(
                Repository.repo_root == str(scenario.ctx.repo_root)
            )
        )
        assert repo_row is not None

        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug="test-epic",
            root_branch="main",
            github_repo_host="github.com",
            github_repo_owner="octo",
            github_repo_name="repo",
        )
        session.add(epic)
        await session.flush()

        parent = Task(
            epic_id=epic.id,
            title="Parent",
            branch_name="missing-parent-branch",
        )
        session.add(parent)
        await session.flush()

        child = Task(
            epic_id=epic.id,
            title="Child",
            branch_name="missing-child-branch",
            parent_task_id=parent.id,
        )
        session.add(child)
        await session.commit()
        child_id = child.id

    store = _DummyGithubStore(
        creds=GitHubCredentials(
            access_token="test-token",
            token_type="Bearer",
            connected_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )
    monkeypatch.setattr(api_module, "default_github_credential_store", lambda: store)

    pushes: list[str] = []

    def fake_git_push(
        _repo_root: Path,
        *,
        remote: str,
        branch_name: str,
        set_upstream: bool = False,
        force_with_lease: bool = False,
    ) -> None:
        assert remote == "origin"
        assert set_upstream is True
        assert force_with_lease is False
        pushes.append(branch_name)

    monkeypatch.setattr(github_pr_actions, "git_push", fake_git_push)

    async def fake_detect_pull_request_for_branch(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        github_pr_actions,
        "detect_pull_request_for_branch",
        fake_detect_pull_request_for_branch,
    )

    async def fake_create_pull_request(
        *,
        owner: str,
        repo: str,
        title: str,
        body: str | None,
        head_branch: str,
        base_branch: str,
        access_token: str,
    ) -> GitHubPullRequestInfo:
        assert owner == "octo"
        assert repo == "repo"
        assert access_token == "test-token"
        assert title == "Child"
        assert body is None
        assert head_branch == "missing-child-branch"
        assert base_branch == "main"
        return GitHubPullRequestInfo(
            ref=GitHubPullRequestRef(owner="octo", repo="repo", number=456),
            url="https://github.com/octo/repo/pull/456",
        )

    monkeypatch.setattr(
        github_pr_actions, "create_pull_request", fake_create_pull_request
    )

    resp = await scenario.app.client.post(f"/v1/tasks/{child_id}/github/pr/open")
    resp.raise_for_status()

    payload = resp.json()
    assert payload["prId"] == "octo/repo#456"
    assert payload["task"]["githubPrId"] == "octo/repo#456"
    assert pushes == ["missing-child-branch"]
