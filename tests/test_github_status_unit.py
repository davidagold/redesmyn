from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from redesmyn.integrations import github_status
from redesmyn.integrations.github_status import GitHubRepoInfo, GitHubViewer


@pytest.mark.unit
def test_try_parse_github_owner_repo_handles_ssh_remote() -> None:
    assert github_status._try_parse_github_owner_repo(
        "git@github.com:octo/repo.git"
    ) == (
        "octo",
        "repo",
    )


@pytest.mark.unit
def test_try_parse_github_owner_repo_handles_https_remote() -> None:
    assert github_status._try_parse_github_owner_repo(
        "https://github.com/octo/repo"
    ) == ("octo", "repo")


@pytest.mark.unit
def test_scopes_satisfy_pr_private_repo_requires_repo_scope() -> None:
    repo = GitHubRepoInfo(owner="octo", repo="secret", private=True)
    ok, missing, reason = github_status._scopes_satisfy_pr(
        granted_scopes=frozenset({"read:user"}), repo=repo
    )
    assert reason is None
    assert ok is False
    assert missing == ("repo",)


@pytest.mark.unit
def test_scopes_satisfy_pr_public_repo_accepts_public_repo_scope() -> None:
    repo = GitHubRepoInfo(owner="octo", repo="public", private=False)
    ok, missing, reason = github_status._scopes_satisfy_pr(
        granted_scopes=frozenset({"public_repo"}), repo=repo
    )
    assert reason is None
    assert ok is True
    assert missing is None


@pytest.mark.unit
async def test_fetch_viewer_and_scopes_parses_oauth_scopes_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == httpx.URL("https://api.github.com/user")
        assert request.headers.get("Authorization") == "Bearer test-token"
        return httpx.Response(
            200,
            json={"login": "octo", "name": "Octo Cat"},
            headers={"X-OAuth-Scopes": "repo, read:user"},
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(github_status.httpx, "AsyncClient", patched_async_client)

    viewer, scopes = await github_status.fetch_viewer_and_scopes(
        access_token="test-token"
    )
    assert viewer == GitHubViewer(login="octo", name="Octo Cat")
    assert scopes == frozenset({"repo", "read:user"})


@pytest.mark.unit
async def test_github_auth_status_disconnected_when_token_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "https://api.github.com/user")
    response = httpx.Response(401, request=request)

    async def fake_fetch_viewer_and_scopes(
        *, access_token: str
    ) -> tuple[GitHubViewer, frozenset[str] | None]:
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(
        github_status, "fetch_viewer_and_scopes", fake_fetch_viewer_and_scopes
    )

    connected_at = datetime(2025, 1, 1, tzinfo=UTC)
    status = await github_status.github_auth_status(
        access_token="bad-token",
        connected_at=connected_at,
        repo_root=None,
    )
    assert status.connected is False
    assert status.connected_at == connected_at


@pytest.mark.unit
async def test_github_auth_status_warns_when_missing_pr_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_fetch_viewer_and_scopes(
        *, access_token: str
    ) -> tuple[GitHubViewer, frozenset[str] | None]:
        return GitHubViewer(login="octo"), frozenset({"read:user"})

    monkeypatch.setattr(
        github_status, "fetch_viewer_and_scopes", fake_fetch_viewer_and_scopes
    )
    monkeypatch.setattr(
        github_status,
        "try_parse_current_repo_owner_repo",
        lambda _repo_root: ("octo", "public"),
    )

    async def fake_fetch_repo_info(
        *, owner: str, repo: str, access_token: str | None
    ) -> GitHubRepoInfo | None:
        assert owner == "octo"
        assert repo == "public"
        return GitHubRepoInfo(owner=owner, repo=repo, private=False)

    monkeypatch.setattr(github_status, "fetch_repo_info", fake_fetch_repo_info)

    status = await github_status.github_auth_status(
        access_token="test-token",
        connected_at=datetime(2025, 1, 1, tzinfo=UTC),
        repo_root=Path("/tmp/repo"),
    )

    assert status.connected is True
    assert status.repo is not None and status.repo.full_name == "octo/public"
    assert status.pr_scopes_ok is False
    assert status.missing_pr_scopes == ("repo", "public_repo")
    assert status.warning is True
    assert status.warning_reason == "Missing scopes for PR creation in octo/public"
