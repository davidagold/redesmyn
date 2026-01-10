from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from redesmyn.git_subprocess import run_git

GITHUB_API_BASE_URL = "https://api.github.com"

_SSH_GIT_REMOTE_RE = re.compile(r"^(?P<user>[^@]+)@(?P<host>[^:]+):(?P<path>.+)$")


@dataclass(frozen=True, slots=True)
class GitHubViewer:
    login: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class GitHubRepoInfo:
    owner: str
    repo: str
    private: bool

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass(frozen=True, slots=True)
class GitHubAuthStatus:
    connected: bool
    connected_at: datetime | None
    viewer: GitHubViewer | None = None
    granted_scopes: frozenset[str] | None = None
    repo: GitHubRepoInfo | None = None
    pr_scopes_ok: bool | None = None
    missing_pr_scopes: tuple[str, ...] | None = None
    warning: bool = False
    warning_reason: str | None = None


def _split_scopes(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    scopes = {p for p in parts if p}
    return frozenset(scopes)


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    proc = run_git(args, cwd=repo_root, timeout_s=2)
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    return value or None


def _try_parse_github_owner_repo(remote: str) -> tuple[str, str] | None:
    value = remote.strip()
    match = _SSH_GIT_REMOTE_RE.match(value)
    if match:
        host = match.group("host").strip().lower()
        path = match.group("path").strip().lstrip("/")
        if host != "github.com":
            return None
        if path.endswith(".git"):
            path = path[: -len(".git")]
        if "/" not in path:
            return None
        owner, repo = path.split("/", 1)
        owner = owner.strip()
        repo = repo.strip()
        if owner and repo:
            return owner, repo
        return None

    for prefix in ("https://", "http://", "ssh://", "git://"):
        if value.startswith(prefix):
            remainder = value[len(prefix) :]
            remainder = remainder.split("#", 1)[0].split("?", 1)[0]
            remainder = remainder.rstrip("/")
            if remainder.endswith(".git"):
                remainder = remainder[: -len(".git")]
            if "/" not in remainder:
                return None
            host, rest = remainder.split("/", 1)
            if host.strip().lower() != "github.com":
                return None
            rest = rest.lstrip("/")
            if "/" not in rest:
                return None
            owner, repo = rest.split("/", 1)
            owner = owner.strip()
            repo = repo.strip()
            if owner and repo:
                return owner, repo
            return None

    return None


def try_parse_current_repo_owner_repo(repo_root: Path) -> tuple[str, str] | None:
    remote = _run_git(repo_root, ["config", "--get", "remote.origin.url"])
    if remote is None:
        remote = _run_git(repo_root, ["config", "--get", "remote.upstream.url"])
    if remote is None:
        return None
    return _try_parse_github_owner_repo(remote)


async def fetch_viewer_and_scopes(
    *, access_token: str
) -> tuple[GitHubViewer, frozenset[str] | None]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GITHUB_API_BASE_URL}/user",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        resp.raise_for_status()
        payload: dict[str, object] = resp.json()

    login = payload.get("login")
    if not isinstance(login, str) or not login:
        raise ValueError("GitHub /user response missing login")
    name = payload.get("name")
    viewer = GitHubViewer(login=login, name=name if isinstance(name, str) else None)
    granted_scopes = _split_scopes(resp.headers.get("X-OAuth-Scopes"))
    return viewer, granted_scopes


async def fetch_repo_info(
    *, owner: str, repo: str, access_token: str | None
) -> GitHubRepoInfo | None:
    headers = {"Accept": "application/vnd.github+json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}",
            headers=headers,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload: dict[str, object] = resp.json()

    private = payload.get("private")
    if not isinstance(private, bool):
        raise ValueError("GitHub repo response missing private")
    return GitHubRepoInfo(owner=owner, repo=repo, private=private)


def _scopes_satisfy_pr(
    *,
    granted_scopes: frozenset[str] | None,
    repo: GitHubRepoInfo | None,
) -> tuple[bool | None, tuple[str, ...] | None, str | None]:
    if granted_scopes is None:
        return None, None, "Granted scopes unknown (missing X-OAuth-Scopes header)"
    if repo is None:
        return None, None, "GitHub repo unknown (could not parse local git remote)"

    if repo.private:
        required = {"repo"}
    else:
        required = {"repo", "public_repo"}

    missing = tuple(sorted(required - set(granted_scopes)))
    return (len(missing) == 0), (missing or None), None


async def github_auth_status(
    *,
    access_token: str | None,
    connected_at: datetime | None,
    repo_root: Path | None,
) -> GitHubAuthStatus:
    if not access_token:
        return GitHubAuthStatus(connected=False, connected_at=None)

    try:
        viewer, granted_scopes = await fetch_viewer_and_scopes(
            access_token=access_token
        )
    except httpx.HTTPStatusError:
        return GitHubAuthStatus(connected=False, connected_at=connected_at)

    repo: GitHubRepoInfo | None = None
    if repo_root is not None:
        owner_repo = try_parse_current_repo_owner_repo(repo_root)
        if owner_repo is not None:
            owner, repo_name = owner_repo
            repo = await fetch_repo_info(
                owner=owner, repo=repo_name, access_token=access_token
            ) or await fetch_repo_info(owner=owner, repo=repo_name, access_token=None)

    pr_scopes_ok, missing_pr_scopes, reason = _scopes_satisfy_pr(
        granted_scopes=granted_scopes, repo=repo
    )
    warning = pr_scopes_ok is False
    warning_reason = None
    if warning:
        warning_reason = f"Missing scopes for PR creation in {repo.full_name if repo else 'this repo'}"
    elif pr_scopes_ok is None and reason is not None:
        warning_reason = reason

    return GitHubAuthStatus(
        connected=True,
        connected_at=connected_at,
        viewer=viewer,
        granted_scopes=granted_scopes,
        repo=repo,
        pr_scopes_ok=pr_scopes_ok,
        missing_pr_scopes=missing_pr_scopes,
        warning=warning,
        warning_reason=warning_reason,
    )
