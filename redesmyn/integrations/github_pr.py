from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from redesmyn.integrations.github_status import GITHUB_API_BASE_URL


class GitHubPullRequestError(RuntimeError):
    pass


_PR_ID_RE = re.compile(r"^(?P<owner>[^/]+)/(?P<repo>[^#]+)#(?P<number>\d+)$")


@dataclass(frozen=True, slots=True)
class GitHubPullRequestRef:
    owner: str
    repo: str
    number: int

    def to_id(self) -> str:
        return f"{self.owner}/{self.repo}#{self.number}"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}/pull/{self.number}"

    @staticmethod
    def parse(value: str) -> "GitHubPullRequestRef":
        match = _PR_ID_RE.match(value.strip())
        if not match:
            raise ValueError(
                f"Invalid GitHub PR id: {value!r} (expected 'owner/repo#<number>')"
            )
        number = int(match.group("number"))
        if number <= 0:
            raise ValueError(f"Invalid GitHub PR id: {value!r} (number must be > 0)")
        return GitHubPullRequestRef(
            owner=match.group("owner"),
            repo=match.group("repo"),
            number=number,
        )


@dataclass(frozen=True, slots=True)
class GitHubPullRequestInfo:
    ref: GitHubPullRequestRef
    url: str
    state: str | None = None
    title: str | None = None


def _headers(access_token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {access_token}",
    }


def _parse_pr_payload(
    owner: str, repo: str, payload: dict[str, object]
) -> GitHubPullRequestInfo:
    number = payload.get("number")
    if not isinstance(number, int) or number <= 0:
        raise GitHubPullRequestError("GitHub PR response missing number")
    url = payload.get("html_url")
    if not isinstance(url, str) or not url:
        url = f"https://github.com/{owner}/{repo}/pull/{number}"
    state = payload.get("state")
    title = payload.get("title")
    return GitHubPullRequestInfo(
        ref=GitHubPullRequestRef(owner=owner, repo=repo, number=number),
        url=url,
        state=state if isinstance(state, str) else None,
        title=title if isinstance(title, str) else None,
    )


async def detect_pull_request_for_branch(
    *,
    owner: str,
    repo: str,
    head_branch: str,
    access_token: str,
) -> GitHubPullRequestInfo | None:
    """
    Best-effort PR detection for same-repo branches.

    Prefers open PRs when multiple are present, but will fall back to the most
    recently updated PR (v0).
    """
    head = f"{owner}:{head_branch}"
    params = {"head": head, "state": "all", "per_page": "30"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/pulls",
            headers=_headers(access_token),
            params=params,
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (resp.text or "").strip()
            raise GitHubPullRequestError(
                f"GitHub PR detection failed ({resp.status_code}): {detail or e}"
            ) from e
        raw = resp.json()

    if not isinstance(raw, list):
        raise GitHubPullRequestError("GitHub pulls list response is not a list")

    candidates: list[GitHubPullRequestInfo] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        candidates.append(_parse_pr_payload(owner, repo, item))

    if not candidates:
        return None

    open_prs = [pr for pr in candidates if pr.state == "open"]
    if open_prs:
        return open_prs[0]
    return candidates[0]


async def create_pull_request(
    *,
    owner: str,
    repo: str,
    title: str,
    body: str | None,
    head_branch: str,
    base_branch: str,
    access_token: str,
) -> GitHubPullRequestInfo:
    payload: dict[str, object] = {
        "title": title,
        "head": f"{owner}:{head_branch}",
        "base": base_branch,
    }
    if body is not None:
        payload["body"] = body

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/pulls",
            headers=_headers(access_token),
            json=payload,
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (resp.text or "").strip()
            raise GitHubPullRequestError(
                f"GitHub PR create failed ({resp.status_code}): {detail or e}"
            ) from e
        raw = resp.json()

    if not isinstance(raw, dict):
        raise GitHubPullRequestError("GitHub PR create response is not an object")
    return _parse_pr_payload(owner, repo, raw)
