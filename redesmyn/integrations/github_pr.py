from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

import httpx

from redesmyn.integrations.github_status import GITHUB_API_BASE_URL
from urllib.parse import quote


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
    draft: bool | None = None
    merged: bool | None = None
    base_branch: str | None = None
    base_sha: str | None = None
    head_branch: str | None = None
    head_sha: str | None = None


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

    draft = payload.get("draft")
    draft_value = draft if isinstance(draft, bool) else None

    merged_at = payload.get("merged_at")
    merged_value: bool | None
    if merged_at is None:
        merged_value = False if "merged_at" in payload else None
    elif isinstance(merged_at, str):
        merged_value = True
    else:
        merged_value = None

    base_branch: str | None = None
    base_sha: str | None = None
    base = payload.get("base")
    if isinstance(base, dict):
        base_dict = cast(dict[str, object], base)
        ref = base_dict.get("ref")
        if isinstance(ref, str) and ref:
            base_branch = ref
        sha = base_dict.get("sha")
        if isinstance(sha, str) and sha:
            base_sha = sha

    head_branch: str | None = None
    head_sha: str | None = None
    head = payload.get("head")
    if isinstance(head, dict):
        head_dict = cast(dict[str, object], head)
        ref = head_dict.get("ref")
        if isinstance(ref, str) and ref:
            head_branch = ref
        sha = head_dict.get("sha")
        if isinstance(sha, str) and sha:
            head_sha = sha

    return GitHubPullRequestInfo(
        ref=GitHubPullRequestRef(owner=owner, repo=repo, number=number),
        url=url,
        state=state if isinstance(state, str) else None,
        title=title if isinstance(title, str) else None,
        draft=draft_value,
        merged=merged_value,
        base_branch=base_branch,
        base_sha=base_sha,
        head_branch=head_branch,
        head_sha=head_sha,
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


async def fetch_pull_request(
    *,
    owner: str,
    repo: str,
    number: int,
    access_token: str,
) -> GitHubPullRequestInfo:
    if number <= 0:
        raise GitHubPullRequestError("PR number must be > 0")

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/pulls/{number}",
            headers=_headers(access_token),
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (resp.text or "").strip()
            raise GitHubPullRequestError(
                f"GitHub PR fetch failed ({resp.status_code}): {detail or e}"
            ) from e
        raw = resp.json()

    if not isinstance(raw, dict):
        raise GitHubPullRequestError("GitHub PR fetch response is not an object")
    return _parse_pr_payload(owner, repo, raw)


def _quote_path_segment(value: str) -> str:
    # Branch names commonly include slashes (e.g. rn/gpui/T-10-...). GitHub's REST
    # APIs expect these to be URL-encoded when used as path segments.
    return quote(value, safe="")


async def fetch_repo_default_branch(
    *,
    owner: str,
    repo: str,
    access_token: str,
) -> str | None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}",
            headers=_headers(access_token),
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (resp.text or "").strip()
            raise GitHubPullRequestError(
                f"GitHub repo fetch failed ({resp.status_code}): {detail or e}"
            ) from e
        payload = resp.json()

    if not isinstance(payload, dict):
        raise GitHubPullRequestError("GitHub repo fetch response is not an object")
    default_branch = payload.get("default_branch")
    if isinstance(default_branch, str):
        return default_branch.strip() or None
    return None


async def list_repo_branches(
    *,
    owner: str,
    repo: str,
    access_token: str,
    per_page: int = 100,
) -> list[str]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/branches",
            headers=_headers(access_token),
            params={"per_page": str(per_page)},
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (resp.text or "").strip()
            raise GitHubPullRequestError(
                f"GitHub branches list failed ({resp.status_code}): {detail or e}"
            ) from e
        payload = resp.json()

    if not isinstance(payload, list):
        raise GitHubPullRequestError("GitHub branches list response is not a list")
    names: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


async def fetch_branch_head_sha(
    *,
    owner: str,
    repo: str,
    branch: str,
    access_token: str,
) -> str:
    if not branch.strip():
        raise GitHubPullRequestError("Branch name is empty")
    encoded = _quote_path_segment(branch.strip())
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/branches/{encoded}",
            headers=_headers(access_token),
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (resp.text or "").strip()
            raise GitHubPullRequestError(
                f"GitHub branch fetch failed ({resp.status_code}): {detail or e}"
            ) from e
        payload = resp.json()

    if not isinstance(payload, dict):
        raise GitHubPullRequestError("GitHub branch fetch response is not an object")
    commit = payload.get("commit")
    if not isinstance(commit, dict):
        raise GitHubPullRequestError("GitHub branch response missing commit")
    sha = commit.get("sha")
    if not isinstance(sha, str) or not sha:
        raise GitHubPullRequestError("GitHub branch response missing commit sha")
    return sha


async def update_pull_request_base(
    *,
    owner: str,
    repo: str,
    number: int,
    base_branch: str,
    access_token: str,
) -> GitHubPullRequestInfo:
    if number <= 0:
        raise GitHubPullRequestError("PR number must be > 0")
    base_value = base_branch.strip()
    if not base_value:
        raise GitHubPullRequestError("Base branch is empty")

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.patch(
            f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/pulls/{number}",
            headers=_headers(access_token),
            json={"base": base_value},
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (resp.text or "").strip()
            raise GitHubPullRequestError(
                f"GitHub PR base update failed ({resp.status_code}): {detail or e}"
            ) from e
        payload = resp.json()

    if not isinstance(payload, dict):
        raise GitHubPullRequestError("GitHub PR base update response is not an object")
    return _parse_pr_payload(owner, repo, payload)
