from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from redesmyn.git_subprocess import run_git

_SSH_GIT_REMOTE_RE = re.compile(r"^(?P<user>[^@]+)@(?P<host>[^:]+):(?P<path>.+)$")
_GITHUB_OWNER_REPO_RE = re.compile(
    r"^(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}[A-Za-z0-9])?)/(?P<repo>[A-Za-z0-9_.-]+)$"
)


@dataclass(frozen=True, slots=True)
class GithubRepoRef:
    host: str
    owner: str
    repo: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


def _normalize_remote_url(remote: str) -> str:
    value = remote.strip()
    match = _SSH_GIT_REMOTE_RE.match(value)
    if match:
        host = match.group("host").strip().lower()
        path = match.group("path").strip().lstrip("/")
        if path.endswith(".git"):
            path = path[: -len(".git")]
        return f"{host}/{path}"

    for prefix in ("https://", "http://", "ssh://", "git://"):
        if value.startswith(prefix):
            remainder = value[len(prefix) :]
            remainder = remainder.split("#", 1)[0].split("?", 1)[0]
            remainder = remainder.rstrip("/")
            if remainder.endswith(".git"):
                remainder = remainder[: -len(".git")]
            if "/" in remainder:
                host, rest = remainder.split("/", 1)
                return f"{host.strip().lower()}/{rest.strip().lstrip('/')}"
            return remainder.strip().lower()

    value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[: -len(".git")]
    return value


def _normalize_github_host(host: str) -> str:
    normalized = host.strip().lower()
    if normalized == "www.github.com":
        return "github.com"
    if normalized == "ssh.github.com":
        return "github.com"
    return normalized


def parse_github_repo_ref(value: str) -> GithubRepoRef | None:
    """
    Parse a GitHub repo identity from common forms:
    - owner/repo
    - git@github.com:owner/repo.git
    - https://github.com/owner/repo(.git)
    """
    raw = value.strip()
    if not raw:
        return None

    match = _GITHUB_OWNER_REPO_RE.match(raw)
    if match:
        return GithubRepoRef(
            host="github.com",
            owner=match.group("owner"),
            repo=match.group("repo"),
        )

    normalized = _normalize_remote_url(raw)
    if "/" not in normalized:
        return None
    host, path = normalized.split("/", 1)
    host = _normalize_github_host(host)
    if host != "github.com":
        return None

    parts = [p for p in path.split("/") if p]
    if len(parts) != 2:
        return None

    owner, repo = parts
    match = _GITHUB_OWNER_REPO_RE.match(f"{owner}/{repo}")
    if not match:
        return None
    return GithubRepoRef(host=host, owner=owner, repo=repo)


def detect_github_repo_ref(repo_root: Path) -> GithubRepoRef | None:
    proc = run_git(["config", "--get", "remote.origin.url"], cwd=repo_root, timeout_s=2)
    remote = (proc.stdout or "").strip() if proc.returncode == 0 else ""
    if not remote:
        proc = run_git(
            ["config", "--get", "remote.upstream.url"], cwd=repo_root, timeout_s=2
        )
        remote = (proc.stdout or "").strip() if proc.returncode == 0 else ""

    if not remote:
        proc = run_git(["remote", "-v"], cwd=repo_root, timeout_s=2)
        if proc.returncode == 0:
            for line in (proc.stdout or "").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    remote = parts[1]
                    break

    return parse_github_repo_ref(remote) if remote else None
