from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from redesmyn.git_subprocess import run_git

DEFAULT_WORKSPACE_ID = "default"


@dataclass(frozen=True, slots=True)
class RepoKey:
    workspace_id: str
    repo_id: str

    def as_dict(self) -> dict[str, str]:
        return {"workspace_id": self.workspace_id, "repo_id": self.repo_id}


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    proc = run_git(args, cwd=repo_root, timeout_s=2)
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    return value or None


_SSH_GIT_REMOTE_RE = re.compile(r"^(?P<user>[^@]+)@(?P<host>[^:]+):(?P<path>.+)$")


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


def compute_repo_id(repo_root: Path) -> str:
    """
    Compute a stable-ish repo_id for v1.

    Preference order:
    - normalized `remote.origin.url` (stable across checkouts),
    - normalized first remote URL,
    - fallback to the repo_root path string.
    """
    remote = _run_git(repo_root, ["config", "--get", "remote.origin.url"])
    if remote is None:
        remote = _run_git(repo_root, ["config", "--get", "remote.upstream.url"])
    if remote is None:
        remote = _run_git(repo_root, ["remote", "-v"])
        if remote is not None:
            # Parse `origin  <url> (fetch)` lines.
            for line in remote.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    remote = parts[1]
                    break

    source = _normalize_remote_url(remote) if remote else str(repo_root)
    return sha256(source.encode("utf-8")).hexdigest()[:16]
