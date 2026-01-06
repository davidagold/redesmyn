from __future__ import annotations

import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence


class GitFailureKind(StrEnum):
    Timeout = "timeout"
    NotRepo = "not_repo"
    RefMissing = "ref_missing"
    Unknown = "unknown"


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return value.decode(errors="replace")
    return value


def run_git(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _to_text(exc.stdout)
        stderr = _to_text(exc.stderr)
        if stderr:
            stderr = stderr.rstrip() + "\n"
        stderr += f"git timed out after {timeout_s}s"
        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=124,
            stdout=stdout,
            stderr=stderr,
        )


def classify_git_failure(
    proc: subprocess.CompletedProcess[str],
) -> GitFailureKind | None:
    if proc.returncode == 0:
        return None
    if proc.returncode == 124:
        return GitFailureKind.Timeout

    stderr = (proc.stderr or "").lower()
    stdout = (proc.stdout or "").lower()
    text = stderr + "\n" + stdout

    if "not a git repository" in text:
        return GitFailureKind.NotRepo

    if (
        "unknown revision or path not in the working tree" in text
        or "bad revision" in text
        or "ambiguous argument" in text
        or "needed a single revision" in text
    ):
        return GitFailureKind.RefMissing

    return GitFailureKind.Unknown
