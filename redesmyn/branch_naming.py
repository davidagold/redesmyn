from __future__ import annotations

import re

from redesmyn.strings import slugify

_TASK_TITLE_ID_RE = re.compile(r"^(T-\d+)\b")
_TASK_LOCAL_PATH_ID_RE = re.compile(r"(?:^|/)tasks/(T-\d+)/README\.md$")


def _identifier_from_local_path(local_path: str | None) -> str | None:
    if not local_path:
        return None
    match = _TASK_LOCAL_PATH_ID_RE.search(local_path.strip())
    return match.group(1) if match else None


def task_identifier(
    *,
    task_id: int,
    title: str,
    linear_identifier: str | None,
    local_path: str | None,
) -> str:
    if linear_identifier:
        return linear_identifier

    from_path = _identifier_from_local_path(local_path)
    if from_path:
        return from_path

    match = _TASK_TITLE_ID_RE.match(title.strip())
    if match:
        return match.group(1)

    return f"task-{task_id}"


def default_task_branch_name(
    *,
    epic_slug: str,
    task_id: int,
    title: str,
    linear_identifier: str | None,
    local_path: str | None,
) -> str:
    identifier = task_identifier(
        task_id=task_id,
        title=title,
        linear_identifier=linear_identifier,
        local_path=local_path,
    )

    full_title = (title or "").strip()
    title_remainder = full_title
    if identifier and full_title.startswith(identifier):
        title_remainder = full_title[len(identifier) :].strip()
    if title_remainder.startswith("-"):
        title_remainder = title_remainder[1:].strip()

    # Prefer a concise "task abbreviation" (similar to the UI branch label):
    # - drop parenthetical detail
    # - take the first '+'-separated segment
    # - drop common boilerplate suffixes
    title_remainder = re.sub(r"\([^)]*\)", " ", title_remainder).strip()
    title_remainder = title_remainder.split("+", 1)[0].strip()
    title_remainder = re.sub(r"\bimplementation\b", "", title_remainder, flags=re.I)
    title_remainder = re.sub(r"\s+", " ", title_remainder).strip()

    short = (
        slugify(title_remainder or full_title, fallback="task")[:60].strip("-")
        or "task"
    )
    return f"rn/{epic_slug}/{identifier}-{short}"
