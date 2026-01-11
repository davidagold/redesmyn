from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.context import RepoContext
from redesmyn.db import (
    Epic,
    Event,
    GitMergeBaseByInstance,
    GitRefStateByInstance,
    GitTrunkTimelineByInstance,
    Repository,
    Task,
)
from redesmyn.repo import (
    CommitInfo,
    git_commit_info,
    git_for_each_ref,
    git_is_ancestor,
    git_merge_base,
    git_rev_list,
    git_rev_list_count,
    git_rev_list_range,
    git_rev_parse,
)

TRUNK_TIMELINE_LIMIT = 4


def _choose_newest_sha(repo_root, shas: list[str]) -> str | None:
    if not shas:
        return None

    best = shas[0]
    for sha in shas[1:]:
        if git_is_ancestor(repo_root, best, sha):
            best = sha
    return best


def _build_commit_payload(
    sha: str, commit_info: dict[str, CommitInfo]
) -> dict[str, Any]:
    info = commit_info.get(sha)
    authored_at_raw = info.get("authored_at") if info is not None else None
    authored_at = authored_at_raw.isoformat() if authored_at_raw is not None else None
    committed_at_raw = info.get("committed_at") if info is not None else None
    committed_at = (
        committed_at_raw.isoformat() if committed_at_raw is not None else None
    )
    return {
        "sha": sha,
        "author_name": info.get("author_name") if info is not None else None,
        "author_email": info.get("author_email") if info is not None else None,
        "authored_at": authored_at,
        "committer_name": info.get("committer_name") if info is not None else None,
        "committer_email": info.get("committer_email") if info is not None else None,
        "committed_at": committed_at,
        "title": info.get("title") if info is not None else None,
        "message": info.get("message") if info is not None else None,
    }


def compute_trunk_timeline_snapshot(
    *,
    repo_root,
    root_branch: str,
    root_task_branches: list[str],
    limit: int = TRUNK_TIMELINE_LIMIT,
) -> dict[str, Any] | None:
    tip_sha = git_rev_parse(repo_root, root_branch)
    if tip_sha is None:
        return None

    merge_bases: list[str] = []
    for branch in root_task_branches:
        if not branch:
            continue
        mb = git_merge_base(repo_root, root_branch, branch)
        if mb:
            merge_bases.append(mb)

    base_sha = _choose_newest_sha(repo_root, merge_bases) or tip_sha

    commits_before = git_rev_list(
        repo_root, f"{base_sha}^", first_parent=True, max_count=limit
    )
    commits_after = git_rev_list_range(
        repo_root,
        f"{base_sha}..{root_branch}",
        first_parent=True,
        max_count=limit,
        reverse=True,
    )

    has_more_before = (
        git_rev_list_count(repo_root, f"{base_sha}^", first_parent=True) > limit
    )
    has_more_after = (
        git_rev_list_count(repo_root, f"{base_sha}..{root_branch}", first_parent=True)
        > limit
    )

    commit_info = git_commit_info(
        repo_root, [base_sha, *commits_before, *commits_after]
    )

    return {
        "base_sha": base_sha,
        "base_commit": _build_commit_payload(base_sha, commit_info),
        "commits_before": [
            _build_commit_payload(sha, commit_info) for sha in commits_before
        ],
        "commits_after": [
            _build_commit_payload(sha, commit_info) for sha in commits_after
        ],
        "has_more_before": has_more_before,
        "has_more_after": has_more_after,
    }


def _diff_refs(old: dict[str, str], new: dict[str, str]) -> list[dict[str, str | None]]:
    changes: list[dict[str, str | None]] = []
    all_refs = set(old) | set(new)
    for ref in sorted(all_refs):
        old_sha = old.get(ref)
        new_sha = new.get(ref)
        if old_sha == new_sha:
            continue
        changes.append({"ref": ref, "old_sha": old_sha, "new_sha": new_sha})
    return changes


async def update_git_projections_in_session(
    *,
    ctx: RepoContext,
    session: AsyncSession,
    repo: Repository,
    host_key: str,
    now: datetime | None = None,
    epics: Sequence[Epic] | None = None,
    tasks: Sequence[Task] | None = None,
) -> int:
    """
    Best-effort git-derived snapshots/events produced locally (repo-executor/daemon side).

    The API/control-plane reads these from the DB and must not execute git.

    Returns the number of git-ref-moved events emitted.
    """

    created_at = now or datetime.now(UTC)

    refs = git_for_each_ref(ctx.repo_root, prefix="refs/heads")
    head_sha = git_rev_parse(ctx.repo_root, "HEAD")
    if head_sha:
        refs["HEAD"] = head_sha

    ref_moved_events = 0

    # Avoid implicit flushes (and therefore long-lived SQLite write transactions)
    # while we execute git commands. Autoflush-triggered writes can hold the
    # single-writer lock long enough to trip the busy timeout in unrelated API
    # requests (e.g. agent restart emits).
    with session.no_autoflush:
        existing_state = await session.get(GitRefStateByInstance, (repo.id, host_key))
        if existing_state is None:
            session.add(
                GitRefStateByInstance(
                    repository_id=repo.id,
                    host_key=host_key,
                    refs=refs,
                    observed_at=created_at,
                    updated_at=created_at,
                )
            )
        else:
            changes = _diff_refs(existing_state.refs, refs)
            for change in changes:
                session.add(
                    Event(
                        event_type="git.ref_moved",
                        data={
                            **change,
                            "workspace_id": repo.workspace_id,
                            "repo_id": repo.repo_id,
                            "host_key": host_key,
                        },
                        created_at=created_at,
                    )
                )
                ref_moved_events += 1
            existing_state.refs = refs
            existing_state.observed_at = created_at
            existing_state.updated_at = created_at

        epic_rows: Sequence[Epic]
        if epics is None:
            epic_rows = list(
                await session.scalars(
                    select(Epic).where(Epic.repository_id == repo.id).order_by(Epic.id)
                )
            )
        else:
            epic_rows = epics

        if not epic_rows:
            return ref_moved_events

        all_tasks: Sequence[Task]
        if tasks is None:
            all_tasks = list(
                await session.scalars(
                    select(Task).where(Task.epic_id.in_([e.id for e in epic_rows]))
                )
            )
        else:
            all_tasks = tasks

        tasks_by_epic: dict[int, list[Task]] = {}
        for task in all_tasks:
            tasks_by_epic.setdefault(task.epic_id, []).append(task)

        for epic in epic_rows:
            epic_tasks = tasks_by_epic.get(epic.id, [])
            root_tasks = [t for t in epic_tasks if t.parent_task_id is None]
            root_task_branches = [t.branch_name for t in root_tasks if t.branch_name]

            trunk = compute_trunk_timeline_snapshot(
                repo_root=ctx.repo_root,
                root_branch=epic.root_branch,
                root_task_branches=root_task_branches,
            )
            if trunk is not None:
                existing_trunk = await session.get(
                    GitTrunkTimelineByInstance, (epic.id, host_key)
                )
                if existing_trunk is None:
                    session.add(
                        GitTrunkTimelineByInstance(
                            epic_id=epic.id,
                            host_key=host_key,
                            data=trunk,
                            observed_at=created_at,
                            updated_at=created_at,
                        )
                    )
                else:
                    existing_trunk.data = trunk
                    existing_trunk.observed_at = created_at
                    existing_trunk.updated_at = created_at

            for task in epic_tasks:
                mb = (
                    git_merge_base(ctx.repo_root, epic.root_branch, task.branch_name)
                    if task.branch_name
                    else None
                )
                existing_mb = await session.get(
                    GitMergeBaseByInstance, (task.id, host_key)
                )
                if existing_mb is None:
                    session.add(
                        GitMergeBaseByInstance(
                            task_id=task.id,
                            host_key=host_key,
                            merge_base_sha=mb,
                            observed_at=created_at,
                            updated_at=created_at,
                        )
                    )
                else:
                    existing_mb.merge_base_sha = mb
                    existing_mb.observed_at = created_at
                    existing_mb.updated_at = created_at

    return ref_moved_events
