from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from redesmyn.context import RepoContext
from redesmyn.db import (
    Epic,
    Event,
    GitMergeBase,
    GitRefState,
    GitTrunkTimeline,
    Node,
    Repository,
    create_engine,
    create_sessionmaker,
    init_db,
)
from redesmyn.repo import (
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
    sha: str, commit_info: dict[str, dict[str, object]]
) -> dict[str, Any]:
    info = commit_info.get(sha, {})
    authored_at_raw = info.get("authored_at")
    authored_at: str | None
    if isinstance(authored_at_raw, datetime):
        authored_at = authored_at_raw.isoformat()
    else:
        authored_at = None
    return {
        "sha": sha,
        "author_name": info.get("author_name"),
        "author_email": info.get("author_email"),
        "authored_at": authored_at,
    }


def compute_trunk_timeline_snapshot(
    *,
    repo_root,
    root_branch: str,
    root_node_branches: list[str],
    limit: int = TRUNK_TIMELINE_LIMIT,
) -> dict[str, Any] | None:
    tip_sha = git_rev_parse(repo_root, root_branch)
    if tip_sha is None:
        return None

    merge_bases: list[str] = []
    for branch in root_node_branches:
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


async def update_git_projections(ctx: RepoContext) -> None:
    """
    Best-effort git-derived snapshots/events produced locally (daemon side).

    The API/control-plane reads these from the DB and must not execute git.
    """
    ctx.state_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(ctx.db_path)
    try:
        await init_db(engine)
        sessionmaker = create_sessionmaker(engine)

        async with sessionmaker() as session:
            repo = await session.scalar(
                select(Repository).where(Repository.repo_root == str(ctx.repo_root))
            )
            if repo is None:
                return

            now = datetime.now(UTC)

            refs = git_for_each_ref(ctx.repo_root, prefix="refs/heads")
            head_sha = git_rev_parse(ctx.repo_root, "HEAD")
            if head_sha:
                refs["HEAD"] = head_sha

            existing_state = await session.get(GitRefState, repo.id)
            if existing_state is None:
                session.add(
                    GitRefState(
                        repository_id=repo.id,
                        refs=refs,
                        observed_at=now,
                        updated_at=now,
                    )
                )
            else:
                changes = _diff_refs(existing_state.refs, refs)
                for change in changes:
                    session.add(
                        Event(
                            event_type="git.ref_moved",
                            data=change,
                            created_at=now,
                        )
                    )
                existing_state.refs = refs
                existing_state.observed_at = now
                existing_state.updated_at = now

            epics = list(
                await session.scalars(
                    select(Epic).where(Epic.repository_id == repo.id).order_by(Epic.id)
                )
            )
            if not epics:
                await session.commit()
                return

            nodes = list(
                await session.scalars(
                    select(Node).where(Node.epic_id.in_([e.id for e in epics]))
                )
            )
            nodes_by_epic: dict[int, list[Node]] = {}
            for node in nodes:
                nodes_by_epic.setdefault(node.epic_id, []).append(node)

            for epic in epics:
                epic_nodes = nodes_by_epic.get(epic.id, [])
                root_nodes = [n for n in epic_nodes if n.parent_node_id is None]
                root_node_branches = [
                    n.branch_name for n in root_nodes if n.branch_name
                ]

                trunk = compute_trunk_timeline_snapshot(
                    repo_root=ctx.repo_root,
                    root_branch=epic.root_branch,
                    root_node_branches=root_node_branches,
                )
                if trunk is not None:
                    existing_trunk = await session.get(GitTrunkTimeline, epic.id)
                    if existing_trunk is None:
                        session.add(
                            GitTrunkTimeline(
                                epic_id=epic.id,
                                data=trunk,
                                observed_at=now,
                                updated_at=now,
                            )
                        )
                    else:
                        existing_trunk.data = trunk
                        existing_trunk.observed_at = now
                        existing_trunk.updated_at = now

                for node in epic_nodes:
                    mb = (
                        git_merge_base(
                            ctx.repo_root, epic.root_branch, node.branch_name
                        )
                        if node.branch_name
                        else None
                    )
                    existing_mb = await session.get(GitMergeBase, node.id)
                    if existing_mb is None:
                        session.add(
                            GitMergeBase(
                                node_id=node.id,
                                merge_base_sha=mb,
                                observed_at=now,
                                updated_at=now,
                            )
                        )
                    else:
                        existing_mb.merge_base_sha = mb
                        existing_mb.observed_at = now
                        existing_mb.updated_at = now

            await session.commit()
    finally:
        await engine.dispose()
