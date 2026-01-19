from __future__ import annotations

import structlog

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.db import Epic, Task
from redesmyn.integrations.github_credentials import default_github_credential_store
from redesmyn.integrations.github_pr import (
    GitHubPullRequestError,
    GitHubPullRequestRef,
    fetch_branch_head_sha,
    fetch_pull_request,
    fetch_repo_default_branch,
    list_repo_branches,
    update_pull_request_base,
)

log = structlog.get_logger("redesmyn.integrations.github_pr_refresh")


async def _choose_temporary_base_branch(
    *,
    owner: str,
    repo: str,
    current_base: str,
    head_branch: str | None,
    access_token: str,
) -> str | None:
    excluded = {current_base}
    if head_branch:
        excluded.add(head_branch)

    candidates: list[str] = []
    try:
        default_branch = await fetch_repo_default_branch(
            owner=owner,
            repo=repo,
            access_token=access_token,
        )
    except GitHubPullRequestError:
        default_branch = None

    if default_branch:
        candidates.append(default_branch)

    candidates.extend(["main", "master"])

    seen: set[str] = set()
    for candidate in candidates:
        name = candidate.strip()
        if not name or name in seen or name in excluded:
            continue
        seen.add(name)
        try:
            await fetch_branch_head_sha(
                owner=owner,
                repo=repo,
                branch=name,
                access_token=access_token,
            )
        except GitHubPullRequestError:
            continue
        return name

    try:
        branches = await list_repo_branches(
            owner=owner,
            repo=repo,
            access_token=access_token,
        )
    except GitHubPullRequestError:
        return None

    for name in branches:
        if name in excluded:
            continue
        return name
    return None


async def refresh_epic_pull_request_bases(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    epic_id: int,
) -> None:
    """
    Best-effort: ensure GitHub's PR comparison base keeps up with branch updates.

    Occasionally GitHub PRs appear to keep using an older base branch OID even
    after the base branch advances (e.g. after stack merges/restacks). We work
    around this by first attempting a "no-op" base update (set the base to the
    same branch again), and falling back to temporarily retargeting the PR base
    branch and restoring it, which forces GitHub to recompute the base.
    """
    store = default_github_credential_store()
    creds = store.get()
    if creds is None:
        return

    access_token = creds.access_token

    async with sessionmaker() as session:
        epic = await session.get(Epic, epic_id)
        if epic is None:
            return
        tasks = list(
            await session.scalars(
                select(Task).where(
                    Task.epic_id == epic_id, Task.github_pr_id.isnot(None)
                )
            )
        )

    if not tasks:
        return

    refreshed = 0
    for task in tasks:
        pr_id = (task.github_pr_id or "").strip()
        if not pr_id:
            continue
        try:
            ref = GitHubPullRequestRef.parse(pr_id)
        except ValueError:
            log.warning("github.pr_id.invalid", task_id=task.id, pr_id=pr_id)
            continue

        try:
            pr = await fetch_pull_request(
                owner=ref.owner,
                repo=ref.repo,
                number=ref.number,
                access_token=access_token,
            )
        except GitHubPullRequestError as e:
            log.warning(
                "github.pr.fetch_failed",
                task_id=task.id,
                pr_id=pr_id,
                error=str(e),
            )
            continue

        if pr.state != "open":
            continue
        base_branch = (pr.base_branch or "").strip()
        base_sha = (pr.base_sha or "").strip()
        if not base_branch or not base_sha:
            continue

        try:
            head_sha = await fetch_branch_head_sha(
                owner=ref.owner,
                repo=ref.repo,
                branch=base_branch,
                access_token=access_token,
            )
        except GitHubPullRequestError as e:
            log.warning(
                "github.branch.fetch_failed",
                task_id=task.id,
                pr_id=pr_id,
                branch=base_branch,
                error=str(e),
            )
            continue

        if head_sha == base_sha:
            continue

        # Prefer a no-op base update first. In practice, GitHub often refreshes
        # its comparison base even when you "change" the base branch to the
        # same value (e.g. `gh pr edit --base gpui` when base is already gpui),
        # and it avoids noisy timeline entries.
        try:
            updated = await update_pull_request_base(
                owner=ref.owner,
                repo=ref.repo,
                number=ref.number,
                base_branch=base_branch,
                access_token=access_token,
            )
        except GitHubPullRequestError as e:
            log.warning(
                "github.pr_base_refresh.noop_failed",
                task_id=task.id,
                pr_id=pr_id,
                base_branch=base_branch,
                error=str(e),
            )
        else:
            updated_sha = (updated.base_sha or "").strip()
            if not updated_sha:
                try:
                    updated = await fetch_pull_request(
                        owner=ref.owner,
                        repo=ref.repo,
                        number=ref.number,
                        access_token=access_token,
                    )
                except GitHubPullRequestError:
                    updated = updated
                updated_sha = (updated.base_sha or "").strip()

            if updated_sha == head_sha:
                refreshed += 1
                log.info(
                    "github.pr_base_refreshed",
                    task_id=task.id,
                    pr_id=pr_id,
                    method="noop",
                    base_branch=base_branch,
                    old_base_sha=base_sha,
                    base_head_sha=head_sha,
                    new_base_sha=updated.base_sha,
                )
                continue

        temp_base = await _choose_temporary_base_branch(
            owner=ref.owner,
            repo=ref.repo,
            current_base=base_branch,
            head_branch=pr.head_branch,
            access_token=access_token,
        )
        if temp_base is None:
            log.warning(
                "github.pr_base_refresh.no_temp_base",
                task_id=task.id,
                pr_id=pr_id,
                base_branch=base_branch,
                base_sha=base_sha,
                base_head_sha=head_sha,
            )
            continue

        try:
            await update_pull_request_base(
                owner=ref.owner,
                repo=ref.repo,
                number=ref.number,
                base_branch=temp_base,
                access_token=access_token,
            )
            updated = await update_pull_request_base(
                owner=ref.owner,
                repo=ref.repo,
                number=ref.number,
                base_branch=base_branch,
                access_token=access_token,
            )
        except GitHubPullRequestError as e:
            log.warning(
                "github.pr_base_refresh.failed",
                task_id=task.id,
                pr_id=pr_id,
                base_branch=base_branch,
                temp_base=temp_base,
                error=str(e),
            )
            continue

        refreshed += 1
        log.info(
            "github.pr_base_refreshed",
            task_id=task.id,
            pr_id=pr_id,
            method="retarget",
            base_branch=base_branch,
            old_base_sha=base_sha,
            base_head_sha=head_sha,
            new_base_sha=updated.base_sha,
            temp_base=temp_base,
        )

    if refreshed:
        log.info(
            "github.pr_base_refresh.completed",
            epic_id=epic_id,
            refreshed=refreshed,
            task_count=len(tasks),
        )
