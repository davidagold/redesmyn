from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.context import RepoContext
from redesmyn.db import Epic, LinearAuth, Task
from redesmyn.domain.enums import TaskState
from redesmyn.integrations.linear import (
    LinearApiError,
    LinearClient,
    fetch_issue,
    fetch_issue_team_id,
    fetch_label_by_name,
    refresh_access_token,
    resolve_team_state_id,
    update_issue_state,
)
from redesmyn.integrations.linear_credentials import (
    LinearCredentials,
    default_linear_credential_store,
    is_expiring_soon,
)
from redesmyn.integrations.linear_state import linear_state_type_from_task_state
from redesmyn.settings import load_settings

log = structlog.get_logger("redesmyn.integrations.linear_automation")


async def maybe_push_task_state_to_linear(
    ctx: RepoContext,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    task_id: int,
    desired_task_state: TaskState,
    timeout_s: float = 3.0,
) -> None:
    if desired_task_state not in {TaskState.InProgress, TaskState.Done}:
        return

    async def _run() -> None:
        async with sessionmaker() as session:
            task = await session.get(Task, task_id)
            if task is None:
                return
            epic = await session.get(Epic, task.epic_id)
            if epic is None:
                return

            issue_id = task.linear_issue_id
            if issue_id is None:
                return

            creds = await _maybe_require_fresh_linear_credentials(ctx, session=session)
            if creds is None:
                return

            client = LinearClient(access_token=creds.access_token)
            issue = await fetch_issue(client, issue_id=issue_id)

            observed_at = datetime.now(UTC)
            task.linear_state_type = issue.state_type
            task.linear_state_observed_at = observed_at
            await _safe_commit(session)

            label = await fetch_label_by_name(client, label_name=epic.slug)
            if label is None or label.id not in issue.label_ids:
                return

            desired_state_type = linear_state_type_from_task_state(desired_task_state)
            current_state = (issue.state_type or "").strip().lower()
            if current_state == desired_state_type.strip().lower():
                return

            team_id = issue.team_id or await fetch_issue_team_id(
                client, issue_id=issue.id
            )
            if not team_id:
                return

            state_id = await resolve_team_state_id(
                client, team_id=team_id, state_type=desired_state_type
            )
            updated = await update_issue_state(
                client,
                issue_id=issue.id,
                state_id=state_id,
            )

            task.linear_state_type = updated.state_type
            task.linear_state_observed_at = observed_at
            await _safe_commit(session)

    try:
        await asyncio.wait_for(_run(), timeout=timeout_s)
    except asyncio.TimeoutError:
        log.info(
            "linear.automation.push_state.timeout",
            task_id=task_id,
            desired_state=str(desired_task_state),
            timeout_s=timeout_s,
        )
    except LinearApiError as exc:
        log.info(
            "linear.automation.push_state.failed",
            task_id=task_id,
            desired_state=str(desired_task_state),
            code=exc.code,
            status_code=exc.status_code,
            operation=exc.operation,
        )
    except Exception as exc:
        log.info(
            "linear.automation.push_state.failed",
            task_id=task_id,
            desired_state=str(desired_task_state),
            error=str(exc),
        )


async def maybe_push_task_merge_ready_to_linear(
    ctx: RepoContext,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    task_id: int,
    timeout_s: float = 15.0,
) -> None:
    step = "init"

    async def _run() -> None:
        nonlocal step

        async with sessionmaker() as session:
            step = "load_task"
            task = await session.get(Task, task_id)
            if (
                task is None
                or task.merge_ready_at is None
                or task.linear_issue_id is None
            ):
                log.info(
                    "linear.automation.push_merge_ready.skipped",
                    task_id=task_id,
                    reason="missing_task_merge_ready_or_linear_issue",
                )
                return
            step = "load_epic"
            epic = await session.get(Epic, task.epic_id)
            if epic is None:
                log.info(
                    "linear.automation.push_merge_ready.skipped",
                    task_id=task_id,
                    reason="missing_epic",
                )
                return

            step = "require_credentials"
            creds = await _maybe_require_fresh_linear_credentials(ctx, session=session)
            if creds is None:
                log.info(
                    "linear.automation.push_merge_ready.skipped",
                    task_id=task_id,
                    reason="missing_credentials",
                )
                return

            client = LinearClient(access_token=creds.access_token)
            step = "fetch_issue"
            issue = await fetch_issue(client, issue_id=task.linear_issue_id)
            observed_at = datetime.now(UTC)

            step = "fetch_label"
            label = await fetch_label_by_name(client, label_name=epic.slug)
            if label is None or label.id not in issue.label_ids:
                task.linear_state_type = issue.state_type
                task.linear_state_observed_at = observed_at
                await _safe_commit(session)
                log.info(
                    "linear.automation.push_merge_ready.skipped",
                    task_id=task_id,
                    reason="epic_label_missing_from_issue",
                    epic_slug=epic.slug,
                )
                return

            step = "resolve_team_id"
            team_id = issue.team_id or await fetch_issue_team_id(
                client, issue_id=issue.id
            )
            if not team_id:
                task.linear_state_type = issue.state_type
                task.linear_state_observed_at = observed_at
                await _safe_commit(session)
                log.info(
                    "linear.automation.push_merge_ready.skipped",
                    task_id=task_id,
                    reason="missing_team_id",
                )
                return

            # Best-effort mapping: treat merge-ready as "started" and pick the
            # last started-state in the workflow as a reasonable "ready/review"
            # approximation when teams have multiple started states.
            step = "resolve_state_id"
            state_id = await resolve_team_state_id(
                client, team_id=team_id, state_type="started", pick="last"
            )
            step = "update_issue_state"
            updated = await update_issue_state(
                client,
                issue_id=issue.id,
                state_id=state_id,
            )
            step = "commit"
            task.linear_state_type = updated.state_type
            task.linear_state_observed_at = datetime.now(UTC)
            await _safe_commit(session)
            log.info(
                "linear.automation.push_merge_ready.updated",
                task_id=task_id,
                from_state_type=issue.state_type,
                to_state_type=updated.state_type,
            )

    start = time.monotonic()
    log.info(
        "linear.automation.push_merge_ready.started",
        task_id=task_id,
        timeout_s=timeout_s,
    )
    try:
        await _run()
    except LinearApiError as exc:
        log.info(
            "linear.automation.push_merge_ready.failed",
            task_id=task_id,
            code=exc.code,
            status_code=exc.status_code,
            operation=exc.operation,
            step=step,
            duration_s=round(time.monotonic() - start, 3),
        )
    except Exception as exc:
        log.info(
            "linear.automation.push_merge_ready.failed",
            task_id=task_id,
            error=str(exc),
            step=step,
            duration_s=round(time.monotonic() - start, 3),
        )
    else:
        log.info(
            "linear.automation.push_merge_ready.finished",
            task_id=task_id,
            step=step,
            duration_s=round(time.monotonic() - start, 3),
        )


async def _maybe_require_fresh_linear_credentials(
    ctx: RepoContext,
    *,
    session: AsyncSession,
    skew: timedelta = timedelta(minutes=5),
) -> LinearCredentials | None:
    store = default_linear_credential_store()

    creds = store.get()
    if creds is None:
        auth = await session.scalar(
            select(LinearAuth).order_by(desc(LinearAuth.id)).limit(1)
        )
        if auth is not None:
            creds = LinearCredentials(
                access_token=auth.access_token,
                refresh_token=auth.refresh_token,
                token_type=auth.token_type,
                scope=auth.scope,
                expires_at=auth.expires_at,
                connected_at=auth.created_at,
            )
            try:
                store.set(creds)
                await session.execute(delete(LinearAuth))
                await session.commit()
            except Exception:
                try:
                    await session.rollback()
                except Exception:
                    pass
                return None

    if creds is None:
        return None

    if not is_expiring_soon(creds, skew=skew):
        return creds

    if not creds.refresh_token:
        return None

    try:
        settings = load_settings(repo_root=ctx.repo_root)
        token = await refresh_access_token(settings, refresh_token=creds.refresh_token)
    except Exception:
        return None

    next_creds = LinearCredentials(
        access_token=token.access_token,
        refresh_token=token.refresh_token or creds.refresh_token,
        token_type=token.token_type,
        scope=token.scope or creds.scope,
        expires_at=token.expires_at,
        connected_at=creds.connected_at,
    )
    store.set(next_creds)
    return next_creds


async def _safe_commit(session: AsyncSession) -> None:
    try:
        await session.commit()
    except Exception:
        try:
            await session.rollback()
        except Exception:
            pass
        raise
