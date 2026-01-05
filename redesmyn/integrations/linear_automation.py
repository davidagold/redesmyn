from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

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

logger = logging.getLogger("redesmyn.integrations.linear_automation")


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
        logger.info(
            "linear.automation.push_state.timeout",
            extra={
                "task_id": task_id,
                "desired_state": desired_task_state,
            },
        )
    except LinearApiError as exc:
        logger.info(
            "linear.automation.push_state.failed",
            extra={
                "task_id": task_id,
                "desired_state": desired_task_state,
                "code": exc.code,
                "status_code": exc.status_code,
                "operation": exc.operation,
            },
        )
    except Exception as exc:
        logger.info(
            "linear.automation.push_state.failed",
            extra={
                "task_id": task_id,
                "desired_state": desired_task_state,
                "error": str(exc),
            },
        )


async def maybe_push_task_merge_ready_to_linear(
    ctx: RepoContext,
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    task_id: int,
    timeout_s: float = 15.0,
) -> None:
    async def _run() -> None:
        async with sessionmaker() as session:
            task = await session.get(Task, task_id)
            if (
                task is None
                or task.merge_ready_at is None
                or task.linear_issue_id is None
            ):
                return
            epic = await session.get(Epic, task.epic_id)
            if epic is None:
                return

            creds = await _maybe_require_fresh_linear_credentials(ctx, session=session)
            if creds is None:
                return

            client = LinearClient(access_token=creds.access_token)
            issue = await fetch_issue(client, issue_id=task.linear_issue_id)

            observed_at = datetime.now(UTC)
            task.linear_state_type = issue.state_type
            task.linear_state_observed_at = observed_at
            await _safe_commit(session)

            label = await fetch_label_by_name(client, label_name=epic.slug)
            if label is None or label.id not in issue.label_ids:
                return

            team_id = issue.team_id or await fetch_issue_team_id(
                client, issue_id=issue.id
            )
            if not team_id:
                return

            # Best-effort mapping: treat merge-ready as "started" and pick the
            # last started-state in the workflow as a reasonable "ready/review"
            # approximation when teams have multiple started states.
            state_id = await resolve_team_state_id(
                client, team_id=team_id, state_type="started", pick="last"
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
        logger.info(
            "linear.automation.push_merge_ready.timeout",
            extra={"task_id": task_id, "timeout_s": timeout_s},
        )
    except LinearApiError as exc:
        logger.info(
            "linear.automation.push_merge_ready.failed",
            extra={
                "task_id": task_id,
                "code": exc.code,
                "status_code": exc.status_code,
                "operation": exc.operation,
            },
        )
    except Exception as exc:
        logger.info(
            "linear.automation.push_merge_ready.failed",
            extra={"task_id": task_id, "error": str(exc)},
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
