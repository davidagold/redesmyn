from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.db import Event, MergeRun
from redesmyn.db.models import MergeRunPlanData, MergeRunPlanStepData
from redesmyn.domain.enums import MergeRunStatus
from redesmyn.git_mechanics_v0 import MergeRunStepUpdate


async def apply_merge_run_event_update(
    session: AsyncSession,
    *,
    run_id: str,
    host_key: str | None,
    data: dict[str, object],
) -> None:
    row = await session.scalar(select(MergeRun).where(MergeRun.run_id == run_id))
    if row is None:
        return

    if row.status in {MergeRunStatus.Succeeded, MergeRunStatus.Canceled}:
        return

    if host_key is not None:
        row.host_key = host_key

    status_value = data.get("status")
    if isinstance(status_value, str):
        try:
            row.status = MergeRunStatus(status_value)
        except Exception:
            pass

    plan_value = data.get("plan") or data.get("plan_snapshot")
    if isinstance(plan_value, dict):
        try:
            plan = MergeRunPlanData.model_validate(plan_value)
        except Exception:
            plan = None
        if plan is not None:
            row.plan = plan.model_dump(mode="python")

    blocked_step_index = data.get("blocked_step_index")
    if isinstance(blocked_step_index, int):
        row.blocked_step_index = blocked_step_index
    blocked_step_kind = data.get("blocked_step_kind")
    if isinstance(blocked_step_kind, str):
        row.blocked_step_kind = blocked_step_kind
    blocked_task_id = data.get("blocked_task_id")
    if isinstance(blocked_task_id, int):
        row.blocked_task_id = blocked_task_id
    blocked_branch_name = data.get("blocked_branch_name")
    if isinstance(blocked_branch_name, str):
        row.blocked_branch_name = blocked_branch_name
    blocked_worktree_path = data.get("blocked_worktree_path")
    if isinstance(blocked_worktree_path, str):
        row.blocked_worktree_path = blocked_worktree_path

    error_value = data.get("error")
    if isinstance(error_value, str):
        row.blocked_error = error_value

    if row.status in {MergeRunStatus.Succeeded, MergeRunStatus.Canceled}:
        row.current_step_index = None
        row.blocked_step_index = None
        row.blocked_step_kind = None
        row.blocked_task_id = None
        row.blocked_branch_name = None
        row.blocked_worktree_path = None
        row.blocked_error = None


async def emit_task_merge_event(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    payload: dict[str, object],
    host_key: str | None = None,
) -> None:
    if host_key is not None:
        payload = {**payload, "host_key": host_key}
    async with sessionmaker() as session:
        session.add(
            Event(
                event_type="task.merge",
                data=payload,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def emit_merge_run_event(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    task_id: int,
    epic_id: int,
    requested_task_id: int,
    status: MergeRunStatus,
    host_key: str | None = None,
    operation: Literal["merge", "restack"] = "merge",
    blocked_step_index: int | None = None,
    blocked_step_kind: str | None = None,
    blocked_branch_name: str | None = None,
) -> None:
    data: dict[str, object] = {
        "run_id": run_id,
        "task_id": task_id,
        "epic_id": epic_id,
        "requested_task_id": requested_task_id,
        "status": status,
        "operation": operation,
    }
    if host_key is not None:
        data["host_key"] = host_key
    if blocked_step_index is not None:
        data["blocked_step_index"] = blocked_step_index
    if blocked_step_kind is not None:
        data["blocked_step_kind"] = blocked_step_kind
    if blocked_branch_name is not None:
        data["blocked_branch_name"] = blocked_branch_name

    async with sessionmaker() as session:
        session.add(
            Event(
                event_type="merge.run",
                data=data,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


def merge_run_plan_snapshot(
    *,
    plan,
    operation: Literal["merge", "restack"],
) -> dict[str, object]:
    steps = [
        MergeRunPlanStepData(
            index=index,
            kind=step.kind,
            task_id=step.task_id,
            branch_name=step.branch_name,
            worktree_path=str(step.worktree_path),
            upstream_ref=step.upstream_ref,
            base_branch=step.base_branch,
        )
        for index, step in enumerate(plan.steps)
    ]

    base_worktree_value = ""
    try:
        base_worktree = getattr(plan, "base_worktree", None)
        if base_worktree is not None:
            base_worktree_value = str(base_worktree)
    except Exception:
        base_worktree_value = ""

    payload = MergeRunPlanData(
        operation=operation,
        base_branch=plan.base_branch,
        base_worktree=base_worktree_value,
        scope=plan.scope,
        restack_mode=getattr(plan, "restack_mode", "strict"),
        spine_task_ids=list(getattr(plan, "spine_task_ids", [])),
        affected_task_ids=list(plan.affected_task_ids),
        steps=steps,
    )
    return payload.model_dump(mode="python")


async def upsert_merge_run(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    epic_id: int,
    task_id: int,
    host_key: str | None,
    canonical: bool,
    scope: str,
    allow_running: bool,
    force: bool,
    plan_snapshot: dict[str, object],
) -> None:
    async with sessionmaker() as session:
        row = await session.scalar(select(MergeRun).where(MergeRun.run_id == run_id))
        if row is None:
            session.add(
                MergeRun(
                    run_id=run_id,
                    epic_id=epic_id,
                    requested_task_id=task_id,
                    host_key=host_key,
                    canonical=canonical,
                    status=MergeRunStatus.Running,
                    scope=scope,
                    allow_running=allow_running,
                    force=force,
                    plan=plan_snapshot,
                )
            )
        else:
            row.epic_id = epic_id
            row.requested_task_id = task_id
            row.host_key = host_key
            row.canonical = canonical
            row.status = MergeRunStatus.Running
            row.scope = scope
            row.allow_running = allow_running
            row.force = force
            row.plan = plan_snapshot
            row.current_step_index = None
            row.blocked_step_index = None
            row.blocked_step_kind = None
            row.blocked_task_id = None
            row.blocked_branch_name = None
            row.blocked_worktree_path = None
            row.blocked_error = None
        await session.commit()


async def record_merge_run_step_update(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    update: MergeRunStepUpdate,
) -> None:
    now = datetime.now(UTC)
    async with sessionmaker() as session:
        row = await session.scalar(select(MergeRun).where(MergeRun.run_id == run_id))
        if row is None:
            return
        row.current_step_index = update.step_index
        if update.phase == "failed":
            row.status = (
                MergeRunStatus.Blocked if update.blocked else MergeRunStatus.Failed
            )
            row.blocked_step_index = update.step_index
            row.blocked_step_kind = update.step.kind
            row.blocked_task_id = update.step.task_id
            row.blocked_branch_name = update.step.branch_name
            row.blocked_worktree_path = str(update.step.worktree_path)
            row.blocked_error = update.error
            session.add(
                Event(
                    event_type="merge.run",
                    data={
                        "run_id": run_id,
                        "task_id": update.step.task_id,
                        "epic_id": row.epic_id,
                        "requested_task_id": row.requested_task_id,
                        "status": row.status,
                        "operation": row.operation,
                        "host_key": row.host_key,
                        "blocked_step_index": row.blocked_step_index,
                        "blocked_step_kind": row.blocked_step_kind,
                        "blocked_branch_name": row.blocked_branch_name,
                    },
                    created_at=now,
                )
            )
        await session.commit()


async def set_merge_run_status(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    status: MergeRunStatus,
    error: str | None = None,
) -> None:
    now = datetime.now(UTC)
    async with sessionmaker() as session:
        row = await session.scalar(select(MergeRun).where(MergeRun.run_id == run_id))
        if row is None:
            return

        task_id: int | None = row.blocked_task_id
        blocked_step_index = row.blocked_step_index
        blocked_step_kind = row.blocked_step_kind
        blocked_branch_name = row.blocked_branch_name
        if task_id is None:
            try:
                plan = MergeRunPlanData.model_validate(row.plan)
                if plan.spine_task_ids:
                    task_id = plan.spine_task_ids[-1]
                elif plan.steps and plan.steps[0].task_id is not None:
                    task_id = plan.steps[0].task_id
            except Exception:
                task_id = None

        row.status = status
        if status in {MergeRunStatus.Succeeded, MergeRunStatus.Canceled}:
            row.current_step_index = None
            row.blocked_step_index = None
            row.blocked_step_kind = None
            row.blocked_task_id = None
            row.blocked_branch_name = None
            row.blocked_worktree_path = None
            row.blocked_error = None
        if error is not None:
            row.blocked_error = error

        if task_id is not None:
            data: dict[str, object] = {
                "run_id": run_id,
                "task_id": task_id,
                "epic_id": row.epic_id,
                "requested_task_id": row.requested_task_id,
                "status": status,
                "operation": row.operation,
            }
            if row.host_key is not None:
                data["host_key"] = row.host_key
            if blocked_step_index is not None:
                data["blocked_step_index"] = blocked_step_index
            if blocked_step_kind is not None:
                data["blocked_step_kind"] = blocked_step_kind
            if blocked_branch_name is not None:
                data["blocked_branch_name"] = blocked_branch_name
            session.add(
                Event(
                    event_type="merge.run",
                    data=data,
                    created_at=now,
                )
            )
        await session.commit()
