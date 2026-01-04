from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, cast

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.context import RepoContext
from redesmyn.db import DaemonCommand, MergeRun, Repository, Task
from redesmyn.db.models import MergeRunPlanData
from redesmyn.domain.enums import CommandState, MergeRunStatus
from redesmyn.git_mechanics_v0 import (
    MergeBlockedByRunningAgents,
    MergeCascadePlan,
    MergePlanError,
    RestackPlan,
    build_merge_cascade_plan,
    build_restack_plan,
    execute_merge_cascade_plan,
    execute_restack_plan,
)
from redesmyn.merge_runs import (
    emit_merge_run_event,
    emit_task_merge_event,
    merge_run_plan_snapshot,
    record_merge_run_step_update,
    set_merge_run_status,
    upsert_merge_run,
)
from redesmyn.repo import GitCommandError
from redesmyn.repo_identity import RepoKey
from redesmyn.schemas.core import (
    DaemonCommandResponse,
    MergeRunResumeRequest,
    MergeRunResumeResponse,
    TaskMergePlanStepResponse,
    TaskMergeRequest,
    TaskMergeResponse,
    TaskRestackRequest,
    TaskRestackResponse,
)
from redesmyn.ws_protocol import ServerCommand
from redesmyn.ws_runtime import DaemonConnectionRegistry


@dataclass(frozen=True, slots=True)
class RepoExecutorTarget:
    repo: Repository
    repo_key: RepoKey
    target_host_key: str
    primary_host_key: str | None
    canonical: bool
    is_local: bool


@dataclass(frozen=True, slots=True)
class RepoExecutorStatus:
    primary_host_key: str | None
    attached_host_keys: list[str]


class RepoExecutorError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class RepoExecutor(Protocol):
    async def get_status(
        self,
        *,
        repo: Repository,
        primary_host_key: str | None,
    ) -> RepoExecutorStatus: ...

    async def merge(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskMergeRequest,
    ) -> TaskMergeResponse: ...

    async def restack(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskRestackRequest,
    ) -> TaskRestackResponse: ...

    async def resume_merge_run(
        self,
        *,
        target: RepoExecutorTarget,
        run: MergeRun,
        request: MergeRunResumeRequest,
    ) -> MergeRunResumeResponse: ...


def _require_run_id(*, run_id: str | None) -> str:
    run_id_value = run_id or ""
    if not run_id_value:
        raise RepoExecutorError("Missing run id", status_code=400)
    return run_id_value


def _plan_steps_response(*, plan_steps) -> list[TaskMergePlanStepResponse]:
    return [
        TaskMergePlanStepResponse(
            kind=step.kind,
            task_id=step.task_id,
            branch_name=step.branch_name,
            worktree_path=str(step.worktree_path),
            upstream_ref=step.upstream_ref,
            base_branch=step.base_branch,
        )
        for step in plan_steps
    ]


def _raise_if_running_agents(
    *,
    has_running_agents: bool,
    allow_running: bool,
    message: str,
) -> None:
    if has_running_agents and not allow_running:
        raise RepoExecutorError(f"RUNNING_AGENTS: {message}", status_code=409)


async def _mark_merge_run_running(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    epic_id: int,
    requested_task_id: int,
    merge_run_task_id: int,
    host_key: str,
    canonical: bool,
    scope: str,
    allow_running: bool,
    force: bool,
    plan_snapshot: dict[str, object],
    operation: Literal["merge", "restack"],
) -> None:
    await upsert_merge_run(
        sessionmaker=sessionmaker,
        run_id=run_id,
        epic_id=epic_id,
        task_id=requested_task_id,
        host_key=host_key,
        canonical=canonical,
        scope=scope,
        allow_running=allow_running,
        force=force,
        plan_snapshot=plan_snapshot,
    )
    await emit_merge_run_event(
        sessionmaker=sessionmaker,
        run_id=run_id,
        task_id=merge_run_task_id,
        epic_id=epic_id,
        requested_task_id=requested_task_id,
        status=MergeRunStatus.Running,
        host_key=host_key,
        operation=operation,
    )


async def _run_local_merge_run(
    *,
    execute: Callable[[], Awaitable[None]],
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    task_id: int,
    operation: Literal["merge", "restack"],
    failure_kind: Literal["merge_ff", "rebase"],
    base_branch: str,
    host_key: str,
) -> None:
    try:
        await execute()
        await set_merge_run_status(
            sessionmaker=sessionmaker,
            run_id=run_id,
            status=MergeRunStatus.Succeeded,
        )
    except MergeBlockedByRunningAgents:
        return
    except Exception as e:
        if not isinstance(e, GitCommandError):
            await set_merge_run_status(
                sessionmaker=sessionmaker,
                run_id=run_id,
                status=MergeRunStatus.Failed,
                error=str(e),
            )
        await emit_task_merge_event(
            sessionmaker=sessionmaker,
            payload={
                "run_id": run_id,
                "task_id": task_id,
                "kind": failure_kind,
                "phase": "failed",
                "operation": operation,
                "branch_name": base_branch,
                "error": str(e),
            },
            host_key=host_key,
        )


def _spawn_local_merge_run(
    *,
    execute: Callable[[], Awaitable[None]],
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    task_id: int,
    operation: Literal["merge", "restack"],
    failure_kind: Literal["merge_ff", "rebase"],
    base_branch: str,
    host_key: str,
) -> None:
    asyncio.create_task(
        _run_local_merge_run(
            execute=execute,
            sessionmaker=sessionmaker,
            run_id=run_id,
            task_id=task_id,
            operation=operation,
            failure_kind=failure_kind,
            base_branch=base_branch,
            host_key=host_key,
        )
    )


async def _require_epic_id_for_task(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    task_id: int,
) -> int:
    async with sessionmaker() as session:
        task_row = await session.get(Task, task_id)
        if task_row is None:
            raise RepoExecutorError("Task not found", status_code=404)
        return task_row.epic_id


def _daemon_placeholder_plan_snapshot(
    *,
    operation: Literal["merge", "restack"],
    scope: Literal["descendants", "spine"],
    restack_mode: Literal["strict", "merge_then_restack"],
) -> dict[str, object]:
    return MergeRunPlanData(
        operation=operation,
        base_branch="",
        base_worktree="",
        scope=scope,
        restack_mode=restack_mode,
        spine_task_ids=[],
        affected_task_ids=[],
        steps=[],
    ).model_dump(mode="python")


class _DaemonRepoPlanAckData(BaseModel):
    model_config = ConfigDict(extra="allow")

    plan: MergeRunPlanData = Field(
        validation_alias=AliasChoices("plan", "plan_snapshot"),
    )
    running_agents: bool = False


@dataclass(frozen=True, slots=True)
class LocalRepoExecutor:
    ctx: RepoContext
    sessionmaker: async_sessionmaker[AsyncSession]

    async def merge(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskMergeRequest,
    ) -> TaskMergeResponse:
        run_id = _require_run_id(run_id=request.run_id)

        scope: Literal["descendants", "spine"] = (
            request.scope if request.cascade else "spine"
        )
        restack_mode: Literal["strict", "merge_then_restack"] = request.restack_mode
        try:
            plan = await build_merge_cascade_plan(
                ctx=self.ctx,
                sessionmaker=self.sessionmaker,
                task_id=task_id,
                run_id=run_id,
                scope=scope,
                restack_mode=restack_mode,
                force=request.force,
            )
        except MergePlanError as e:
            raise RepoExecutorError(str(e), status_code=400) from e

        _raise_if_running_agents(
            has_running_agents=bool(plan.running_agents),
            allow_running=request.allow_running,
            message="Merge affects running tasks; retry with allowRunning=true once you confirm.",
        )

        if request.dry_run:
            steps = _plan_steps_response(plan_steps=plan.steps)
            return TaskMergeResponse(
                run_id=run_id,
                dry_run=True,
                base_branch=plan.base_branch,
                steps=steps,
            )

        plan_snapshot = merge_run_plan_snapshot(plan=plan, operation="merge")
        merge_task_id = plan.spine_task_ids[-1] if plan.spine_task_ids else task_id
        await _mark_merge_run_running(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=plan.epic_id,
            requested_task_id=task_id,
            merge_run_task_id=merge_task_id,
            host_key=target.target_host_key,
            canonical=target.canonical,
            scope=scope,
            allow_running=request.allow_running,
            force=request.force,
            plan_snapshot=plan_snapshot,
            operation="merge",
        )

        _spawn_local_merge_run(
            execute=lambda: execute_merge_cascade_plan(
                ctx=self.ctx,
                sessionmaker=self.sessionmaker,
                plan=plan,
                allow_running=request.allow_running,
                emit_event=lambda payload: emit_task_merge_event(
                    sessionmaker=self.sessionmaker,
                    payload={**payload, "operation": "merge"},
                    host_key=target.target_host_key,
                ),
                update_run=lambda update: record_merge_run_step_update(
                    sessionmaker=self.sessionmaker,
                    run_id=run_id,
                    update=update,
                ),
            ),
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            operation="merge",
            failure_kind="merge_ff",
            base_branch=plan.base_branch,
            host_key=target.target_host_key,
        )
        return TaskMergeResponse(
            run_id=run_id,
            dry_run=False,
            base_branch=plan.base_branch,
        )

    async def restack(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskRestackRequest,
    ) -> TaskRestackResponse:
        run_id = _require_run_id(run_id=request.run_id)

        scope: Literal["descendants", "spine"] = request.scope
        try:
            plan = await build_restack_plan(
                ctx=self.ctx,
                sessionmaker=self.sessionmaker,
                task_id=task_id,
                run_id=run_id,
                scope=scope,
            )
        except MergePlanError as e:
            raise RepoExecutorError(str(e), status_code=400) from e

        _raise_if_running_agents(
            has_running_agents=bool(plan.running_agents),
            allow_running=request.allow_running,
            message="Restack affects running tasks; retry with allowRunning=true once you confirm.",
        )

        if request.dry_run:
            steps = _plan_steps_response(plan_steps=plan.steps)
            return TaskRestackResponse(
                run_id=run_id,
                dry_run=True,
                base_branch=plan.base_branch,
                steps=steps,
            )

        plan_snapshot = merge_run_plan_snapshot(plan=plan, operation="restack")
        await _mark_merge_run_running(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=plan.epic_id,
            requested_task_id=task_id,
            merge_run_task_id=task_id,
            host_key=target.target_host_key,
            canonical=target.canonical,
            scope=scope,
            allow_running=request.allow_running,
            force=False,
            plan_snapshot=plan_snapshot,
            operation="restack",
        )

        _spawn_local_merge_run(
            execute=lambda: execute_restack_plan(
                ctx=self.ctx,
                sessionmaker=self.sessionmaker,
                plan=plan,
                allow_running=request.allow_running,
                emit_event=lambda payload: emit_task_merge_event(
                    sessionmaker=self.sessionmaker,
                    payload={**payload, "operation": "restack"},
                    host_key=target.target_host_key,
                ),
                update_run=lambda update: record_merge_run_step_update(
                    sessionmaker=self.sessionmaker,
                    run_id=run_id,
                    update=update,
                ),
            ),
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            operation="restack",
            failure_kind="rebase",
            base_branch=plan.base_branch,
            host_key=target.target_host_key,
        )
        return TaskRestackResponse(
            run_id=run_id,
            dry_run=False,
            base_branch=plan.base_branch,
        )

    async def resume_merge_run(
        self,
        *,
        target: RepoExecutorTarget,
        run: MergeRun,
        request: MergeRunResumeRequest,
    ) -> MergeRunResumeResponse:
        run_id = run.run_id
        task_id = run.requested_task_id
        scope_value = run.scope
        force = run.force
        allow_running = run.allow_running or request.allow_running
        operation: Literal["merge", "restack"] = run.operation
        restack_mode: Literal["strict", "merge_then_restack"] = run.restack_mode

        blocked_step_index = run.blocked_step_index
        blocked_step_kind = run.blocked_step_kind
        blocked_branch_name = run.blocked_branch_name

        normalized_scope = scope_value.strip().lower()
        if normalized_scope not in {"descendants", "spine"}:
            raise RepoExecutorError(
                f"Invalid merge run scope: {scope_value!r}", status_code=400
            )

        merge_plan: MergeCascadePlan | None = None
        restack_plan: RestackPlan | None = None
        try:
            if operation == "restack":
                restack_plan = await build_restack_plan(
                    ctx=self.ctx,
                    sessionmaker=self.sessionmaker,
                    task_id=task_id,
                    run_id=run_id,
                    scope=cast(Literal["descendants", "spine"], normalized_scope),
                )
            else:
                merge_plan = await build_merge_cascade_plan(
                    ctx=self.ctx,
                    sessionmaker=self.sessionmaker,
                    task_id=task_id,
                    run_id=run_id,
                    scope=cast(Literal["descendants", "spine"], normalized_scope),
                    restack_mode=restack_mode,
                    force=force,
                )
        except MergePlanError as e:
            raise RepoExecutorError(str(e), status_code=400) from e

        plan: MergeCascadePlan | RestackPlan | None = (
            restack_plan if operation == "restack" else merge_plan
        )
        if plan is None:
            raise RepoExecutorError("Failed to rebuild merge run plan", status_code=500)

        _raise_if_running_agents(
            has_running_agents=bool(plan.running_agents),
            allow_running=allow_running,
            message="Merge affects running tasks; retry with allowRunning=true once you confirm.",
        )

        start_at_step_index = blocked_step_index or 0
        if start_at_step_index >= len(plan.steps):
            start_at_step_index = 0
        if start_at_step_index and blocked_step_kind and blocked_branch_name:
            step = plan.steps[start_at_step_index]
            if (
                step.kind != blocked_step_kind
                or step.branch_name != blocked_branch_name
            ):
                start_at_step_index = 0

        plan_snapshot = merge_run_plan_snapshot(plan=plan, operation=operation)
        if operation == "merge":
            assert merge_plan is not None
            merge_task_id = (
                merge_plan.spine_task_ids[-1] if merge_plan.spine_task_ids else task_id
            )
        else:
            merge_task_id = task_id
        await _mark_merge_run_running(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=plan.epic_id,
            requested_task_id=task_id,
            merge_run_task_id=merge_task_id,
            host_key=target.target_host_key,
            canonical=run.canonical,
            scope=normalized_scope,
            allow_running=allow_running,
            force=force,
            plan_snapshot=plan_snapshot,
            operation=operation,
        )

        _spawn_local_merge_run(
            execute=(
                (
                    lambda: execute_restack_plan(
                        ctx=self.ctx,
                        sessionmaker=self.sessionmaker,
                        plan=cast(RestackPlan, plan),
                        allow_running=allow_running,
                        emit_event=lambda payload: emit_task_merge_event(
                            sessionmaker=self.sessionmaker,
                            payload={**payload, "operation": "restack"},
                            host_key=target.target_host_key,
                        ),
                        update_run=lambda update: record_merge_run_step_update(
                            sessionmaker=self.sessionmaker,
                            run_id=run_id,
                            update=update,
                        ),
                        start_at_step_index=start_at_step_index,
                    )
                )
                if operation == "restack"
                else (
                    lambda: execute_merge_cascade_plan(
                        ctx=self.ctx,
                        sessionmaker=self.sessionmaker,
                        plan=cast(MergeCascadePlan, plan),
                        allow_running=allow_running,
                        emit_event=lambda payload: emit_task_merge_event(
                            sessionmaker=self.sessionmaker,
                            payload={**payload, "operation": "merge"},
                            host_key=target.target_host_key,
                        ),
                        update_run=lambda update: record_merge_run_step_update(
                            sessionmaker=self.sessionmaker,
                            run_id=run_id,
                            update=update,
                        ),
                        start_at_step_index=start_at_step_index,
                    )
                )
            ),
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            operation=operation,
            failure_kind="rebase" if operation == "restack" else "merge_ff",
            base_branch=plan.base_branch,
            host_key=target.target_host_key,
        )
        return MergeRunResumeResponse(run_id=run_id, base_branch=plan.base_branch)


@dataclass(frozen=True, slots=True)
class DaemonRepoExecutor:
    sessionmaker: async_sessionmaker[AsyncSession]
    daemon_connections: DaemonConnectionRegistry

    async def _daemon_supports_repo_plans(self, host_key: str) -> bool:
        presence = (await self.daemon_connections.snapshot()).get(host_key)
        if presence is None:
            return False
        return presence.capabilities.get("repo_plan_v1") is True

    async def _wait_for_command_final_state(
        self,
        *,
        command_id: int,
        timeout_s: float,
    ) -> DaemonCommand:
        deadline = time.monotonic() + timeout_s
        while True:
            async with self.sessionmaker() as session:
                row = await session.get(DaemonCommand, command_id)
            if row is None:
                raise RepoExecutorError("Daemon command not found", status_code=500)
            if row.state in {
                CommandState.Succeeded,
                CommandState.Failed,
                CommandState.Canceled,
            }:
                return row
            if time.monotonic() >= deadline:
                raise RepoExecutorError(
                    "Timed out waiting for daemon response.", status_code=504
                )
            await asyncio.sleep(0.1)

    async def _compute_plan(
        self,
        *,
        target: RepoExecutorTarget,
        command_type: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> _DaemonRepoPlanAckData:
        if not await self.daemon_connections.is_connected(target.target_host_key):
            raise RepoExecutorError(
                "No connected daemon available to compute this plan.", status_code=503
            )
        if not await self._daemon_supports_repo_plans(target.target_host_key):
            raise RepoExecutorError(
                "Daemon does not support repo plan computation yet (repo_plan_v1=false).",
                status_code=501,
            )

        command = await self._enqueue_daemon_command(
            host_key=target.target_host_key,
            command_type=command_type,
            workspace_id=target.repo.workspace_id,
            repo_id=target.repo.repo_id,
            payload=payload,
        )
        row = await self._wait_for_command_final_state(
            command_id=command.id,
            timeout_s=timeout_s,
        )
        if row.state != CommandState.Succeeded:
            detail = (
                row.ack_data.get("detail") if isinstance(row.ack_data, dict) else None
            )
            raise RepoExecutorError(
                f"Daemon failed to compute plan{': ' + str(detail) if detail else ''}.",
                status_code=502,
            )
        try:
            return _DaemonRepoPlanAckData.model_validate(row.ack_data)
        except Exception as e:
            raise RepoExecutorError(
                "Daemon returned an invalid plan response.", status_code=502
            ) from e

    async def _enqueue_daemon_command(
        self,
        *,
        host_key: str,
        command_type: str,
        workspace_id: str | None,
        repo_id: str | None,
        payload: dict[str, Any],
    ) -> DaemonCommandResponse:
        async with self.sessionmaker() as session:
            cmd = DaemonCommand(
                host_key=host_key,
                command_type=command_type,
                workspace_id=workspace_id,
                repo_id=repo_id,
                data=payload,
            )
            session.add(cmd)
            await session.commit()
            await session.refresh(cmd)

        command = DaemonCommandResponse.model_validate(cmd, from_attributes=True)
        ws_command = ServerCommand(
            command_id=command.id,
            command_type=command.command_type,
            workspace_id=command.workspace_id,
            repo_id=command.repo_id,
            data=command.payload,
        )
        await self.daemon_connections.send(
            host_key,
            {
                "type": "command",
                "command": ws_command.model_dump(),
            },
        )
        return command

    async def merge(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskMergeRequest,
    ) -> TaskMergeResponse:
        run_id = _require_run_id(run_id=request.run_id)

        scope: Literal["descendants", "spine"] = (
            request.scope if request.cascade else "spine"
        )
        restack_mode: Literal["strict", "merge_then_restack"] = request.restack_mode

        if request.dry_run:
            plan_resp = await self._compute_plan(
                target=target,
                command_type="repo.merge_run.plan",
                payload={
                    "run_id": run_id,
                    "task_id": task_id,
                    "operation": "merge",
                    "scope": scope,
                    "restack_mode": restack_mode,
                    "force": request.force,
                },
                timeout_s=15.0,
            )
            _raise_if_running_agents(
                has_running_agents=plan_resp.running_agents,
                allow_running=request.allow_running,
                message="Merge affects running tasks; retry with allowRunning=true once you confirm.",
            )
            return TaskMergeResponse(
                run_id=run_id,
                dry_run=True,
                base_branch=plan_resp.plan.base_branch,
                steps=_plan_steps_response(plan_steps=plan_resp.plan.steps),
            )

        epic_id = await _require_epic_id_for_task(
            sessionmaker=self.sessionmaker,
            task_id=task_id,
        )
        base_branch: str | None = None
        plan_snapshot = _daemon_placeholder_plan_snapshot(
            operation="merge", scope=scope, restack_mode=restack_mode
        )
        if await self._daemon_supports_repo_plans(target.target_host_key):
            try:
                plan_resp = await self._compute_plan(
                    target=target,
                    command_type="repo.merge_run.plan",
                    payload={
                        "run_id": run_id,
                        "task_id": task_id,
                        "operation": "merge",
                        "scope": scope,
                        "restack_mode": restack_mode,
                        "force": request.force,
                    },
                    timeout_s=5.0,
                )
                _raise_if_running_agents(
                    has_running_agents=plan_resp.running_agents,
                    allow_running=request.allow_running,
                    message="Merge affects running tasks; retry with allowRunning=true once you confirm.",
                )
                plan_snapshot = plan_resp.plan.model_dump(mode="python")
                base_branch = plan_resp.plan.base_branch
            except RepoExecutorError:
                pass

        await _mark_merge_run_running(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=epic_id,
            requested_task_id=task_id,
            merge_run_task_id=task_id,
            host_key=target.target_host_key,
            canonical=target.canonical,
            scope=scope,
            allow_running=request.allow_running,
            force=request.force,
            plan_snapshot=plan_snapshot,
            operation="merge",
        )
        await self._enqueue_daemon_command(
            host_key=target.target_host_key,
            command_type="repo.merge_run.start",
            workspace_id=target.repo.workspace_id,
            repo_id=target.repo.repo_id,
            payload={
                "run_id": run_id,
                "task_id": task_id,
                "operation": "merge",
                "scope": scope,
                "restack_mode": restack_mode,
                "allow_running": request.allow_running,
                "force": request.force,
                "canonical": target.canonical,
            },
        )
        return TaskMergeResponse(run_id=run_id, dry_run=False, base_branch=base_branch)

    async def restack(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskRestackRequest,
    ) -> TaskRestackResponse:
        run_id = _require_run_id(run_id=request.run_id)
        scope: Literal["descendants", "spine"] = request.scope

        if request.dry_run:
            plan_resp = await self._compute_plan(
                target=target,
                command_type="repo.restack.plan",
                payload={
                    "run_id": run_id,
                    "task_id": task_id,
                    "operation": "restack",
                    "scope": scope,
                },
                timeout_s=15.0,
            )
            _raise_if_running_agents(
                has_running_agents=plan_resp.running_agents,
                allow_running=request.allow_running,
                message="Restack affects running tasks; retry with allowRunning=true once you confirm.",
            )
            return TaskRestackResponse(
                run_id=run_id,
                dry_run=True,
                base_branch=plan_resp.plan.base_branch,
                steps=_plan_steps_response(plan_steps=plan_resp.plan.steps),
            )

        epic_id = await _require_epic_id_for_task(
            sessionmaker=self.sessionmaker,
            task_id=task_id,
        )
        base_branch: str | None = None
        plan_snapshot = _daemon_placeholder_plan_snapshot(
            operation="restack", scope=scope, restack_mode="strict"
        )
        if await self._daemon_supports_repo_plans(target.target_host_key):
            try:
                plan_resp = await self._compute_plan(
                    target=target,
                    command_type="repo.restack.plan",
                    payload={
                        "run_id": run_id,
                        "task_id": task_id,
                        "operation": "restack",
                        "scope": scope,
                    },
                    timeout_s=5.0,
                )
                _raise_if_running_agents(
                    has_running_agents=plan_resp.running_agents,
                    allow_running=request.allow_running,
                    message="Restack affects running tasks; retry with allowRunning=true once you confirm.",
                )
                plan_snapshot = plan_resp.plan.model_dump(mode="python")
                base_branch = plan_resp.plan.base_branch
            except RepoExecutorError:
                pass

        await _mark_merge_run_running(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=epic_id,
            requested_task_id=task_id,
            merge_run_task_id=task_id,
            host_key=target.target_host_key,
            canonical=target.canonical,
            scope=scope,
            allow_running=request.allow_running,
            force=False,
            plan_snapshot=plan_snapshot,
            operation="restack",
        )
        await self._enqueue_daemon_command(
            host_key=target.target_host_key,
            command_type="repo.merge_run.start",
            workspace_id=target.repo.workspace_id,
            repo_id=target.repo.repo_id,
            payload={
                "run_id": run_id,
                "task_id": task_id,
                "operation": "restack",
                "scope": scope,
                "allow_running": request.allow_running,
                "canonical": target.canonical,
            },
        )
        return TaskRestackResponse(
            run_id=run_id, dry_run=False, base_branch=base_branch
        )

    async def resume_merge_run(
        self,
        *,
        target: RepoExecutorTarget,
        run: MergeRun,
        request: MergeRunResumeRequest,
    ) -> MergeRunResumeResponse:
        run_id = run.run_id
        base_branch: str | None = None

        if await self._daemon_supports_repo_plans(target.target_host_key):
            try:
                command_type = (
                    "repo.restack.plan"
                    if run.operation == "restack"
                    else "repo.merge_run.plan"
                )
                plan_resp = await self._compute_plan(
                    target=target,
                    command_type=command_type,
                    payload={
                        "run_id": run_id,
                        "task_id": run.requested_task_id,
                        "operation": run.operation,
                        "scope": run.scope,
                        "restack_mode": run.restack_mode,
                        "force": run.force,
                    },
                    timeout_s=5.0,
                )
                _raise_if_running_agents(
                    has_running_agents=plan_resp.running_agents,
                    allow_running=run.allow_running or request.allow_running,
                    message="Merge affects running tasks; retry with allowRunning=true once you confirm.",
                )
                plan_snapshot = plan_resp.plan.model_dump(mode="python")
                base_branch = plan_resp.plan.base_branch
            except RepoExecutorError:
                plan_snapshot = dict(run.plan) if isinstance(run.plan, dict) else {}
        else:
            plan_snapshot = dict(run.plan) if isinstance(run.plan, dict) else {}

        normalized_scope = run.scope.strip().lower()
        scope_literal: Literal["descendants", "spine"] = (
            "descendants" if normalized_scope == "descendants" else "spine"
        )
        if not plan_snapshot:
            plan_snapshot = _daemon_placeholder_plan_snapshot(
                operation=run.operation,
                scope=scope_literal,
                restack_mode=run.restack_mode,
            )
        await _mark_merge_run_running(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=run.epic_id,
            requested_task_id=run.requested_task_id,
            merge_run_task_id=run.requested_task_id,
            host_key=target.target_host_key,
            canonical=run.canonical,
            scope=run.scope,
            allow_running=run.allow_running or request.allow_running,
            force=run.force,
            plan_snapshot=plan_snapshot,
            operation=run.operation,
        )
        await self._enqueue_daemon_command(
            host_key=target.target_host_key,
            command_type="repo.merge_run.resume",
            workspace_id=target.repo.workspace_id,
            repo_id=target.repo.repo_id,
            payload={
                "run_id": run_id,
                "allow_running": run.allow_running or request.allow_running,
                "canonical": run.canonical,
            },
        )
        return MergeRunResumeResponse(run_id=run_id, base_branch=base_branch)


@dataclass(frozen=True, slots=True)
class RepoExecutorRouter:
    runner_mode: Literal["local", "remote"]
    local_host_key: str | None
    daemon_connections: DaemonConnectionRegistry
    local: LocalRepoExecutor | None
    daemon: DaemonRepoExecutor

    async def get_status(
        self,
        *,
        repo: Repository,
        primary_host_key: str | None,
    ) -> RepoExecutorStatus:
        attached: set[str] = set()
        if self.runner_mode == "local" and self.local_host_key is not None:
            attached.add(self.local_host_key)

        connected = await self.daemon_connections.snapshot()
        for host_key, presence in connected.items():
            if any(
                r.get("workspace_id") == repo.workspace_id
                and r.get("repo_id") == repo.repo_id
                for r in presence.attached_repos
            ):
                attached.add(host_key)

        return RepoExecutorStatus(
            primary_host_key=primary_host_key,
            attached_host_keys=sorted(attached),
        )

    async def merge(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskMergeRequest,
    ) -> TaskMergeResponse:
        impl = self._impl_for_target(target)
        return await impl.merge(target=target, task_id=task_id, request=request)

    async def restack(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskRestackRequest,
    ) -> TaskRestackResponse:
        impl = self._impl_for_target(target)
        return await impl.restack(target=target, task_id=task_id, request=request)

    async def resume_merge_run(
        self,
        *,
        target: RepoExecutorTarget,
        run: MergeRun,
        request: MergeRunResumeRequest,
    ) -> MergeRunResumeResponse:
        impl = self._impl_for_target(target)
        return await impl.resume_merge_run(target=target, run=run, request=request)

    def _impl_for_target(
        self, target: RepoExecutorTarget
    ) -> LocalRepoExecutor | DaemonRepoExecutor:
        if target.is_local and self.local is not None:
            return self.local
        return self.daemon


def make_repo_executor(
    *,
    runner_mode: Literal["local", "remote"],
    local_host_key: str | None,
    ctx: RepoContext,
    sessionmaker: async_sessionmaker[AsyncSession],
    daemon_connections: DaemonConnectionRegistry,
) -> RepoExecutor:
    local = (
        LocalRepoExecutor(ctx=ctx, sessionmaker=sessionmaker)
        if runner_mode == "local" and local_host_key is not None
        else None
    )
    daemon = DaemonRepoExecutor(
        sessionmaker=sessionmaker,
        daemon_connections=daemon_connections,
    )
    return RepoExecutorRouter(
        runner_mode=runner_mode,
        local_host_key=local_host_key,
        daemon_connections=daemon_connections,
        local=local,
        daemon=daemon,
    )
