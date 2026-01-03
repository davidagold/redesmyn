from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.context import RepoContext
from redesmyn.db import DaemonCommand, MergeRun, Repository, Task
from redesmyn.db.models import MergeRunPlanData
from redesmyn.domain.enums import MergeRunStatus
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
        run_id = request.run_id or ""
        if not run_id:
            raise RepoExecutorError("Missing run id", status_code=400)

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

        if plan.running_agents and not request.allow_running:
            raise RepoExecutorError(
                "Merge affects running tasks; retry with allowRunning=true once you confirm.",
                status_code=409,
            )

        if request.dry_run:
            steps = [
                TaskMergePlanStepResponse(
                    kind=step.kind,
                    task_id=step.task_id,
                    branch_name=step.branch_name,
                    worktree_path=str(step.worktree_path),
                    upstream_ref=step.upstream_ref,
                    base_branch=step.base_branch,
                )
                for step in plan.steps
            ]
            return TaskMergeResponse(
                run_id=run_id,
                dry_run=True,
                base_branch=plan.base_branch,
                steps=steps,
            )

        plan_snapshot = merge_run_plan_snapshot(plan=plan, operation="merge")
        await upsert_merge_run(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=plan.epic_id,
            task_id=task_id,
            host_key=target.target_host_key,
            canonical=target.canonical,
            scope=scope,
            allow_running=request.allow_running,
            force=request.force,
            plan_snapshot=plan_snapshot,
        )
        merge_task_id = plan.spine_task_ids[-1] if plan.spine_task_ids else task_id
        await emit_merge_run_event(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=merge_task_id,
            epic_id=plan.epic_id,
            requested_task_id=task_id,
            status=MergeRunStatus.Running,
            host_key=target.target_host_key,
            operation="merge",
        )

        async def _run() -> None:
            try:
                await execute_merge_cascade_plan(
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
                )
                await set_merge_run_status(
                    sessionmaker=self.sessionmaker,
                    run_id=run_id,
                    status=MergeRunStatus.Succeeded,
                )
            except MergeBlockedByRunningAgents:
                return
            except Exception as e:
                if not isinstance(e, GitCommandError):
                    await set_merge_run_status(
                        sessionmaker=self.sessionmaker,
                        run_id=run_id,
                        status=MergeRunStatus.Failed,
                        error=str(e),
                    )
                await emit_task_merge_event(
                    sessionmaker=self.sessionmaker,
                    payload={
                        "run_id": run_id,
                        "operation": "merge",
                        "task_id": task_id,
                        "kind": "merge_ff",
                        "phase": "failed",
                        "branch_name": plan.base_branch,
                        "error": str(e),
                    },
                    host_key=target.target_host_key,
                )

        asyncio.create_task(_run())
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
        run_id = request.run_id or ""
        if not run_id:
            raise RepoExecutorError("Missing run id", status_code=400)

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

        if plan.running_agents and not request.allow_running:
            raise RepoExecutorError(
                "Restack affects running tasks; retry with allowRunning=true once you confirm.",
                status_code=409,
            )

        if request.dry_run:
            steps = [
                TaskMergePlanStepResponse(
                    kind=step.kind,
                    task_id=step.task_id,
                    branch_name=step.branch_name,
                    worktree_path=str(step.worktree_path),
                    upstream_ref=step.upstream_ref,
                    base_branch=step.base_branch,
                )
                for step in plan.steps
            ]
            return TaskRestackResponse(
                run_id=run_id,
                dry_run=True,
                base_branch=plan.base_branch,
                steps=steps,
            )

        plan_snapshot = merge_run_plan_snapshot(plan=plan, operation="restack")
        await upsert_merge_run(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=plan.epic_id,
            task_id=task_id,
            host_key=target.target_host_key,
            canonical=target.canonical,
            scope=scope,
            allow_running=request.allow_running,
            force=False,
            plan_snapshot=plan_snapshot,
        )
        await emit_merge_run_event(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            epic_id=plan.epic_id,
            requested_task_id=task_id,
            status=MergeRunStatus.Running,
            operation="restack",
            host_key=target.target_host_key,
        )

        async def _run() -> None:
            try:
                await execute_restack_plan(
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
                )
                await set_merge_run_status(
                    sessionmaker=self.sessionmaker,
                    run_id=run_id,
                    status=MergeRunStatus.Succeeded,
                )
            except MergeBlockedByRunningAgents:
                return
            except Exception as e:
                if not isinstance(e, GitCommandError):
                    await set_merge_run_status(
                        sessionmaker=self.sessionmaker,
                        run_id=run_id,
                        status=MergeRunStatus.Failed,
                        error=str(e),
                    )
                await emit_task_merge_event(
                    sessionmaker=self.sessionmaker,
                    payload={
                        "run_id": run_id,
                        "operation": "restack",
                        "task_id": task_id,
                        "kind": "rebase",
                        "phase": "failed",
                        "branch_name": plan.base_branch,
                        "error": str(e),
                    },
                    host_key=target.target_host_key,
                )

        asyncio.create_task(_run())
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

        if plan.running_agents and not allow_running:
            raise RepoExecutorError(
                "Merge affects running tasks; retry with allowRunning=true once you confirm.",
                status_code=409,
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
        await upsert_merge_run(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=plan.epic_id,
            task_id=task_id,
            host_key=target.target_host_key,
            canonical=run.canonical,
            scope=normalized_scope,
            allow_running=allow_running,
            force=force,
            plan_snapshot=plan_snapshot,
        )
        if operation == "merge":
            assert merge_plan is not None
            merge_task_id = (
                merge_plan.spine_task_ids[-1] if merge_plan.spine_task_ids else task_id
            )
        else:
            merge_task_id = task_id
        await emit_merge_run_event(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=merge_task_id,
            epic_id=plan.epic_id,
            requested_task_id=task_id,
            status=MergeRunStatus.Running,
            operation=operation,
            host_key=target.target_host_key,
        )

        async def _run() -> None:
            try:
                if operation == "restack":
                    await execute_restack_plan(
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
                else:
                    await execute_merge_cascade_plan(
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
                await set_merge_run_status(
                    sessionmaker=self.sessionmaker,
                    run_id=run_id,
                    status=MergeRunStatus.Succeeded,
                )
            except MergeBlockedByRunningAgents:
                return
            except Exception as e:
                if not isinstance(e, GitCommandError):
                    await set_merge_run_status(
                        sessionmaker=self.sessionmaker,
                        run_id=run_id,
                        status=MergeRunStatus.Failed,
                        error=str(e),
                    )
                await emit_task_merge_event(
                    sessionmaker=self.sessionmaker,
                    payload={
                        "run_id": run_id,
                        "task_id": task_id,
                        "kind": "merge_ff" if operation == "merge" else "rebase",
                        "phase": "failed",
                        "operation": operation,
                        "branch_name": plan.base_branch,
                        "error": str(e),
                    },
                    host_key=target.target_host_key,
                )

        asyncio.create_task(_run())
        return MergeRunResumeResponse(run_id=run_id, base_branch=plan.base_branch)


@dataclass(frozen=True, slots=True)
class DaemonRepoExecutor:
    sessionmaker: async_sessionmaker[AsyncSession]
    daemon_connections: DaemonConnectionRegistry

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
        run_id = request.run_id or ""
        if not run_id:
            raise RepoExecutorError("Missing run id", status_code=400)
        if request.dry_run:
            raise RepoExecutorError(
                "dryRun is not supported when routing to a remote repo executor.",
                status_code=501,
            )

        scope: Literal["descendants", "spine"] = (
            request.scope if request.cascade else "spine"
        )
        restack_mode: Literal["strict", "merge_then_restack"] = request.restack_mode

        async with self.sessionmaker() as session:
            task_row = await session.get(Task, task_id)
            if task_row is None:
                raise RepoExecutorError("Task not found", status_code=404)
            epic_id = task_row.epic_id

        plan_snapshot = MergeRunPlanData(
            operation="merge",
            base_branch="",
            base_worktree="",
            scope=scope,
            restack_mode=restack_mode,
            spine_task_ids=[],
            affected_task_ids=[],
            steps=[],
        ).model_dump(mode="python")
        await upsert_merge_run(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=epic_id,
            task_id=task_id,
            host_key=target.target_host_key,
            canonical=target.canonical,
            scope=scope,
            allow_running=request.allow_running,
            force=request.force,
            plan_snapshot=plan_snapshot,
        )
        await emit_merge_run_event(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            epic_id=epic_id,
            requested_task_id=task_id,
            status=MergeRunStatus.Running,
            host_key=target.target_host_key,
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
        return TaskMergeResponse(run_id=run_id, dry_run=False, base_branch=None)

    async def restack(
        self,
        *,
        target: RepoExecutorTarget,
        task_id: int,
        request: TaskRestackRequest,
    ) -> TaskRestackResponse:
        run_id = request.run_id or ""
        if not run_id:
            raise RepoExecutorError("Missing run id", status_code=400)
        if request.dry_run:
            raise RepoExecutorError(
                "dryRun is not supported when routing to a remote repo executor.",
                status_code=501,
            )

        scope: Literal["descendants", "spine"] = request.scope
        async with self.sessionmaker() as session:
            task_row = await session.get(Task, task_id)
            if task_row is None:
                raise RepoExecutorError("Task not found", status_code=404)
            epic_id = task_row.epic_id

        plan_snapshot = MergeRunPlanData(
            operation="restack",
            base_branch="",
            base_worktree="",
            scope=scope,
            restack_mode="strict",
            spine_task_ids=[],
            affected_task_ids=[],
            steps=[],
        ).model_dump(mode="python")
        await upsert_merge_run(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=epic_id,
            task_id=task_id,
            host_key=target.target_host_key,
            canonical=target.canonical,
            scope=scope,
            allow_running=request.allow_running,
            force=False,
            plan_snapshot=plan_snapshot,
        )
        await emit_merge_run_event(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            epic_id=epic_id,
            requested_task_id=task_id,
            status=MergeRunStatus.Running,
            host_key=target.target_host_key,
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
        return TaskRestackResponse(run_id=run_id, dry_run=False, base_branch=None)

    async def resume_merge_run(
        self,
        *,
        target: RepoExecutorTarget,
        run: MergeRun,
        request: MergeRunResumeRequest,
    ) -> MergeRunResumeResponse:
        run_id = run.run_id
        if run.plan is None:
            raise RepoExecutorError("Merge run plan is missing", status_code=400)

        plan_snapshot = (
            dict(run.plan) if isinstance(run.plan, dict) else None
        ) or MergeRunPlanData(
            operation=run.operation,
            base_branch="",
            base_worktree="",
            scope="spine",
        ).model_dump(mode="python")

        await upsert_merge_run(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            epic_id=run.epic_id,
            task_id=run.requested_task_id,
            host_key=target.target_host_key,
            canonical=run.canonical,
            scope=run.scope,
            allow_running=run.allow_running or request.allow_running,
            force=run.force,
            plan_snapshot=plan_snapshot,
        )
        await emit_merge_run_event(
            sessionmaker=self.sessionmaker,
            run_id=run_id,
            task_id=run.requested_task_id,
            epic_id=run.epic_id,
            requested_task_id=run.requested_task_id,
            status=MergeRunStatus.Running,
            host_key=target.target_host_key,
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
        return MergeRunResumeResponse(run_id=run_id, base_branch=None)


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
