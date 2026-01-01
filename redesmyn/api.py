from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, overload
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.responses import RedirectResponse

from redesmyn.agent_prelude import DEFAULT_AGENT_PRELUDE_TEMPLATE
from redesmyn.agent_monitor import run_agent_monitor
from redesmyn.agent_runtime import StartAgentResult, agent_log_path_for_row
from redesmyn.context import RepoContext, get_repo_context
from redesmyn.db import (
    Agent,
    Block,
    BlockScope,
    Epic,
    Event,
    HarnessProfile,
    Host,
    LinearAuth,
    MergeRun,
    Node,
    Repository,
    Task,
    create_engine,
    create_sessionmaker,
)
from redesmyn.db.models import HostCapabilities, MergeRunPlanData, MergeRunPlanStepData
from redesmyn.domain.enums import BlockPolicy, MergeRunStatus
from redesmyn.event_stream import run_event_stream
from redesmyn.integrations.linear import (
    exchange_code_for_token,
    linear_authorize_url,
    linear_redirect_uri,
    new_oauth_state,
)
from redesmyn.orchestrator import init_repo
from redesmyn.orchestration_config import (
    load_orchestration_defaults,
    read_config_file,
    repo_config_path,
    set_config_value,
    write_config,
)
from redesmyn.repo import (
    GitCommandError,
    branch_exists,
    git_commit_info,
    git_is_ancestor,
    git_merge_base,
    git_rev_list,
)
from redesmyn.repo_observer import run_repo_observer
from redesmyn.runner_backend import (
    RunnerBackend,
    RunnerBackendError,
    make_runner_backend,
)
from redesmyn.sandbox import make_sandbox_provider
from redesmyn.schemas.core import (
    ApiStatusResponse,
    AgentResponse,
    AttachInfoResponse,
    BlockScopeResponse,
    BlockStatusResponse,
    EpicGraphResponse,
    EpicResponse,
    HarnessProfileDefinitionResponse,
    HarnessProfileResponse,
    HarnessProfileUpsertRequest,
    HostResponse,
    HostUpsertRequest,
    LinearStatusResponse,
    MergeRunResumeRequest,
    MergeRunResumeResponse,
    MergeRunSummaryResponse,
    NodeSetAgentRequest,
    NodeResponse,
    OrchestrationDefaultsResponse,
    OrchestrationFleetDefaultsResponse,
    OrchestrationHarnessDefaultsResponse,
    OrchestrationSandboxDefaultsResponse,
    OrchestrationDefaultsUpdateRequest,
    ReleaseConditionResponse,
    SandboxCapabilitiesResponse,
    TaskResponse,
    TaskAgentRestartRequest,
    TaskAgentStartRequest,
    TaskAgentStartResponse,
    TaskAgentStopResponse,
    TaskAgentBulkActionItemRequest,
    TaskAgentBulkActionRequest,
    TaskAgentBulkActionResponse,
    TaskAgentBulkRunRequest,
    TaskAgentBulkRunResponse,
    TaskMergeReadyRequest,
    TaskMergeRequest,
    TaskMergeResponse,
    TaskMergePlanStepResponse,
    TrunkCommitResponse,
    TrunkTimelineResponse,
)
from redesmyn.settings import load_settings
from redesmyn.git_mechanics_v0 import (
    MergeBlockedByRunningAgents,
    MergePlanError,
    MergeRunStepUpdate,
    build_merge_cascade_plan,
    execute_merge_cascade_plan,
)


class AppState(Protocol):
    ctx: RepoContext
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    runner_backend: RunnerBackend
    linear_oauth_states: dict[str, datetime]


class App(FastAPI):
    state: AppState


class DashboardStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            if path and "." in Path(path).name:
                raise
            return await super().get_response("index.html", scope)


@asynccontextmanager
async def lifespan(app: App):
    ctx = get_repo_context()
    await init_repo(ctx)
    settings = load_settings(repo_root=ctx.repo_root)

    app.state.ctx = ctx
    app.state.engine = create_engine(ctx.db_path)
    app.state.sessionmaker = create_sessionmaker(app.state.engine)
    app.state.runner_backend = make_runner_backend(mode=settings.runner_mode, ctx=ctx)
    app.state.linear_oauth_states = {}
    maybe_mount_dashboard(app, ctx.worktree_root)

    observer_task: asyncio.Task[None] | None = None
    agent_monitor_task: asyncio.Task[None] | None = None
    if os.environ.get("REDESMYN_NO_OBSERVER") not in {"1", "true", "TRUE"}:
        observer_task = asyncio.create_task(
            run_repo_observer(
                ctx,
                interval_s=1.0,
                emit_baseline=False,
                once=False,
            )
        )

    if os.environ.get("REDESMYN_NO_AGENT_MONITOR") not in {"1", "true", "TRUE"}:
        agent_monitor_task = asyncio.create_task(
            run_agent_monitor(
                ctx,
                app.state.sessionmaker,
                interval_s=1.0,
                once=False,
            )
        )

    yield

    if observer_task is not None:
        observer_task.cancel()
        try:
            await observer_task
        except asyncio.CancelledError:
            pass

    if agent_monitor_task is not None:
        agent_monitor_task.cancel()
        try:
            await agent_monitor_task
        except asyncio.CancelledError:
            pass

    await app.state.engine.dispose()


def maybe_mount_dashboard(app_: FastAPI, worktree_root: Path) -> None:
    if (dist_path := _dist_path(worktree_root)).is_dir():
        app_.mount(
            "/", DashboardStaticFiles(directory=str(dist_path)), name="dashboard"
        )


app = App(title="Redesmyn", lifespan=lifespan)
v1 = APIRouter(prefix="/v1")


def _tail_text(path: Path, *, max_bytes: int = 65536) -> tuple[str, bool]:
    try:
        size = path.stat().st_size
    except OSError:
        size = None

    try:
        with path.open("rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                actual_size = f.tell()
                offset = max(0, actual_size - max_bytes)
                f.seek(offset, os.SEEK_SET)
            except OSError:
                pass
            data = f.read()
    except OSError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    truncated = False
    if size is not None and size > max_bytes:
        truncated = True

    text = data.decode("utf-8", errors="replace")
    return text, truncated


@v1.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@v1.websocket("/ws")
async def v1_ws(
    websocket: WebSocket,
    epic: str | None = None,
    after_id: int | None = None,
) -> None:
    """WebSocket event stream (v0: activity + presence)."""
    await websocket.accept()
    try:
        await run_event_stream(
            websocket,
            app.state.sessionmaker,
            epic=epic,
            after_id=after_id,
        )
    except WebSocketDisconnect:
        return


async def _repo_id(session: AsyncSession) -> int | None:
    repo = await session.scalar(
        select(Repository).where(Repository.repo_root == str(app.state.ctx.repo_root))
    )
    return repo.id if repo is not None else None


async def _resolve_epic_row(session: AsyncSession, *, epic: str) -> Epic:
    repo_id = await _repo_id(session)
    if repo_id is None:
        raise HTTPException(status_code=404, detail="Repository not initialized")

    if epic.isdigit():
        row = await session.get(Epic, int(epic))
        if row is None or row.repository_id != repo_id:
            raise HTTPException(status_code=404, detail="Epic not found")
        return row

    row = await session.scalar(
        select(Epic).where(Epic.repository_id == repo_id, Epic.slug == epic)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Epic not found")
    return row


@v1.get("/epics", response_model=list[EpicResponse])
async def list_epics() -> list[EpicResponse]:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        repo_id = await _repo_id(session)
        if repo_id is None:
            return []
        rows = await session.scalars(
            select(Epic).where(Epic.repository_id == repo_id).order_by(Epic.id)
        )
        return [EpicResponse.model_validate(r, from_attributes=True) for r in rows]


@v1.get("/epics/{epic}", response_model=EpicResponse)
async def get_epic(epic: str) -> EpicResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        row = await _resolve_epic_row(session, epic=epic)
        return EpicResponse.model_validate(row, from_attributes=True)


@v1.get("/epics/{epic}/graph", response_model=EpicGraphResponse)
async def epic_graph(epic: str) -> EpicGraphResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        epic_row = await _resolve_epic_row(session, epic=epic)
        tasks = list(
            await session.scalars(
                select(Task).where(Task.epic_id == epic_row.id).order_by(Task.id)
            )
        )
        nodes = list(
            await session.scalars(
                select(Node).where(Node.epic_id == epic_row.id).order_by(Node.id)
            )
        )
        agent_ids = {n.agent_id for n in nodes if n.agent_id is not None}
        agents = (
            list(
                await session.scalars(
                    select(Agent)
                    .where(Agent.id.in_(list(agent_ids)))
                    .where(Agent.started_at.is_not(None))
                    .order_by(Agent.id)
                )
            )
            if agent_ids
            else []
        )
        merge_runs = list(
            await session.scalars(
                select(MergeRun)
                .where(MergeRun.epic_id == epic_row.id)
                .where(
                    MergeRun.status.in_(
                        [
                            MergeRunStatus.Running,
                            MergeRunStatus.Blocked,
                            MergeRunStatus.Resumable,
                        ]
                    )
                )
                .order_by(desc(MergeRun.id))
            )
        )

    stack_in_sync_by_node_id: dict[int, bool | None] = {}
    try:
        repo_root = app.state.ctx.repo_root
        nodes_by_id: dict[int, Node] = {n.id: n for n in nodes}
        for node in nodes:
            if not branch_exists(repo_root, node.branch_name):
                stack_in_sync_by_node_id[node.id] = None
                continue
            if node.parent_node_id is None:
                upstream = epic_row.root_branch
            else:
                parent = nodes_by_id.get(node.parent_node_id)
                upstream = parent.branch_name if parent is not None else None
            if upstream is None or not branch_exists(repo_root, upstream):
                stack_in_sync_by_node_id[node.id] = None
                continue
            stack_in_sync_by_node_id[node.id] = git_is_ancestor(
                repo_root, upstream, node.branch_name
            )
    except Exception:
        stack_in_sync_by_node_id = {}

    trunk: TrunkTimelineResponse | None = None
    try:
        repo_root = app.state.ctx.repo_root
        root_branch = epic_row.root_branch
        commits = git_rev_list(repo_root, root_branch, first_parent=True)
        base_sha = commits[0] if commits else None
        if commits:
            root_nodes = [n for n in nodes if n.parent_node_id is None]
            merge_bases = {
                mb
                for mb in (
                    git_merge_base(repo_root, root_branch, n.branch_name)
                    for n in root_nodes
                )
                if mb
            }
            if merge_bases:
                for sha in commits:
                    if sha in merge_bases:
                        base_sha = sha
                        break

            if base_sha:
                try:
                    base_index = commits.index(base_sha)
                except ValueError:
                    base_index = 0
                    base_sha = commits[0]

                limit = 4
                newer = commits[:base_index]
                older = commits[base_index + 1 :]

                commits_before = older[:limit]
                commits_after = list(reversed(newer))[:limit]

                commit_info = git_commit_info(
                    repo_root, [base_sha, *commits_before, *commits_after]
                )

                def build_commit(sha: str) -> TrunkCommitResponse:
                    info = commit_info.get(sha, {})
                    authored_at: datetime | None = None
                    authored_at_raw = info.get("authored_at")
                    if isinstance(authored_at_raw, str):
                        try:
                            authored_at = datetime.fromisoformat(authored_at_raw)
                        except ValueError:
                            authored_at = None
                    return TrunkCommitResponse(
                        sha=sha,
                        author_name=info.get("author_name"),
                        author_email=info.get("author_email"),
                        authored_at=authored_at,
                    )

                trunk = TrunkTimelineResponse(
                    base_sha=base_sha,
                    base_commit=build_commit(base_sha),
                    commits_before=[build_commit(sha) for sha in commits_before],
                    commits_after=[build_commit(sha) for sha in commits_after],
                    has_more_before=len(older) > limit,
                    has_more_after=len(newer) > limit,
                )
    except Exception:
        trunk = None

    node_responses: list[NodeResponse] = []
    for node in nodes:
        resp = NodeResponse.model_validate(node, from_attributes=True)
        resp.stack_in_sync = stack_in_sync_by_node_id.get(node.id)
        node_responses.append(resp)

    return EpicGraphResponse(
        epic=EpicResponse.model_validate(epic_row, from_attributes=True),
        tasks=[TaskResponse.model_validate(t, from_attributes=True) for t in tasks],
        nodes=node_responses,
        agents=[AgentResponse.model_validate(a, from_attributes=True) for a in agents],
        merge_runs=[
            MergeRunSummaryResponse.model_validate(r, from_attributes=True)
            for r in merge_runs
        ],
        trunk=trunk,
    )


@v1.get("/hosts", response_model=list[HostResponse])
async def list_hosts() -> list[HostResponse]:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        rows = list(await session.scalars(select(Host).order_by(Host.id)))
    return [HostResponse.model_validate(r, from_attributes=True) for r in rows]


@v1.post("/hosts/upsert", response_model=HostResponse)
async def upsert_host(request: HostUpsertRequest) -> HostResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        row = await session.scalar(
            select(Host).where(Host.host_key == request.host_key)
        )
        if row is None:
            row = Host(
                host_key=request.host_key,
                display_name=request.display_name,
                capabilities=(
                    request.capabilities.model_dump(mode="python")
                    if request.capabilities is not None
                    else HostCapabilities().model_dump(mode="python")
                ),
                last_seen_at=datetime.now(UTC),
            )
            session.add(row)
        else:
            row.display_name = request.display_name
            if request.capabilities is not None:
                row.capabilities = request.capabilities.model_dump(mode="python")
            row.last_seen_at = datetime.now(UTC)

        await session.commit()
        await session.refresh(row)
        return HostResponse.model_validate(row, from_attributes=True)


@v1.get("/harness-profiles", response_model=list[HarnessProfileResponse])
async def list_harness_profiles() -> list[HarnessProfileResponse]:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        rows = list(
            await session.scalars(select(HarnessProfile).order_by(HarnessProfile.id))
        )
    return [
        HarnessProfileResponse.model_validate(r, from_attributes=True) for r in rows
    ]


@v1.post("/harness-profiles", response_model=HarnessProfileResponse)
async def upsert_harness_profile(
    request: HarnessProfileUpsertRequest,
) -> HarnessProfileResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        row = await session.get(HarnessProfile, request.id)
        if row is None:
            row = HarnessProfile(
                id=request.id,
                kind=request.kind,
                source=request.source,
                display_name=request.display_name,
                definition=request.definition.model_dump(mode="python"),
            )
            session.add(row)
        else:
            row.kind = request.kind
            row.source = request.source
            row.display_name = request.display_name
            row.definition = request.definition.model_dump(mode="python")

        await session.commit()
        await session.refresh(row)
        return HarnessProfileResponse.model_validate(row, from_attributes=True)


async def _require_task_node(
    session: AsyncSession, *, task_id: int
) -> tuple[Task, Node]:
    task = await session.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.node_id is None:
        raise HTTPException(status_code=409, detail="Task is not mapped to a node")
    node = await session.get(Node, task.node_id)
    if node is None:
        raise HTTPException(status_code=409, detail="Node not found for task")
    return task, node


@v1.post("/tasks/{task_id}/agent/start", response_model=TaskAgentStartResponse)
async def start_task_agent(
    task_id: int,
    request: TaskAgentStartRequest,
) -> TaskAgentStartResponse:
    run_id = uuid4().hex
    try:
        _, node, result, warnings = await _perform_task_agent_action(
            sessionmaker=app.state.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            action="start",
            harness=request.harness,
            detach=request.detach,
            prelude=request.prelude,
        )
    except RunnerBackendError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    row = result.agent  # type: ignore[attr-defined]

    return TaskAgentStartResponse(
        task_id=task_id,
        node_id=node.id,
        agent_id=row.id,
        agent_name=row.display_name,
        agent_status=row.status,
        harness_profile_id=row.harness_profile_id or "",
        attach=TypeAdapter(AttachInfoResponse).validate_python(row.attach),
        resolved_profile=TypeAdapter(
            HarnessProfileDefinitionResponse | None
        ).validate_python(row.resolved_profile),
        started_at=row.started_at or datetime.now(UTC),
        started=result.started,  # type: ignore[attr-defined]
        warnings=warnings,
    )


async def _emit_task_agent_run_event(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    node_id: int,
    task_id: int,
    action: str,
    phase: str,
    agent_id: int | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "run_id": run_id,
        "node_id": node_id,
        "task_id": task_id,
        "action": action,
        "phase": phase,
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    if warnings is not None:
        payload["warnings"] = warnings
    if error is not None:
        payload["error"] = error

    async with sessionmaker() as session:
        session.add(
            Event(
                event_type="task.agent_run",
                data=payload,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def _emit_task_agent_action_event(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    node_id: int,
    task_id: int,
    action: str,
    phase: str,
    agent_id: int | None = None,
    stopped: bool | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "run_id": run_id,
        "node_id": node_id,
        "task_id": task_id,
        "action": action,
        "phase": phase,
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    if stopped is not None:
        payload["stopped"] = stopped
    if warnings is not None:
        payload["warnings"] = warnings
    if error is not None:
        payload["error"] = error

    async with sessionmaker() as session:
        session.add(
            Event(
                event_type="task.agent_action",
                data=payload,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def _emit_task_merge_event(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    payload: dict[str, object],
) -> None:
    async with sessionmaker() as session:
        session.add(
            Event(
                event_type="task.merge",
                data=payload,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def _emit_merge_run_event(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    node_id: int,
    epic_id: int,
    requested_task_id: int,
    status: MergeRunStatus,
    blocked_step_index: int | None = None,
    blocked_step_kind: str | None = None,
    blocked_branch_name: str | None = None,
) -> None:
    data: dict[str, object] = {
        "run_id": run_id,
        "node_id": node_id,
        "epic_id": epic_id,
        "requested_task_id": requested_task_id,
        "status": status,
    }
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


def _merge_run_plan_snapshot(*, plan) -> dict[str, object]:
    steps = [
        MergeRunPlanStepData(
            index=index,
            kind=step.kind,
            node_id=step.node_id,
            task_id=step.task_id,
            branch_name=step.branch_name,
            worktree_path=str(step.worktree_path),
            upstream_ref=step.upstream_ref,
            base_branch=step.base_branch,
        )
        for index, step in enumerate(plan.steps)
    ]
    payload = MergeRunPlanData(
        base_branch=plan.base_branch,
        base_worktree=str(plan.base_worktree),
        scope=plan.scope,
        restack_mode=plan.restack_mode,
        spine_node_ids=list(plan.spine_node_ids),
        affected_node_ids=list(plan.affected_node_ids),
        steps=steps,
    )
    return payload.model_dump(mode="python")


async def _upsert_merge_run(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    epic_id: int,
    task_id: int,
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
            row.status = MergeRunStatus.Running
            row.scope = scope
            row.allow_running = allow_running
            row.force = force
            row.plan = plan_snapshot
            row.current_step_index = None
            row.blocked_step_index = None
            row.blocked_step_kind = None
            row.blocked_node_id = None
            row.blocked_task_id = None
            row.blocked_branch_name = None
            row.blocked_worktree_path = None
            row.blocked_error = None
        await session.commit()


async def _record_merge_run_step_update(
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
            row.blocked_node_id = update.step.node_id
            row.blocked_task_id = update.step.task_id
            row.blocked_branch_name = update.step.branch_name
            row.blocked_worktree_path = str(update.step.worktree_path)
            row.blocked_error = update.error
            if update.step.node_id is not None:
                session.add(
                    Event(
                        event_type="merge.run",
                        data={
                            "run_id": run_id,
                            "node_id": update.step.node_id,
                            "epic_id": row.epic_id,
                            "requested_task_id": row.requested_task_id,
                            "status": row.status,
                            "blocked_step_index": row.blocked_step_index,
                            "blocked_step_kind": row.blocked_step_kind,
                            "blocked_branch_name": row.blocked_branch_name,
                        },
                        created_at=now,
                    )
                )
        await session.commit()


async def _set_merge_run_status(
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

        node_id: int | None = row.blocked_node_id
        blocked_step_index = row.blocked_step_index
        blocked_step_kind = row.blocked_step_kind
        blocked_branch_name = row.blocked_branch_name
        if node_id is None:
            try:
                plan = MergeRunPlanData.model_validate(row.plan)
                if plan.spine_node_ids:
                    node_id = plan.spine_node_ids[-1]
                elif plan.steps:
                    node_id = plan.steps[0].node_id
            except Exception:
                node_id = None

        row.status = status
        if status in {MergeRunStatus.Succeeded, MergeRunStatus.Canceled}:
            row.current_step_index = None
            row.blocked_step_index = None
            row.blocked_step_kind = None
            row.blocked_node_id = None
            row.blocked_task_id = None
            row.blocked_branch_name = None
            row.blocked_worktree_path = None
            row.blocked_error = None
        if error is not None:
            row.blocked_error = error

        if node_id is not None:
            data: dict[str, object] = {
                "run_id": run_id,
                "node_id": node_id,
                "epic_id": row.epic_id,
                "requested_task_id": row.requested_task_id,
                "status": status,
            }
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


async def _task_node_info(
    sessionmaker: async_sessionmaker[AsyncSession], *, task_id: int
) -> tuple[Task, Node, int | None]:
    async with sessionmaker() as session:
        task, node = await _require_task_node(session, task_id=task_id)
        return task, node, node.agent_id


@overload
async def _perform_task_agent_action(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    task_id: int,
    action: Literal["stop"],
    harness: str | None,
    detach: bool,
    prelude: str | None,
) -> tuple[Task, Node, bool, list[str]]: ...


@overload
async def _perform_task_agent_action(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    task_id: int,
    action: Literal["start", "restart"],
    harness: str | None,
    detach: bool,
    prelude: str | None,
) -> tuple[Task, Node, StartAgentResult, list[str]]: ...


async def _perform_task_agent_action(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    task_id: int,
    action: Literal["start", "restart", "stop"],
    harness: str | None,
    detach: bool,
    prelude: str | None,
) -> tuple[Task, Node, StartAgentResult | bool, list[str]]:
    """Execute an agent action and emit WS-visible progress events.

    Returns (task, node, result, warnings).

    - For start/restart, result is a StartAgentResult.
    - For stop, result is a bool (stopped).
    """

    task, node, existing_agent_id = await _task_node_info(sessionmaker, task_id=task_id)
    await _emit_task_agent_action_event(
        sessionmaker=sessionmaker,
        run_id=run_id,
        node_id=node.id,
        task_id=task.id,
        action=action,
        phase="requested",
    )

    try:
        if action == "start":
            result = await app.state.runner_backend.start_task_agent(
                task_id=task.id,
                harness_command=harness or "",
                detach=detach,
                prelude_override=prelude,
            )
            warnings = list(result.warnings)
            await _emit_task_agent_action_event(
                sessionmaker=sessionmaker,
                run_id=run_id,
                node_id=node.id,
                task_id=task.id,
                action="start",
                phase="started",
                agent_id=result.agent.id,
                warnings=warnings,
            )
            return task, node, result, warnings

        if action == "restart":
            result = await app.state.runner_backend.restart_task_agent(
                task_id=task.id,
                harness_command=harness,
                detach=detach,
                prelude_override=prelude,
            )
            warnings = list(result.warnings)
            await _emit_task_agent_action_event(
                sessionmaker=sessionmaker,
                run_id=run_id,
                node_id=node.id,
                task_id=task.id,
                action="restart",
                phase="started",
                agent_id=result.agent.id,
                warnings=warnings,
            )
            return task, node, result, warnings

        if action == "stop":
            stopped = await app.state.runner_backend.stop_task_agent(task_id=task.id)
            await _emit_task_agent_action_event(
                sessionmaker=sessionmaker,
                run_id=run_id,
                node_id=node.id,
                task_id=task.id,
                action="stop",
                phase="stopped",
                agent_id=existing_agent_id,
                stopped=stopped,
            )
            return task, node, stopped, []

        raise ValueError(f"Unknown action: {action!r}")
    except Exception as e:
        await _emit_task_agent_action_event(
            sessionmaker=sessionmaker,
            run_id=run_id,
            node_id=node.id,
            task_id=task.id,
            action=action,
            phase="failed",
            agent_id=existing_agent_id,
            error=str(e),
        )
        raise


async def _run_task_agents_bulk(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    start_task_ids: list[int],
    restart_task_ids: list[int],
    harness: str | None,
    detach: bool,
    prelude: str | None,
) -> None:
    async def _node_id_for_task(task_id: int) -> int:
        async with sessionmaker() as session:
            _, node = await _require_task_node(session, task_id=task_id)
            return node.id

    for task_id in sorted(set(start_task_ids)):
        node_id = await _node_id_for_task(task_id)
        await _emit_task_agent_run_event(
            sessionmaker=sessionmaker,
            run_id=run_id,
            node_id=node_id,
            task_id=task_id,
            action="start",
            phase="requested",
        )
        try:
            result = await app.state.runner_backend.start_task_agent(
                task_id=task_id,
                harness_command=harness or "",
                detach=detach,
                prelude_override=prelude,
            )
            await _emit_task_agent_run_event(
                sessionmaker=sessionmaker,
                run_id=run_id,
                node_id=node_id,
                task_id=task_id,
                action="start",
                phase="started",
                agent_id=result.agent.id,
                warnings=list(result.warnings),
            )
        except Exception as e:
            await _emit_task_agent_run_event(
                sessionmaker=sessionmaker,
                run_id=run_id,
                node_id=node_id,
                task_id=task_id,
                action="start",
                phase="failed",
                error=str(e),
            )

    for task_id in sorted(set(restart_task_ids)):
        node_id = await _node_id_for_task(task_id)
        await _emit_task_agent_run_event(
            sessionmaker=sessionmaker,
            run_id=run_id,
            node_id=node_id,
            task_id=task_id,
            action="restart",
            phase="requested",
        )
        try:
            result = await app.state.runner_backend.restart_task_agent(
                task_id=task_id,
                harness_command=None,
                detach=detach,
                prelude_override=prelude,
            )
            await _emit_task_agent_run_event(
                sessionmaker=sessionmaker,
                run_id=run_id,
                node_id=node_id,
                task_id=task_id,
                action="restart",
                phase="started",
                agent_id=result.agent.id,
                warnings=list(result.warnings),
            )
        except Exception as e:
            await _emit_task_agent_run_event(
                sessionmaker=sessionmaker,
                run_id=run_id,
                node_id=node_id,
                task_id=task_id,
                action="restart",
                phase="failed",
                error=str(e),
            )


async def _run_task_agents_bulk_actions(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    actions: list[TaskAgentBulkActionItemRequest],
    harness: str | None,
    detach: bool,
    prelude: str | None,
) -> None:
    for item in sorted(actions, key=lambda a: (a.task_id, a.action)):
        try:
            await _perform_task_agent_action(
                sessionmaker=sessionmaker,
                run_id=run_id,
                task_id=item.task_id,
                action=item.action,
                harness=harness,
                detach=detach,
                prelude=prelude,
            )
        except Exception:
            continue


@v1.post("/tasks/agent/actions", response_model=TaskAgentBulkActionResponse)
async def bulk_task_agent_actions(
    request: TaskAgentBulkActionRequest,
) -> TaskAgentBulkActionResponse:
    actions = request.actions
    requires_harness = any(a.action == "start" for a in actions)
    if requires_harness and not request.harness:
        raise HTTPException(
            status_code=400,
            detail="harness is required when the action list includes 'start'",
        )
    run_id = request.run_id or uuid4().hex
    submitted = len(actions)
    asyncio.create_task(
        _run_task_agents_bulk_actions(
            sessionmaker=app.state.sessionmaker,
            run_id=run_id,
            actions=actions,
            harness=request.harness,
            detach=request.detach,
            prelude=request.prelude,
        )
    )
    return TaskAgentBulkActionResponse(run_id=run_id, submitted=submitted)


@v1.post("/tasks/agent/run", response_model=TaskAgentBulkRunResponse)
async def run_task_agents_bulk(
    request: TaskAgentBulkRunRequest,
) -> TaskAgentBulkRunResponse:
    if request.start_task_ids and not request.harness:
        raise HTTPException(
            status_code=400,
            detail="harness is required when start_task_ids is non-empty",
        )
    run_id = request.run_id or uuid4().hex
    submitted = len(set(request.start_task_ids)) + len(set(request.restart_task_ids))
    asyncio.create_task(
        _run_task_agents_bulk(
            sessionmaker=app.state.sessionmaker,
            run_id=run_id,
            start_task_ids=request.start_task_ids,
            restart_task_ids=request.restart_task_ids,
            harness=request.harness,
            detach=request.detach,
            prelude=request.prelude,
        )
    )
    return TaskAgentBulkRunResponse(run_id=run_id, submitted=submitted)


@v1.post("/tasks/{task_id}/agent/stop", response_model=TaskAgentStopResponse)
async def stop_task_agent(task_id: int) -> TaskAgentStopResponse:
    run_id = uuid4().hex
    try:
        _, node, stopped, _ = await _perform_task_agent_action(
            sessionmaker=app.state.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            action="stop",
            harness=None,
            detach=True,
            prelude=None,
        )
    except RunnerBackendError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        agent = await session.get(Agent, node.agent_id) if node.agent_id else None

    return TaskAgentStopResponse(
        task_id=task_id,
        node_id=node.id,
        agent_id=None if agent is None else agent.id,
        agent_name=None if agent is None else agent.display_name,
        agent_status=None if agent is None else agent.status,
        stopped=stopped,
    )


@v1.post("/tasks/{task_id}/agent/restart", response_model=TaskAgentStartResponse)
async def restart_task_agent(
    task_id: int,
    request: TaskAgentRestartRequest,
) -> TaskAgentStartResponse:
    run_id = uuid4().hex
    try:
        _, node, result, warnings = await _perform_task_agent_action(
            sessionmaker=app.state.sessionmaker,
            run_id=run_id,
            task_id=task_id,
            action="restart",
            harness=request.harness,
            detach=request.detach,
            prelude=request.prelude,
        )
    except RunnerBackendError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    row = result.agent  # type: ignore[attr-defined]

    return TaskAgentStartResponse(
        task_id=task_id,
        node_id=node.id,
        agent_id=row.id,
        agent_name=row.display_name,
        agent_status=row.status,
        harness_profile_id=row.harness_profile_id or "",
        attach=TypeAdapter(AttachInfoResponse).validate_python(row.attach),
        resolved_profile=TypeAdapter(
            HarnessProfileDefinitionResponse | None
        ).validate_python(row.resolved_profile),
        started_at=row.started_at or datetime.now(UTC),
        started=result.started,  # type: ignore[attr-defined]
        warnings=warnings,
    )


@v1.post("/tasks/{task_id}/merge-ready", response_model=TaskResponse)
async def set_task_merge_ready(
    task_id: int,
    request: TaskMergeReadyRequest,
) -> TaskResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        task = await session.get(Task, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        task.merge_ready_at = datetime.now(UTC) if request.ready else None
        await session.commit()
        await session.refresh(task)
        return TaskResponse.model_validate(task, from_attributes=True)


@v1.post("/tasks/{task_id}/merge", response_model=TaskMergeResponse)
async def merge_task(task_id: int, request: TaskMergeRequest) -> TaskMergeResponse:
    run_id = request.run_id or uuid4().hex
    scope: Literal["descendants", "spine"] = (
        request.scope if request.cascade else "spine"
    )
    restack_mode: Literal["strict", "merge_then_restack"] = request.restack_mode
    try:
        plan = await build_merge_cascade_plan(
            ctx=app.state.ctx,
            sessionmaker=app.state.sessionmaker,
            task_id=task_id,
            run_id=run_id,
            scope=scope,
            restack_mode=restack_mode,
            force=request.force,
        )
    except MergePlanError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if plan.running_agents and not request.allow_running:
        raise HTTPException(
            status_code=409,
            detail="Merge affects running tasks; retry with allowRunning=true once you confirm.",
        )

    if request.dry_run:
        steps = [
            TaskMergePlanStepResponse(
                kind=step.kind,
                node_id=step.node_id,
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

    plan_snapshot = _merge_run_plan_snapshot(plan=plan)
    await _upsert_merge_run(
        sessionmaker=app.state.sessionmaker,
        run_id=run_id,
        epic_id=plan.epic_id,
        task_id=task_id,
        scope=scope,
        allow_running=request.allow_running,
        force=request.force,
        plan_snapshot=plan_snapshot,
    )
    merge_node_id = (
        plan.spine_node_ids[-1]
        if plan.spine_node_ids
        else next((s.node_id for s in plan.steps if s.node_id is not None), None)
    )
    if merge_node_id is not None:
        await _emit_merge_run_event(
            sessionmaker=app.state.sessionmaker,
            run_id=run_id,
            node_id=merge_node_id,
            epic_id=plan.epic_id,
            requested_task_id=task_id,
            status=MergeRunStatus.Running,
        )

    async def _run() -> None:
        try:
            await execute_merge_cascade_plan(
                ctx=app.state.ctx,
                sessionmaker=app.state.sessionmaker,
                plan=plan,
                allow_running=request.allow_running,
                emit_event=lambda payload: _emit_task_merge_event(
                    sessionmaker=app.state.sessionmaker, payload=payload
                ),
                update_run=lambda update: _record_merge_run_step_update(
                    sessionmaker=app.state.sessionmaker,
                    run_id=run_id,
                    update=update,
                ),
            )
            await _set_merge_run_status(
                sessionmaker=app.state.sessionmaker,
                run_id=run_id,
                status=MergeRunStatus.Succeeded,
            )
        except MergeBlockedByRunningAgents:
            # The plan was computed with running agents, but the request didn't allow them.
            # We return 409 to the caller before spawning, so this should not happen.
            return
        except Exception as e:
            if not isinstance(e, GitCommandError):
                await _set_merge_run_status(
                    sessionmaker=app.state.sessionmaker,
                    run_id=run_id,
                    status=MergeRunStatus.Failed,
                    error=str(e),
                )
            await _emit_task_merge_event(
                sessionmaker=app.state.sessionmaker,
                payload={
                    "run_id": run_id,
                    "node_id": None,
                    "task_id": task_id,
                    "kind": "merge_ff",
                    "phase": "failed",
                    "branch_name": plan.base_branch,
                    "error": str(e),
                },
            )

    asyncio.create_task(_run())

    return TaskMergeResponse(run_id=run_id, dry_run=False, base_branch=plan.base_branch)


@v1.post("/merge-runs/{run_id}/resume", response_model=MergeRunResumeResponse)
async def resume_merge_run(
    run_id: str, request: MergeRunResumeRequest
) -> MergeRunResumeResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        run = await session.scalar(select(MergeRun).where(MergeRun.run_id == run_id))
        if run is None:
            raise HTTPException(status_code=404, detail="Merge run not found")

        if run.status == MergeRunStatus.Running:
            raise HTTPException(status_code=409, detail="Merge run is already running")
        if run.status == MergeRunStatus.Blocked:
            raise HTTPException(
                status_code=409,
                detail="Merge run is blocked; resolve the in-progress git operation first.",
            )
        if run.status != MergeRunStatus.Resumable:
            raise HTTPException(status_code=400, detail="Merge run is not resumable")

        task_id = run.requested_task_id
        scope_value = run.scope
        force = run.force
        allow_running = run.allow_running or request.allow_running
        restack_mode: Literal["strict", "merge_then_restack"] = run.restack_mode

        blocked_step_index = run.blocked_step_index
        blocked_step_kind = run.blocked_step_kind
        blocked_branch_name = run.blocked_branch_name

    normalized_scope = scope_value.strip().lower()
    if normalized_scope not in {"descendants", "spine"}:
        raise HTTPException(
            status_code=400, detail=f"Invalid merge run scope: {scope_value!r}"
        )

    try:
        plan = await build_merge_cascade_plan(
            ctx=app.state.ctx,
            sessionmaker=app.state.sessionmaker,
            task_id=task_id,
            run_id=run_id,
            scope=normalized_scope,  # type: ignore[arg-type]
            restack_mode=restack_mode,
            force=force,
        )
    except MergePlanError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if plan.running_agents and not allow_running:
        raise HTTPException(
            status_code=409,
            detail="Merge affects running tasks; retry with allowRunning=true once you confirm.",
        )

    start_at_step_index = blocked_step_index or 0
    if start_at_step_index >= len(plan.steps):
        start_at_step_index = 0
    if start_at_step_index and blocked_step_kind and blocked_branch_name:
        step = plan.steps[start_at_step_index]
        if step.kind != blocked_step_kind or step.branch_name != blocked_branch_name:
            start_at_step_index = 0

    plan_snapshot = _merge_run_plan_snapshot(plan=plan)
    await _upsert_merge_run(
        sessionmaker=app.state.sessionmaker,
        run_id=run_id,
        epic_id=plan.epic_id,
        task_id=task_id,
        scope=normalized_scope,
        allow_running=allow_running,
        force=force,
        plan_snapshot=plan_snapshot,
    )
    merge_node_id = (
        plan.spine_node_ids[-1]
        if plan.spine_node_ids
        else next((s.node_id for s in plan.steps if s.node_id is not None), None)
    )
    if merge_node_id is not None:
        await _emit_merge_run_event(
            sessionmaker=app.state.sessionmaker,
            run_id=run_id,
            node_id=merge_node_id,
            epic_id=plan.epic_id,
            requested_task_id=task_id,
            status=MergeRunStatus.Running,
        )

    async def _run() -> None:
        try:
            await execute_merge_cascade_plan(
                ctx=app.state.ctx,
                sessionmaker=app.state.sessionmaker,
                plan=plan,
                allow_running=allow_running,
                emit_event=lambda payload: _emit_task_merge_event(
                    sessionmaker=app.state.sessionmaker, payload=payload
                ),
                update_run=lambda update: _record_merge_run_step_update(
                    sessionmaker=app.state.sessionmaker,
                    run_id=run_id,
                    update=update,
                ),
                start_at_step_index=start_at_step_index,
            )
            await _set_merge_run_status(
                sessionmaker=app.state.sessionmaker,
                run_id=run_id,
                status=MergeRunStatus.Succeeded,
            )
        except MergeBlockedByRunningAgents:
            return
        except Exception as e:
            if not isinstance(e, GitCommandError):
                await _set_merge_run_status(
                    sessionmaker=app.state.sessionmaker,
                    run_id=run_id,
                    status=MergeRunStatus.Failed,
                    error=str(e),
                )
            await _emit_task_merge_event(
                sessionmaker=app.state.sessionmaker,
                payload={
                    "run_id": run_id,
                    "node_id": None,
                    "task_id": task_id,
                    "kind": "merge_ff",
                    "phase": "failed",
                    "branch_name": plan.base_branch,
                    "error": str(e),
                },
            )

    asyncio.create_task(_run())

    return MergeRunResumeResponse(run_id=run_id, base_branch=plan.base_branch)


@v1.get("/tasks/{task_id}/agent/logs", include_in_schema=False)
async def task_agent_logs(
    task_id: int,
    lines: int = 200,
    max_bytes: int = 65536,
) -> dict[str, object]:
    """Return a tail of the agent log for a task (best-effort)."""
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        _, node = await _require_task_node(session, task_id=task_id)
        agent = await session.get(Agent, node.agent_id) if node.agent_id else None

    if agent is None:
        raise HTTPException(status_code=404, detail="No agent for this task")

    log_path = agent_log_path_for_row(app.state.ctx, agent_row=agent)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No log file found")

    text, truncated = _tail_text(log_path, max_bytes=max_bytes)
    if lines > 0:
        text = "\n".join(text.splitlines()[-lines:])

    return {
        "path": str(log_path),
        "text": text,
        "truncated": truncated,
    }


@v1.post("/nodes/{node_id}/agent", response_model=NodeResponse)
async def set_node_agent(node_id: int, request: NodeSetAgentRequest) -> NodeResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        node = await session.get(Node, node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")

        previous_agent_id = node.agent_id
        if request.agent_id is not None:
            agent = await session.get(Agent, request.agent_id)
            if agent is None:
                raise HTTPException(status_code=400, detail="Unknown agent id")
            node.agent_id = agent.id
        else:
            node.agent_id = None

        if node.agent_id != previous_agent_id:
            session.add(
                Event(
                    event_type="node.agent_set",
                    data={
                        "node_id": node.id,
                        "agent_id": node.agent_id,
                        "previous_agent_id": previous_agent_id,
                    },
                    created_at=datetime.now(UTC),
                )
            )

        await session.commit()
        await session.refresh(node)
        return NodeResponse.model_validate(node, from_attributes=True)


@v1.get("/linear/status", response_model=LinearStatusResponse)
async def linear_status() -> LinearStatusResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        auth = await session.scalar(
            select(LinearAuth).order_by(desc(LinearAuth.id)).limit(1)
        )
    return LinearStatusResponse(
        connected=auth is not None,
        connected_at=auth.created_at if auth is not None else None,
    )


@v1.get("/linear/oauth/start", include_in_schema=False)
async def linear_oauth_start() -> Response:
    settings = load_settings(repo_root=app.state.ctx.repo_root)
    state = new_oauth_state()
    app.state.linear_oauth_states[state] = datetime.now(UTC)

    for key, created in list(app.state.linear_oauth_states.items()):
        if datetime.now(UTC) - created > timedelta(minutes=15):
            app.state.linear_oauth_states.pop(key, None)

    try:
        url = linear_authorize_url(settings, state=state)
    except ValueError as e:
        return HTMLResponse(
            "<h1>Linear is not configured</h1>"
            "<p>Set <code>REDESMYN_LINEAR_CLIENT_ID</code> and <code>REDESMYN_LINEAR_CLIENT_SECRET</code> "
            "in <code>.env</code> (see <code>.env.example</code>).</p>"
            f"<pre>{e}</pre>",
            status_code=500,
        )

    return RedirectResponse(url=url, status_code=302)


@v1.get("/linear/oauth/callback", include_in_schema=False)
async def linear_oauth_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> Response:
    if error:
        return HTMLResponse(
            f"<h1>Linear auth failed</h1><p>{error}</p><p>{error_description or ''}</p>",
            status_code=400,
        )

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code/state")

    if state not in app.state.linear_oauth_states:
        raise HTTPException(status_code=400, detail="Unknown or expired state")
    app.state.linear_oauth_states.pop(state, None)

    settings = load_settings(repo_root=app.state.ctx.repo_root)
    redirect_uri = linear_redirect_uri(settings)
    try:
        token = await exchange_code_for_token(
            settings, code=code, redirect_uri=redirect_uri
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        auth = await session.scalar(
            select(LinearAuth).order_by(desc(LinearAuth.id)).limit(1)
        )
        if auth is None:
            auth = LinearAuth(access_token=token.access_token)
            session.add(auth)

        auth.access_token = token.access_token
        auth.refresh_token = token.refresh_token
        auth.token_type = token.token_type
        auth.scope = token.scope
        auth.expires_at = token.expires_at
        await session.commit()

    return HTMLResponse(
        "<h1>Linear connected</h1><p>You can close this tab and return to Redesmyn.</p>"
    )


@app.get("/", include_in_schema=False)
async def index() -> Response:
    ctx = app.state.ctx
    index_html = ctx.worktree_root / "dashboard" / "dist" / "index.html"
    if index_html.is_file():
        return FileResponse(str(index_html))

    return HTMLResponse(
        "<h1>Redesmyn</h1><p>Dashboard not built yet. Build with `cd dashboard && npm run build`.</p>"
    )


def _dist_path(repo_root: Path) -> Path:
    return repo_root / "dashboard" / "dist"


@v1.get("/config", response_model=OrchestrationDefaultsResponse)
async def get_orchestration_config() -> OrchestrationDefaultsResponse:
    defaults = load_orchestration_defaults(app.state.ctx)
    return OrchestrationDefaultsResponse(
        default_epic=defaults.default_epic,
        fleet=OrchestrationFleetDefaultsResponse(
            mode=defaults.fleet.mode,
            size=defaults.fleet.size,
        ),
        harness=OrchestrationHarnessDefaultsResponse(
            command=defaults.harness.command,
            detach=defaults.harness.detach,
            prelude=defaults.harness.prelude,
            built_in_prelude_template=DEFAULT_AGENT_PRELUDE_TEMPLATE,
            send_prelude=defaults.harness.send_prelude,
            submit_prelude=defaults.harness.submit_prelude,
        ),
        sandbox=OrchestrationSandboxDefaultsResponse(
            type=defaults.sandbox.type,
            network=defaults.sandbox.network,
        ),
    )


@v1.post("/config", response_model=OrchestrationDefaultsResponse)
async def update_orchestration_config(
    request: OrchestrationDefaultsUpdateRequest,
) -> OrchestrationDefaultsResponse:
    ctx = app.state.ctx
    path = repo_config_path(ctx)
    data = read_config_file(path)

    if "default_epic" in request.model_fields_set:
        set_config_value(data, "default_epic", request.default_epic)

    if request.fleet is not None:
        if "mode" in request.fleet.model_fields_set:
            set_config_value(data, "fleet.mode", request.fleet.mode)
        if "size" in request.fleet.model_fields_set:
            set_config_value(data, "fleet.size", request.fleet.size)

    if request.harness is not None:
        if "command" in request.harness.model_fields_set:
            set_config_value(data, "harness.command", request.harness.command)
        if "detach" in request.harness.model_fields_set:
            set_config_value(data, "harness.detach", request.harness.detach)
        if "prelude" in request.harness.model_fields_set:
            set_config_value(data, "harness.prelude", request.harness.prelude)
        if "send_prelude" in request.harness.model_fields_set:
            set_config_value(data, "harness.send_prelude", request.harness.send_prelude)
        if "submit_prelude" in request.harness.model_fields_set:
            set_config_value(
                data, "harness.submit_prelude", request.harness.submit_prelude
            )

    if request.sandbox is not None:
        if "type" in request.sandbox.model_fields_set:
            set_config_value(data, "sandbox.type", request.sandbox.type)
        if "network" in request.sandbox.model_fields_set:
            set_config_value(data, "sandbox.network", request.sandbox.network)

    write_config(path, data)
    return await get_orchestration_config()


@v1.get("/sandbox/capabilities", response_model=SandboxCapabilitiesResponse)
async def sandbox_capabilities() -> SandboxCapabilitiesResponse:
    capabilities = make_sandbox_provider().capabilities()
    return SandboxCapabilitiesResponse.model_validate(
        capabilities.model_dump(mode="python"),
    )


@v1.get("/status", response_model=ApiStatusResponse)
async def api_status() -> ApiStatusResponse:
    ctx = app.state.ctx
    sessionmaker = app.state.sessionmaker

    async with sessionmaker() as session:
        repo = await session.scalar(
            select(Repository).where(Repository.repo_root == str(ctx.repo_root))
        )
        block = await session.scalar(
            select(Block)
            .where(
                Block.scope == BlockScope.for_repo(),
                Block.policy == BlockPolicy.GitMutations,
                Block.cleared_at.is_(None),
            )
            .order_by(desc(Block.id))
            .limit(1)
        )

    return ApiStatusResponse(
        repo_root=str(ctx.repo_root),
        db_path=str(ctx.db_path),
        default_branch=repo.default_branch if repo else None,
        block=None
        if block is None
        else BlockStatusResponse(
            mode=block.mode,
            scope=BlockScopeResponse.model_validate(block.scope, from_attributes=True),
            reason=block.reason,
            policy=block.policy,
            release=TypeAdapter(ReleaseConditionResponse).validate_python(
                block.release
            ),
        ),
    )


app.include_router(v1)
