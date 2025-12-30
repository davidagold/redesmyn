from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, overload
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, TypeAdapter
from sqlalchemy import delete, desc, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.responses import RedirectResponse
from starlette.websockets import WebSocketDisconnect

from redesmyn.agent_prelude import DEFAULT_AGENT_PRELUDE_TEMPLATE
from redesmyn.agent_monitor import run_agent_monitor
from redesmyn.agent_runtime import StartAgentResult, agent_log_path_for_session_row
from redesmyn.context import RepoContext, build_repo_context
from redesmyn.db import (
    Agent,
    AgentSession,
    Block,
    BlockScope,
    DaemonConnection,
    DaemonCommand,
    Epic,
    Event,
    GitTrunkTimelineByInstance,
    HarnessProfile,
    Host,
    LinearAuth,
    MergeRun,
    Repository,
    Task,
    create_engine,
    create_sessionmaker,
    init_db,
)
from redesmyn.db.models import HostCapabilities
from redesmyn.domain.enums import BlockPolicy, CommandState, MergeRunStatus
from redesmyn.event_stream import run_event_stream
from redesmyn.integrations.linear import (
    exchange_code_for_token,
    linear_authorize_url,
    linear_redirect_uri,
    new_oauth_state,
    new_pkce_verifier,
    pkce_code_challenge,
)
from redesmyn.integrations.linear_credentials import (
    LinearCredentials,
    default_linear_credential_store,
)
from redesmyn.orchestrator import init_repo
from redesmyn.orchestration_config import (
    load_orchestration_defaults,
    read_config_file,
    repo_config_path,
    set_config_value,
    write_config,
)
from redesmyn.merge_runs import apply_merge_run_event_update
from redesmyn.repo_observer import run_repo_observer
from redesmyn.repo_executor import (
    RepoExecutor,
    RepoExecutorError,
    RepoExecutorTarget,
    make_repo_executor,
)
from redesmyn.runner_backend import (
    RunnerBackend,
    RunnerBackendError,
    make_runner_backend,
)
from redesmyn.sandbox import make_sandbox_provider
from redesmyn.schemas.core import (
    ApiStatusResponse,
    AgentSessionResponse,
    AttachInfoResponse,
    BlockScopeResponse,
    BlockStatusResponse,
    DaemonCommandResponse,
    DaemonPresenceResponse,
    EpicGraphResponse,
    EpicResponse,
    EventResponse,
    HarnessProfileDefinitionResponse,
    HarnessProfileResponse,
    HarnessProfileUpsertRequest,
    HostResponse,
    HostUpsertRequest,
    LinearStatusResponse,
    MergeRunResumeRequest,
    MergeRunResumeResponse,
    MergeRunSummaryResponse,
    OrchestrationDefaultsResponse,
    OrchestrationFleetDefaultsResponse,
    OrchestrationHarnessDefaultsResponse,
    OrchestrationSandboxDefaultsResponse,
    OrchestrationDefaultsUpdateRequest,
    RepoExecutorStatusResponse,
    RepoKeyResponse,
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
    TaskRestackRequest,
    TaskRestackResponse,
    TrunkTimelineResponse,
)
from redesmyn.settings import load_settings
from redesmyn.logging_config import configure_logging
from redesmyn.host_identity import load_or_create_host_identity
from redesmyn.repo_executor_leases import (
    acquire_or_refresh_primary,
    expire_primary_if_owner,
    get_primary_host_key,
)
from redesmyn.repo_identity import RepoKey
from redesmyn.ws_protocol import DaemonInboundMessage, DaemonHello, ServerCommand
from redesmyn.ws_runtime import DaemonConnectionRegistry, JsonWebSocketHub

logger = logging.getLogger("redesmyn")


class AppState(Protocol):
    ctx: RepoContext
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    runner_backend: RunnerBackend
    repo_executor: RepoExecutor
    runner_mode: str
    local_host_key: str | None
    linear_oauth_states: dict[str, tuple[datetime, str]]
    event_hub: JsonWebSocketHub
    daemon_connections: DaemonConnectionRegistry


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
    env_repo_root_raw = os.environ.get("REDESMYN_REPO_ROOT")
    env_repo_root = Path(env_repo_root_raw) if env_repo_root_raw else None
    settings = load_settings(repo_root=env_repo_root)

    repo_root = settings.repo_root or env_repo_root or Path.cwd()
    worktree_root = settings.worktree_root or repo_root

    ctx = build_repo_context(
        repo_root=repo_root,
        worktree_root=worktree_root,
        state_dir_name=settings.state_dir_name,
        db_filename=settings.db_filename,
        db_path=settings.db_path,
    )
    ctx.db_path.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(state_dir=ctx.state_dir)

    if settings.runner_mode == "local":
        await init_repo(ctx, migrate=False)

    app.state.ctx = ctx
    app.state.engine = create_engine(ctx.db_path)
    await init_db(app.state.engine, migrate=False)
    app.state.sessionmaker = create_sessionmaker(app.state.engine)
    app.state.runner_backend = make_runner_backend(mode=settings.runner_mode, ctx=ctx)
    app.state.runner_mode = settings.runner_mode
    app.state.local_host_key = None
    app.state.linear_oauth_states = {}
    app.state.event_hub = JsonWebSocketHub()
    app.state.daemon_connections = DaemonConnectionRegistry()
    maybe_mount_dashboard(app, ctx.worktree_root)

    lease_refresh_task: asyncio.Task[None] | None = None
    if settings.runner_mode == "local":
        identity = load_or_create_host_identity(ctx)
        app.state.local_host_key = identity.host_key
        async with app.state.sessionmaker() as session:
            host = await session.scalar(
                select(Host).where(Host.host_key == identity.host_key)
            )
            if host is None:
                host = Host(
                    host_key=identity.host_key,
                    display_name=identity.display_name,
                    capabilities=HostCapabilities().model_dump(mode="python"),
                    last_seen_at=datetime.now(UTC),
                )
                session.add(host)
            else:
                host.display_name = identity.display_name
                host.last_seen_at = datetime.now(UTC)

            repo = await session.scalar(
                select(Repository).where(Repository.repo_root == str(ctx.repo_root))
            )
            if repo is not None:
                repo_key = RepoKey(workspace_id=repo.workspace_id, repo_id=repo.repo_id)
                await acquire_or_refresh_primary(
                    session, repo_key, host_key=identity.host_key
                )
            await session.commit()

        async def _refresh_local_lease() -> None:
            while True:
                await asyncio.sleep(20)
                async with app.state.sessionmaker() as session:
                    repo = await session.scalar(
                        select(Repository).where(
                            Repository.repo_root == str(ctx.repo_root)
                        )
                    )
                    if repo is None:
                        continue
                    await acquire_or_refresh_primary(
                        session,
                        RepoKey(
                            workspace_id=repo.workspace_id,
                            repo_id=repo.repo_id,
                        ),
                        host_key=identity.host_key,
                    )
                    await session.commit()

        lease_refresh_task = asyncio.create_task(_refresh_local_lease())

    app.state.repo_executor = make_repo_executor(
        runner_mode=settings.runner_mode,
        local_host_key=app.state.local_host_key,
        ctx=ctx,
        sessionmaker=app.state.sessionmaker,
        daemon_connections=app.state.daemon_connections,
    )

    observer_task: asyncio.Task[None] | None = None
    agent_monitor_task: asyncio.Task[None] | None = None
    if settings.runner_mode == "local" and os.environ.get(
        "REDESMYN_NO_OBSERVER"
    ) not in {"1", "true", "TRUE"}:
        observer_task = asyncio.create_task(
            run_repo_observer(
                ctx,
                interval_s=1.0,
                emit_baseline=False,
                once=False,
            )
        )

    if settings.runner_mode == "local" and os.environ.get(
        "REDESMYN_NO_AGENT_MONITOR"
    ) not in {"1", "true", "TRUE"}:
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

    if lease_refresh_task is not None:
        lease_refresh_task.cancel()
        try:
            await lease_refresh_task
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


@app.middleware("http")
async def log_unhandled_errors(request: Request, call_next):
    request_id = uuid4().hex
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request error request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error (request_id={request_id})"},
        )

    response.headers["x-request-id"] = request_id
    duration_ms = int((time.monotonic() - start) * 1000)
    if response.status_code >= 500:
        logger.error(
            "HTTP %s request_id=%s method=%s path=%s duration_ms=%s",
            response.status_code,
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
    return response


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


async def _append_event(
    session: AsyncSession, *, event_type: str, data: dict[str, Any]
) -> EventResponse:
    event = Event(event_type=event_type, data=data)
    session.add(event)
    await session.commit()
    await session.refresh(event)
    return EventResponse.model_validate(event, from_attributes=True)


async def _broadcast_event(event: EventResponse) -> None:
    await app.state.event_hub.publish(
        {"type": "event", "event": event.model_dump(by_alias=True)}
    )


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
    now = datetime.now(UTC)
    async with sessionmaker() as session:
        epic_row = await _resolve_epic_row(session, epic=epic)
        repo_row = await session.get(Repository, epic_row.repository_id)
        repo_key: RepoKey | None = None
        primary_host_key: str | None = None
        if repo_row is not None:
            repo_key = RepoKey(
                workspace_id=repo_row.workspace_id,
                repo_id=repo_row.repo_id,
            )
            primary_host_key = await get_primary_host_key(session, repo_key, now=now)

        tasks = list(
            await session.scalars(
                select(Task).where(Task.epic_id == epic_row.id).order_by(Task.id)
            )
        )
        task_ids = [t.id for t in tasks]
        latest_sessions: list[AgentSessionResponse] = []
        if task_ids:
            rows = list(
                await session.execute(
                    select(AgentSession, Agent)
                    .join(Agent, AgentSession.agent_id == Agent.id)
                    .where(AgentSession.task_id.in_(task_ids))
                    .order_by(desc(AgentSession.id))
                )
            )
            seen_task_ids: set[int] = set()
            for session_row, agent_row in rows:
                if session_row.task_id is None:
                    continue
                if session_row.task_id in seen_task_ids:
                    continue
                seen_task_ids.add(session_row.task_id)
                latest_sessions.append(
                    AgentSessionResponse(
                        id=session_row.id,
                        agent_id=agent_row.id,
                        agent_name=agent_row.display_name,
                        task_id=session_row.task_id,
                        status=session_row.status,
                        harness_profile_id=session_row.harness_profile_id,
                        resolved_profile=TypeAdapter(
                            HarnessProfileDefinitionResponse | None
                        ).validate_python(session_row.resolved_profile),
                        started_at=session_row.started_at,
                        ended_at=session_row.ended_at,
                    )
                )
        merge_runs = list(
            await session.scalars(
                select(MergeRun)
                .where(MergeRun.epic_id == epic_row.id)
                .where(MergeRun.canonical.is_(True))
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
        trunk_row = (
            await session.get(
                GitTrunkTimelineByInstance,
                (epic_row.id, primary_host_key),
            )
            if primary_host_key is not None
            else None
        )

    trunk: TrunkTimelineResponse | None = None
    if trunk_row is not None and trunk_row.data:
        try:
            trunk = TrunkTimelineResponse.model_validate(trunk_row.data)
        except Exception:
            trunk = None

    task_responses = [
        TaskResponse.model_validate(task, from_attributes=True) for task in tasks
    ]

    repo_executor_status = (
        await app.state.repo_executor.get_status(
            repo=repo_row,
            primary_host_key=primary_host_key,
        )
        if repo_row is not None
        else None
    )

    repo_executor = (
        RepoExecutorStatusResponse(
            workspace_id=repo_row.workspace_id,
            repo_id=repo_row.repo_id,
            primary_host_key=repo_executor_status.primary_host_key,
            attached_host_keys=repo_executor_status.attached_host_keys,
        )
        if repo_row is not None and repo_executor_status is not None
        else None
    )

    return EpicGraphResponse(
        epic=EpicResponse.model_validate(epic_row, from_attributes=True),
        tasks=task_responses,
        agent_sessions=latest_sessions,
        merge_runs=[
            MergeRunSummaryResponse.model_validate(r, from_attributes=True)
            for r in merge_runs
        ],
        trunk=trunk,
        repo_executor=repo_executor,
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


async def _require_task(session: AsyncSession, *, task_id: int) -> Task:
    task = await session.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


async def _require_current_repo(session: AsyncSession) -> Repository:
    repo = await session.scalar(
        select(Repository).where(Repository.repo_root == str(app.state.ctx.repo_root))
    )
    if repo is None:
        raise HTTPException(status_code=500, detail="Repository not initialized")
    return repo


async def _resolve_repo_executor_target(
    session: AsyncSession,
    *,
    requested_host_key: str | None,
    operation: Literal["merge", "restack"],
) -> RepoExecutorTarget:
    repo = await _require_current_repo(session)
    repo_key = RepoKey(workspace_id=repo.workspace_id, repo_id=repo.repo_id)
    primary = await get_primary_host_key(session, repo_key)
    target_host_key = requested_host_key or primary
    if target_host_key is None:
        if requested_host_key is None:
            raise HTTPException(
                status_code=503,
                detail="No primary repo executor available. Start a daemon and attach this repo.",
            )
        raise HTTPException(status_code=503, detail="No repo executor available.")

    canonical = target_host_key == primary
    if requested_host_key is None and not canonical:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Primary executor is {primary or 'unknown'}; "
                f"repo-scoped {operation} requests cannot target a non-primary executor."
            ),
        )

    local_host_key = (
        app.state.local_host_key if app.state.runner_mode == "local" else None
    )
    is_local_executor = local_host_key is not None and target_host_key == local_host_key
    return RepoExecutorTarget(
        repo=repo,
        repo_key=repo_key,
        target_host_key=target_host_key,
        primary_host_key=primary,
        canonical=canonical,
        is_local=is_local_executor,
    )


@v1.post("/tasks/{task_id}/agent/start", response_model=TaskAgentStartResponse)
async def start_task_agent(
    task_id: int,
    request: TaskAgentStartRequest,
) -> TaskAgentStartResponse:
    run_id = uuid4().hex
    try:
        _, result, warnings = await _perform_task_agent_action(
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

    agent_row = result.agent
    session_row = result.agent_session

    return TaskAgentStartResponse(
        task_id=task_id,
        agent_id=agent_row.id,
        agent_session_id=session_row.id,
        agent_name=agent_row.display_name,
        agent_status=session_row.status,
        harness_profile_id=session_row.harness_profile_id or "",
        attach=TypeAdapter(AttachInfoResponse).validate_python(session_row.attach),
        resolved_profile=TypeAdapter(
            HarnessProfileDefinitionResponse | None
        ).validate_python(session_row.resolved_profile),
        started_at=session_row.started_at or datetime.now(UTC),
        started=result.started,
        warnings=warnings,
    )


async def _emit_task_agent_run_event(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    task_id: int,
    action: str,
    phase: str,
    agent_id: int | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "run_id": run_id,
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


async def _task_info(
    sessionmaker: async_sessionmaker[AsyncSession], *, task_id: int
) -> tuple[Task, int | None]:
    async with sessionmaker() as session:
        task = await _require_task(session, task_id=task_id)
        return task, task.agent_id


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
) -> tuple[Task, bool, list[str]]: ...


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
) -> tuple[Task, StartAgentResult, list[str]]: ...


async def _perform_task_agent_action(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    run_id: str,
    task_id: int,
    action: Literal["start", "restart", "stop"],
    harness: str | None,
    detach: bool,
    prelude: str | None,
) -> tuple[Task, StartAgentResult | bool, list[str]]:
    """Execute an agent action and emit WS-visible progress events.

    Returns (task, result, warnings).

    - For start/restart, result is a StartAgentResult.
    - For stop, result is a bool (stopped).
    """

    task, existing_agent_id = await _task_info(sessionmaker, task_id=task_id)
    await _emit_task_agent_action_event(
        sessionmaker=sessionmaker,
        run_id=run_id,
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
                task_id=task.id,
                action="start",
                phase="started",
                agent_id=result.agent.id,
                warnings=warnings,
            )
            return task, result, warnings

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
                task_id=task.id,
                action="restart",
                phase="started",
                agent_id=result.agent.id,
                warnings=warnings,
            )
            return task, result, warnings

        if action == "stop":
            stopped = await app.state.runner_backend.stop_task_agent(task_id=task.id)
            await _emit_task_agent_action_event(
                sessionmaker=sessionmaker,
                run_id=run_id,
                task_id=task.id,
                action="stop",
                phase="stopped",
                agent_id=existing_agent_id,
                stopped=stopped,
            )
            return task, stopped, []

        raise ValueError(f"Unknown action: {action!r}")
    except Exception as e:
        await _emit_task_agent_action_event(
            sessionmaker=sessionmaker,
            run_id=run_id,
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
    for task_id in sorted(set(start_task_ids)):
        await _emit_task_agent_run_event(
            sessionmaker=sessionmaker,
            run_id=run_id,
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
                task_id=task_id,
                action="start",
                phase="failed",
                error=str(e),
            )

    for task_id in sorted(set(restart_task_ids)):
        await _emit_task_agent_run_event(
            sessionmaker=sessionmaker,
            run_id=run_id,
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
        task, stopped, _ = await _perform_task_agent_action(
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
        agent = await session.get(Agent, task.agent_id) if task.agent_id else None
        latest_session: AgentSession | None = None
        if task.agent_id is not None:
            latest_session = await session.scalar(
                select(AgentSession)
                .where(AgentSession.task_id == task_id)
                .order_by(desc(AgentSession.id))
                .limit(1)
            )

    return TaskAgentStopResponse(
        task_id=task_id,
        agent_id=None if agent is None else agent.id,
        agent_name=None if agent is None else agent.display_name,
        agent_status=None if latest_session is None else latest_session.status,
        stopped=stopped,
    )


@v1.post("/tasks/{task_id}/agent/restart", response_model=TaskAgentStartResponse)
async def restart_task_agent(
    task_id: int,
    request: TaskAgentRestartRequest,
) -> TaskAgentStartResponse:
    run_id = uuid4().hex
    try:
        _, result, warnings = await _perform_task_agent_action(
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

    agent_row = result.agent
    session_row = result.agent_session

    return TaskAgentStartResponse(
        task_id=task_id,
        agent_id=agent_row.id,
        agent_session_id=session_row.id,
        agent_name=agent_row.display_name,
        agent_status=session_row.status,
        harness_profile_id=session_row.harness_profile_id or "",
        attach=TypeAdapter(AttachInfoResponse).validate_python(session_row.attach),
        resolved_profile=TypeAdapter(
            HarnessProfileDefinitionResponse | None
        ).validate_python(session_row.resolved_profile),
        started_at=session_row.started_at or datetime.now(UTC),
        started=result.started,
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
    resolved_request = request.model_copy(update={"run_id": run_id})

    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        target = await _resolve_repo_executor_target(
            session,
            requested_host_key=resolved_request.host_key,
            operation="merge",
        )

    try:
        return await app.state.repo_executor.merge(
            target=target,
            task_id=task_id,
            request=resolved_request,
        )
    except RepoExecutorError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e


@v1.post("/tasks/{task_id}/restack", response_model=TaskRestackResponse)
async def restack_task(
    task_id: int, request: TaskRestackRequest
) -> TaskRestackResponse:
    run_id = request.run_id or uuid4().hex
    resolved_request = request.model_copy(update={"run_id": run_id})

    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        target = await _resolve_repo_executor_target(
            session,
            requested_host_key=resolved_request.host_key,
            operation="restack",
        )

    try:
        return await app.state.repo_executor.restack(
            target=target,
            task_id=task_id,
            request=resolved_request,
        )
    except RepoExecutorError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e


@v1.post("/merge-runs/{run_id}/resume", response_model=MergeRunResumeResponse)
async def resume_merge_run(
    run_id: str, request: MergeRunResumeRequest
) -> MergeRunResumeResponse:
    sessionmaker = app.state.sessionmaker
    local_host_key = (
        app.state.local_host_key if app.state.runner_mode == "local" else None
    )
    target: RepoExecutorTarget
    run: MergeRun

    async with sessionmaker() as session:
        run_row = await session.scalar(
            select(MergeRun).where(MergeRun.run_id == run_id)
        )
        if run_row is None:
            raise HTTPException(status_code=404, detail="Merge run not found")
        run = run_row

        if run.status == MergeRunStatus.Running:
            raise HTTPException(status_code=409, detail="Merge run is already running")
        if run.status == MergeRunStatus.Blocked:
            raise HTTPException(
                status_code=409,
                detail="Merge run is blocked; resolve the in-progress git operation first.",
            )
        if run.status != MergeRunStatus.Resumable:
            raise HTTPException(status_code=400, detail="Merge run is not resumable")

        repo = await _require_current_repo(session)
        repo_key = RepoKey(workspace_id=repo.workspace_id, repo_id=repo.repo_id)

        primary_host_key = await get_primary_host_key(
            session,
            repo_key,
        )

        target_host_key = request.host_key or run.host_key or primary_host_key
        if target_host_key is None:
            raise HTTPException(
                status_code=503,
                detail="No repo executor available to resume this merge run.",
            )

        if run.canonical and target_host_key != primary_host_key:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Primary executor is {primary_host_key or 'none'}; "
                    f"cannot resume canonical merge run on {target_host_key}."
                ),
            )

        is_local_executor = (
            local_host_key is not None and target_host_key == local_host_key
        )
        target = RepoExecutorTarget(
            repo=repo,
            repo_key=repo_key,
            target_host_key=target_host_key,
            primary_host_key=primary_host_key,
            canonical=run.canonical,
            is_local=is_local_executor,
        )

    try:
        return await app.state.repo_executor.resume_merge_run(
            target=target,
            run=run,
            request=request,
        )
    except RepoExecutorError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e


@v1.get("/tasks/{task_id}/agent/logs", include_in_schema=False)
async def task_agent_logs(
    task_id: int,
    lines: int = 200,
    max_bytes: int = 65536,
) -> dict[str, object]:
    """Return a tail of the agent log for a task (best-effort)."""
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        task = await _require_task(session, task_id=task_id)
        agent = await session.get(Agent, task.agent_id) if task.agent_id else None
        agent_session: AgentSession | None = None
        if agent is not None:
            agent_session = await session.scalar(
                select(AgentSession)
                .where(AgentSession.task_id == task_id)
                .order_by(desc(AgentSession.id))
                .limit(1)
            )

    if agent_session is None:
        raise HTTPException(status_code=404, detail="No agent session for this task")

    log_path = agent_log_path_for_session_row(
        app.state.ctx, agent_session_row=agent_session
    )
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

@v1.get("/linear/status", response_model=LinearStatusResponse)
async def linear_status() -> LinearStatusResponse:
    store = default_linear_credential_store()
    creds = store.get()
    if creds is None:
        sessionmaker = app.state.sessionmaker
        async with sessionmaker() as session:
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
                store.set(creds)
                await session.execute(delete(LinearAuth))
                await session.commit()
    return LinearStatusResponse(
        connected=creds is not None,
        connected_at=creds.connected_at if creds is not None else None,
    )


@v1.get("/linear/oauth/start", include_in_schema=False)
async def linear_oauth_start() -> Response:
    settings = load_settings(repo_root=app.state.ctx.repo_root)
    state = new_oauth_state()
    code_verifier = new_pkce_verifier()
    app.state.linear_oauth_states[state] = (datetime.now(UTC), code_verifier)

    for key, created in list(app.state.linear_oauth_states.items()):
        created_at, _code_verifier = created
        if datetime.now(UTC) - created_at > timedelta(minutes=15):
            app.state.linear_oauth_states.pop(key, None)

    try:
        url = linear_authorize_url(
            settings,
            state=state,
            code_challenge=pkce_code_challenge(code_verifier),
        )
    except ValueError as e:
        return HTMLResponse(
            "<h1>Linear is not configured</h1>"
            "<p>Set <code>REDESMYN_LINEAR_CLIENT_ID</code> "
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
    _created_at, code_verifier = app.state.linear_oauth_states.pop(state)

    settings = load_settings(repo_root=app.state.ctx.repo_root)
    redirect_uri = linear_redirect_uri(settings)
    try:
        token = await exchange_code_for_token(
            settings, code=code, redirect_uri=redirect_uri, code_verifier=code_verifier
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    store = default_linear_credential_store()
    store.set(
        LinearCredentials(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type=token.token_type,
            scope=token.scope,
            expires_at=token.expires_at,
            connected_at=datetime.now(UTC),
        )
    )

    return HTMLResponse(
        "<h1>Linear connected</h1><p>You can close this tab and return to Redesmyn.</p>"
    )


@v1.get("/daemons", response_model=list[DaemonPresenceResponse])
async def list_daemons() -> list[DaemonPresenceResponse]:
    # NOTE: "connected" is derived from the in-process WS registry, not a DB time window.
    #
    # Multi-instance note: this only reports connections attached to *this* server
    # process. When the control plane runs with multiple instances, we will need
    # to route presence queries to the owning instance or move presence into a
    # shared distributed store.
    connected = await app.state.daemon_connections.snapshot()
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        rows = await session.scalars(select(Host).order_by(Host.host_key))
        result: list[DaemonPresenceResponse] = []
        for row in rows:
            runtime = connected.get(row.host_key)
            result.append(
                DaemonPresenceResponse(
                    host_key=row.host_key,
                    display_name=(
                        runtime.display_name
                        if runtime is not None
                        else row.display_name
                    ),
                    capabilities=(
                        runtime.capabilities
                        if runtime is not None
                        else row.capabilities
                    ),
                    attached_repos=[
                        RepoKeyResponse.model_validate(r)
                        for r in runtime.attached_repos
                    ]
                    if runtime is not None
                    else [],
                    connected=runtime is not None and runtime.connected,
                    last_seen_at=row.last_seen_at,
                    connected_at=runtime.connected_at if runtime is not None else None,
                    disconnected_at=None,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )
        return result


class IssueDaemonCommandRequest(BaseModel):
    command_type: str
    workspace_id: str | None = None
    repo_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


@v1.post("/daemons/{host_key}/commands", response_model=DaemonCommandResponse)
async def issue_daemon_command(
    host_key: str, req: IssueDaemonCommandRequest
) -> DaemonCommandResponse:
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        cmd = DaemonCommand(
            host_key=host_key,
            command_type=req.command_type,
            workspace_id=req.workspace_id,
            repo_id=req.repo_id,
            data=req.payload,
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
    await app.state.daemon_connections.send(
        host_key,
        {
            "type": "command",
            "command": ws_command.model_dump(),
        },
    )
    return command


@v1.websocket("/ws")
async def ui_event_stream(websocket: WebSocket) -> None:
    send_queue = await app.state.event_hub.connect(websocket)
    sender_task = asyncio.create_task(
        app.state.event_hub.sender_loop(websocket, send_queue)
    )
    try:
        while True:
            msg = await websocket.receive_json()
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type")
            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "id": msg.get("id")})
                continue
            if msg_type == "subscribe":
                since_id = msg.get("since_id")
                if not isinstance(since_id, int):
                    continue
                sessionmaker = app.state.sessionmaker
                async with sessionmaker() as session:
                    rows = await session.scalars(
                        select(Event).where(Event.id > since_id).order_by(Event.id)
                    )
                    for row in rows:
                        event = EventResponse.model_validate(row, from_attributes=True)
                        await websocket.send_json(
                            {"type": "event", "event": event.model_dump(by_alias=True)}
                        )
                continue
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        await app.state.event_hub.disconnect(websocket)


@v1.websocket("/daemon/ws")
async def daemon_ws(websocket: WebSocket, token: str | None = None) -> None:
    settings = load_settings(repo_root=app.state.ctx.repo_root)
    if token != settings.daemon_auth_token:
        await websocket.close(code=1008)
        return

    hello_adapter = TypeAdapter(DaemonInboundMessage)
    host_key: str | None = None
    connection_id: int | None = None

    await websocket.accept()
    try:
        raw = await asyncio.wait_for(websocket.receive_json(), timeout=10)
    except Exception:
        await websocket.close(code=1002)
        return

    try:
        first_msg = hello_adapter.validate_python(raw)
    except Exception:
        await websocket.close(code=1003)
        return

    if not isinstance(first_msg, DaemonHello):
        await websocket.close(code=1002)
        return

    host_key = first_msg.host_key
    attached_repos = [
        {"workspace_id": r.workspace_id, "repo_id": r.repo_id}
        for r in first_msg.attached_repos
    ]
    send_queue = await app.state.daemon_connections.register(
        host_key,
        websocket,
        attached_repos=attached_repos,
        capabilities=first_msg.capabilities,
        display_name=first_msg.display_name,
    )
    sender_task = asyncio.create_task(
        app.state.event_hub.sender_loop(websocket, send_queue)
    )

    now = datetime.now(UTC)
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        host = await session.scalar(select(Host).where(Host.host_key == host_key))
        if host is None:
            host = Host(
                host_key=host_key,
                display_name=first_msg.display_name or host_key,
                capabilities=first_msg.capabilities,
                last_seen_at=now,
            )
            session.add(host)
        else:
            host.display_name = first_msg.display_name or host.display_name
            host.capabilities = first_msg.capabilities
            host.last_seen_at = now

        await session.execute(
            update(DaemonConnection)
            .where(
                DaemonConnection.host_key == host_key,
                DaemonConnection.disconnected_at.is_(None),
            )
            .values(disconnected_at=now, disconnect_reason="replaced")
        )

        conn = DaemonConnection(
            host_key=host_key,
            display_name=first_msg.display_name,
            capabilities=first_msg.capabilities,
            attached_repos=attached_repos,
            connected_at=now,
            last_seen_at=now,
        )
        session.add(conn)

        repo = await session.scalar(
            select(Repository).where(
                Repository.repo_root == str(app.state.ctx.repo_root)
            )
        )
        if repo is not None and any(
            r.get("workspace_id") == repo.workspace_id
            and r.get("repo_id") == repo.repo_id
            for r in attached_repos
        ):
            await acquire_or_refresh_primary(
                session,
                RepoKey(workspace_id=repo.workspace_id, repo_id=repo.repo_id),
                host_key=host_key,
                now=now,
            )

        await session.commit()
        await session.refresh(conn)
        connection_id = conn.id

        connected_event = await _append_event(
            session,
            event_type="daemon.connected",
            data={
                "host_key": host_key,
                "connection_id": connection_id,
                "display_name": first_msg.display_name,
                "capabilities": first_msg.capabilities,
                "attached_repos": attached_repos,
            },
        )
    await _broadcast_event(connected_event)

    await websocket.send_json({"type": "hello_ack", "server_time": now.isoformat()})

    async with sessionmaker() as session:
        queued = await session.scalars(
            select(DaemonCommand)
            .where(
                DaemonCommand.host_key == host_key,
                DaemonCommand.state.in_([CommandState.Queued, CommandState.Running]),
            )
            .order_by(DaemonCommand.id)
        )
        for cmd in queued:
            command = DaemonCommandResponse.model_validate(cmd, from_attributes=True)
            ws_command = ServerCommand(
                command_id=command.id,
                command_type=command.command_type,
                workspace_id=command.workspace_id,
                repo_id=command.repo_id,
                data=command.payload,
            )
            await app.state.daemon_connections.send(
                host_key,
                {"type": "command", "command": ws_command.model_dump()},
            )

    try:
        while True:
            msg_raw = await websocket.receive_json()
            try:
                msg = hello_adapter.validate_python(msg_raw)
            except Exception:
                await websocket.send_json({"type": "error", "error": "invalid_message"})
                continue

            msg_type = msg.type
            now = datetime.now(UTC)
            if host_key is not None:
                await app.state.daemon_connections.note_heartbeat(host_key)

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "id": msg.id})
                if host_key is not None:
                    async with sessionmaker() as session:
                        host = await session.scalar(
                            select(Host).where(Host.host_key == host_key)
                        )
                        if host is not None and (
                            host.last_seen_at is None
                            or (now - host.last_seen_at) >= timedelta(seconds=30)
                        ):
                            host.last_seen_at = now
                            await session.commit()
                continue

            if msg_type == "heartbeat":
                async with sessionmaker() as session:
                    updated_attached_repos = attached_repos
                    if msg.attached_repos is not None:
                        updated_attached_repos = [
                            {"workspace_id": r.workspace_id, "repo_id": r.repo_id}
                            for r in msg.attached_repos
                        ]
                    if host_key is not None:
                        host = await session.scalar(
                            select(Host).where(Host.host_key == host_key)
                        )
                        if host is not None and (
                            host.last_seen_at is None
                            or (now - host.last_seen_at) >= timedelta(seconds=30)
                        ):
                            host.last_seen_at = now

                    if (
                        connection_id is not None
                        and updated_attached_repos != attached_repos
                    ):
                        await session.execute(
                            update(DaemonConnection)
                            .where(DaemonConnection.id == connection_id)
                            .values(attached_repos=updated_attached_repos)
                        )

                    repo = await session.scalar(
                        select(Repository).where(
                            Repository.repo_root == str(app.state.ctx.repo_root)
                        )
                    )
                    if (
                        repo is not None
                        and host_key is not None
                        and any(
                            r.get("workspace_id") == repo.workspace_id
                            and r.get("repo_id") == repo.repo_id
                            for r in updated_attached_repos
                        )
                    ):
                        await acquire_or_refresh_primary(
                            session,
                            RepoKey(
                                workspace_id=repo.workspace_id, repo_id=repo.repo_id
                            ),
                            host_key=host_key,
                            now=now,
                        )
                    await session.commit()

                if host_key is not None:
                    await app.state.daemon_connections.note_heartbeat(
                        host_key, attached_repos=updated_attached_repos
                    )
                attached_repos = updated_attached_repos
                await websocket.send_json({"type": "heartbeat_ack"})
                continue

            if msg_type == "event":
                async with sessionmaker() as session:
                    if host_key is not None:
                        host = await session.scalar(
                            select(Host).where(Host.host_key == host_key)
                        )
                        if host is not None and (
                            host.last_seen_at is None
                            or (now - host.last_seen_at) >= timedelta(seconds=30)
                        ):
                            host.last_seen_at = now
                            await session.commit()

                    if msg.event_type == "merge.run" and host_key is not None:
                        run_id_value = msg.data.get("run_id")
                        if isinstance(run_id_value, str) and run_id_value:
                            await apply_merge_run_event_update(
                                session,
                                run_id=run_id_value,
                                host_key=host_key,
                                data=msg.data,
                            )

                    event = await _append_event(
                        session,
                        event_type=msg.event_type,
                        data={
                            "workspace_id": msg.workspace_id,
                            "repo_id": msg.repo_id,
                            "host_key": host_key,
                            "connection_id": connection_id,
                            **msg.data,
                        },
                    )
                await _broadcast_event(event)
                await websocket.send_json({"type": "event_ack", "event_id": event.id})
                continue

            if msg_type == "command_ack":
                async with sessionmaker() as session:
                    cmd = await session.get(DaemonCommand, msg.command_id)
                    if cmd is None or cmd.host_key != host_key:
                        await websocket.send_json(
                            {"type": "error", "error": "unknown_command"}
                        )
                        continue
                    cmd.state = msg.state
                    cmd.ack_data = msg.data
                    await session.commit()

                    command_event = await _append_event(
                        session,
                        event_type="daemon.command_state_changed",
                        data={
                            "host_key": host_key,
                            "connection_id": connection_id,
                            "command_id": cmd.id,
                            "state": cmd.state.value,
                        },
                    )
                await _broadcast_event(command_event)
                await websocket.send_json({"type": "command_ack_ok"})
                continue
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        if host_key is not None:
            await app.state.daemon_connections.unregister(host_key)
            now = datetime.now(UTC)
            disconnected_event: EventResponse | None = None
            async with sessionmaker() as session:
                repo = await session.scalar(
                    select(Repository).where(
                        Repository.repo_root == str(app.state.ctx.repo_root)
                    )
                )
                if repo is not None:
                    await expire_primary_if_owner(
                        session,
                        RepoKey(workspace_id=repo.workspace_id, repo_id=repo.repo_id),
                        host_key=host_key,
                        now=now,
                    )
                if connection_id is not None:
                    await session.execute(
                        update(DaemonConnection)
                        .where(DaemonConnection.id == connection_id)
                        .values(disconnected_at=now, disconnect_reason="disconnected")
                    )
                    await session.commit()
                    disconnected_event = await _append_event(
                        session,
                        event_type="daemon.disconnected",
                        data={
                            "host_key": host_key,
                            "connection_id": connection_id,
                        },
                    )
            if disconnected_event is not None:
                await _broadcast_event(disconnected_event)


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
