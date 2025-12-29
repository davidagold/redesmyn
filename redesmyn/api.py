from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.responses import RedirectResponse

from redesmyn.agent_monitor import run_agent_monitor
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
    Node,
    Repository,
    Task,
    create_engine,
    create_sessionmaker,
)
from redesmyn.db.models import HostCapabilities
from redesmyn.domain.enums import BlockPolicy
from redesmyn.event_stream import run_event_stream
from redesmyn.integrations.linear import (
    exchange_code_for_token,
    linear_authorize_url,
    linear_redirect_uri,
    new_oauth_state,
)
from redesmyn.orchestrator import init_repo
from redesmyn.orchestration_config import load_orchestration_defaults
from redesmyn.repo import git_commit_info, git_merge_base, git_rev_list
from redesmyn.repo_observer import run_repo_observer
from redesmyn.runner_backend import (
    RunnerBackend,
    RunnerBackendError,
    make_runner_backend,
)
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
    NodeSetAgentRequest,
    NodeResponse,
    OrchestrationDefaultsResponse,
    OrchestrationFleetDefaultsResponse,
    OrchestrationHarnessDefaultsResponse,
    ReleaseConditionResponse,
    TaskResponse,
    TaskAgentRestartRequest,
    TaskAgentStartRequest,
    TaskAgentStartResponse,
    TaskAgentStopResponse,
    TrunkCommitResponse,
    TrunkTimelineResponse,
)
from redesmyn.settings import load_settings


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

    return EpicGraphResponse(
        epic=EpicResponse.model_validate(epic_row, from_attributes=True),
        tasks=[TaskResponse.model_validate(t, from_attributes=True) for t in tasks],
        nodes=[NodeResponse.model_validate(n, from_attributes=True) for n in nodes],
        agents=[AgentResponse.model_validate(a, from_attributes=True) for a in agents],
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
    try:
        result = await app.state.runner_backend.start_task_agent(
            task_id=task_id,
            harness_command=request.harness,
            detach=request.detach,
        )
    except RunnerBackendError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    row = result.agent
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        _, node = await _require_task_node(session, task_id=task_id)

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
        started=result.started,
        warnings=list(result.warnings),
    )


@v1.post("/tasks/{task_id}/agent/stop", response_model=TaskAgentStopResponse)
async def stop_task_agent(task_id: int) -> TaskAgentStopResponse:
    try:
        stopped = await app.state.runner_backend.stop_task_agent(task_id=task_id)
    except RunnerBackendError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        _, node = await _require_task_node(session, task_id=task_id)
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
    try:
        result = await app.state.runner_backend.restart_task_agent(
            task_id=task_id,
            harness_command=request.harness,
            detach=request.detach,
        )
    except RunnerBackendError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    row = result.agent
    sessionmaker = app.state.sessionmaker
    async with sessionmaker() as session:
        _, node = await _require_task_node(session, task_id=task_id)

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
        started=result.started,
        warnings=list(result.warnings),
    )


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
        ),
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
