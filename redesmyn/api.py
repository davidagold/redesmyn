from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.responses import Response
from starlette.responses import RedirectResponse

from redesmyn.context import RepoContext, get_repo_context
from redesmyn.db import (
    Agent,
    Block,
    BlockScope,
    Epic,
    LinearAuth,
    Node,
    Repository,
    Task,
    create_engine,
    create_sessionmaker,
)
from redesmyn.domain.enums import BlockPolicy
from redesmyn.integrations.linear import (
    exchange_code_for_token,
    linear_authorize_url,
    linear_redirect_uri,
    new_oauth_state,
)
from redesmyn.orchestrator import init_repo
from redesmyn.schemas.core import (
    ApiStatusResponse,
    AgentResponse,
    BlockScopeResponse,
    BlockStatusResponse,
    EpicGraphResponse,
    EpicResponse,
    LinearStatusResponse,
    NodeResponse,
    ReleaseConditionResponse,
    TaskResponse,
)
from redesmyn.settings import load_settings


class AppState(Protocol):
    ctx: RepoContext
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    linear_oauth_states: dict[str, datetime]


class App(FastAPI):
    state: AppState


@asynccontextmanager
async def lifespan(app: App):
    ctx = get_repo_context()
    await init_repo(ctx)

    app.state.ctx = ctx
    app.state.engine = create_engine(ctx.db_path)
    app.state.sessionmaker = create_sessionmaker(app.state.engine)
    app.state.linear_oauth_states = {}
    maybe_mount_dashboard(app, ctx.repo_root)

    yield

    await app.state.engine.dispose()


def maybe_mount_dashboard(app_: FastAPI, repo_root: Path) -> None:
    if (dist_path := _dist_path(repo_root)).is_dir():
        app_.mount(
            "/", StaticFiles(directory=str(dist_path), html=True), name="dashboard"
        )


app = App(title="Redesmyn", lifespan=lifespan)
v1 = APIRouter(prefix="/v1")
legacy = APIRouter(prefix="/api")


@v1.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@legacy.get("/status", response_model=ApiStatusResponse, include_in_schema=False)
async def legacy_api_status() -> ApiStatusResponse:
    return await api_status()


@legacy.get(
    "/linear/status", response_model=LinearStatusResponse, include_in_schema=False
)
async def legacy_linear_status() -> LinearStatusResponse:
    return await linear_status()


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
                    .order_by(Agent.id)
                )
            )
            if agent_ids
            else []
        )

    return EpicGraphResponse(
        epic=EpicResponse.model_validate(epic_row, from_attributes=True),
        tasks=[TaskResponse.model_validate(t, from_attributes=True) for t in tasks],
        nodes=[NodeResponse.model_validate(n, from_attributes=True) for n in nodes],
        agents=[AgentResponse.model_validate(a, from_attributes=True) for a in agents],
    )


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
        token = await exchange_code_for_token(settings, code=code, redirect_uri=redirect_uri)
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
    index_html = ctx.repo_root / "dashboard" / "dist" / "index.html"
    if index_html.is_file():
        return FileResponse(str(index_html))

    return HTMLResponse(
        "<h1>Redesmyn</h1><p>Dashboard not built yet. Build with `cd dashboard && npm run build`.</p>"
    )


def _dist_path(repo_root: Path) -> Path:
    return repo_root / "dashboard" / "dist"


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
app.include_router(legacy)
