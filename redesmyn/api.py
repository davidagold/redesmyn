from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.responses import Response

from redesmyn.context import RepoContext, get_repo_context
from redesmyn.db import (
    Block,
    BlockScope,
    Repository,
    create_engine,
    create_sessionmaker,
)
from redesmyn.domain.enums import BlockPolicy
from redesmyn.orchestrator import init_repo
from redesmyn.schemas.core import (
    ApiStatusResponse,
    BlockScopeResponse,
    BlockStatusResponse,
    ReleaseConditionResponse,
)


class AppState(Protocol):
    ctx: RepoContext
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]


class App(FastAPI):
    state: AppState


@asynccontextmanager
async def lifespan(app: App):
    ctx = get_repo_context()
    await init_repo(ctx)

    app.state.ctx = ctx
    app.state.engine = create_engine(ctx.db_path)
    app.state.sessionmaker = create_sessionmaker(app.state.engine)
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


@v1.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


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
