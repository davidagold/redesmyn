from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import desc, select
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from redesmyn.context import get_repo_context
from redesmyn.db import Pause, Repository, create_engine, create_sessionmaker
from redesmyn.orchestrator import init_repo

app = FastAPI(title="Redesmyn")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return "<h1>Redesmyn</h1><p>Dashboard not built yet.</p>"


def maybe_mount_dashboard(app_: FastAPI, repo_root: Path) -> None:
    dist = repo_root / "dashboard" / "dist"
    if dist.is_dir():
        app_.mount("/", StaticFiles(directory=str(dist), html=True), name="dashboard")


@app.get("/api/status")
async def api_status() -> dict[str, object]:
    ctx = app.state.ctx
    sessionmaker = app.state.sessionmaker

    async with sessionmaker() as session:
        repo = await session.scalar(select(Repository).where(Repository.repo_root == str(ctx.repo_root)))
        pause = await session.scalar(
            select(Pause)
            .where(Pause.scope == "repo", Pause.cleared_at.is_(None))
            .order_by(desc(Pause.id))
            .limit(1)
        )

    return {
        "repoRoot": str(ctx.repo_root),
        "dbPath": str(ctx.db_path),
        "defaultBranch": repo.default_branch if repo else None,
        "pause": None
        if pause is None
        else {"mode": pause.mode, "scope": pause.scope, "reason": pause.reason},
    }


@app.on_event("startup")
async def _startup() -> None:
    ctx = get_repo_context()
    await init_repo(ctx)

    app.state.ctx = ctx
    app.state.engine = create_engine(ctx.db_path)
    app.state.sessionmaker = create_sessionmaker(app.state.engine)

    maybe_mount_dashboard(app, ctx.repo_root)


@app.on_event("shutdown")
async def _shutdown() -> None:
    engine = getattr(app.state, "engine", None)
    if engine is not None:
        await engine.dispose()
