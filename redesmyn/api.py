from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from redesmyn.context import get_repo_context

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


@app.on_event("startup")
async def _startup() -> None:
    ctx = get_repo_context()
    app.state.repo_root = ctx.repo_root
    maybe_mount_dashboard(app, ctx.repo_root)

