from __future__ import annotations

import asyncio
import subprocess
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.websockets import WebSocketState

from redesmyn.api import App, app as global_app, lifespan as app_lifespan
from redesmyn.context import RepoContext
from redesmyn.db import create_engine, create_sessionmaker, init_db
from redesmyn.ws_runtime import DaemonConnectionRegistry


class GitError(RuntimeError):
    pass


def _run_git(repo_root: Path, args: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd or repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GitError(proc.stderr.strip() or "git command failed")
    return proc.stdout.strip()


@dataclass(frozen=True, slots=True)
class ScenarioRepo:
    repo_root: Path
    worktrees_root: Path

    @classmethod
    def init(cls, tmp_path: Path) -> "ScenarioRepo":
        repo_root = tmp_path / "repo"
        worktrees_root = tmp_path / "worktrees"
        repo_root.mkdir(parents=True, exist_ok=False)
        worktrees_root.mkdir(parents=True, exist_ok=False)

        _run_git(repo_root, ["init", "-b", "main"], cwd=repo_root)
        _run_git(repo_root, ["config", "user.email", "tests@example.invalid"])
        _run_git(repo_root, ["config", "user.name", "Redesmyn Tests"])
        _run_git(repo_root, ["config", "commit.gpgsign", "false"])

        (repo_root / "README.md").write_text("test repo\n", encoding="utf-8")
        _run_git(repo_root, ["add", "-A"])
        _run_git(repo_root, ["commit", "-m", "init"])

        return cls(repo_root=repo_root, worktrees_root=worktrees_root)

    def create_worktree(self, *, branch_name: str, from_ref: str = "main") -> Path:
        path = self.worktrees_root / branch_name
        if path.exists():
            raise FileExistsError(str(path))
        _run_git(
            self.repo_root,
            ["worktree", "add", "-b", branch_name, str(path), from_ref],
        )
        return path

    def commit_file(
        self,
        *,
        worktree_path: Path,
        relpath: str,
        content: str,
        message: str,
    ) -> str:
        file_path = worktree_path / relpath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        _run_git(self.repo_root, ["add", "-A"], cwd=worktree_path)
        _run_git(self.repo_root, ["commit", "-m", message], cwd=worktree_path)
        return _run_git(self.repo_root, ["rev-parse", "HEAD"], cwd=worktree_path)

    def fast_forward_main(self, *, from_branch: str) -> None:
        _run_git(self.repo_root, ["checkout", "main"], cwd=self.repo_root)
        _run_git(
            self.repo_root, ["merge", "--ff-only", from_branch], cwd=self.repo_root
        )


@dataclass(slots=True)
class ScenarioDB:
    db_path: Path
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]

    @classmethod
    async def connect(cls, *, db_path: Path) -> "ScenarioDB":
        engine = create_engine(db_path)
        await init_db(engine, migrate=False)
        return cls(
            db_path=db_path,
            engine=engine,
            sessionmaker=create_sessionmaker(engine),
        )

    async def aclose(self) -> None:
        await self.engine.dispose()


@dataclass(slots=True)
class ScenarioApp:
    app: App
    client: httpx.AsyncClient
    _lifespan: AbstractAsyncContextManager[None]

    @classmethod
    async def open(cls) -> "ScenarioApp":
        lifespan_cm = app_lifespan(global_app)
        await lifespan_cm.__aenter__()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=global_app), base_url="http://test"
        )
        return cls(app=global_app, client=client, _lifespan=lifespan_cm)

    async def aclose(self) -> None:
        await self.client.aclose()
        await self._lifespan.__aexit__(None, None, None)


class _FakeWebSocket:
    client_state = WebSocketState.CONNECTED

    async def send_json(self, _payload: dict[str, Any]) -> None:  # pragma: no cover
        return


@dataclass(frozen=True, slots=True)
class ScenarioDaemonConnection:
    host_key: str
    send_queue: asyncio.Queue[dict[str, Any]]

    async def recv(self, *, timeout_s: float = 1.0) -> dict[str, Any]:
        return await asyncio.wait_for(self.send_queue.get(), timeout=timeout_s)


@dataclass(frozen=True, slots=True)
class ScenarioDaemon:
    registry: DaemonConnectionRegistry

    @classmethod
    def from_app(cls, app: App) -> "ScenarioDaemon":
        return cls(registry=app.state.daemon_connections)

    async def connect(
        self,
        *,
        host_key: str,
        attached_repos: list[dict[str, str]] | None = None,
        capabilities: dict[str, Any] | None = None,
        display_name: str | None = None,
    ) -> ScenarioDaemonConnection:
        queue = await self.registry.register(
            host_key,
            _FakeWebSocket(),  # type: ignore[arg-type]
            attached_repos=attached_repos or [],
            capabilities=capabilities or {},
            display_name=display_name,
        )
        return ScenarioDaemonConnection(host_key=host_key, send_queue=queue)


@dataclass(slots=True)
class Scenario:
    ctx: RepoContext
    repo: ScenarioRepo
    db: ScenarioDB
    app: ScenarioApp
    daemon: ScenarioDaemon
    host_key: str

    async def aclose(self) -> None:
        await self.app.aclose()
        await self.db.aclose()
