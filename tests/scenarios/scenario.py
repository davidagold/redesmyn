from __future__ import annotations

import asyncio
import json
import subprocess
from collections import deque
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.websockets import WebSocketState

from redesmyn.api import App
from redesmyn.context import RepoContext
from redesmyn.db import create_engine, create_sessionmaker, init_db
from redesmyn.ws_runtime import DaemonConnectionRegistry, JsonWebSocketHub


class GitError(RuntimeError):
    pass


def run_git(repo_root: Path, args: list[str], *, cwd: Path | None = None) -> str:
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

        run_git(repo_root, ["init", "-b", "main"], cwd=repo_root)
        run_git(repo_root, ["config", "user.email", "tests@example.invalid"])
        run_git(repo_root, ["config", "user.name", "Redesmyn Tests"])
        run_git(repo_root, ["config", "commit.gpgsign", "false"])

        (repo_root / "README.md").write_text("test repo\n", encoding="utf-8")
        run_git(repo_root, ["add", "-A"])
        run_git(repo_root, ["commit", "-m", "init"])

        return cls(repo_root=repo_root, worktrees_root=worktrees_root)

    def git(self, args: list[str], *, cwd: Path | None = None) -> str:
        return run_git(self.repo_root, args, cwd=cwd)

    def create_worktree(self, *, branch_name: str, from_ref: str = "main") -> Path:
        path = self.worktrees_root / branch_name
        if path.exists():
            raise FileExistsError(str(path))
        self.git(["worktree", "add", "-b", branch_name, str(path), from_ref])
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
        self.git(["add", "-A"], cwd=worktree_path)
        self.git(["commit", "-m", message], cwd=worktree_path)
        return self.git(["rev-parse", "HEAD"], cwd=worktree_path)

    def fast_forward_main(self, *, from_branch: str) -> None:
        self.git(["checkout", "main"], cwd=self.repo_root)
        self.git(["merge", "--ff-only", from_branch], cwd=self.repo_root)

    def rebase(self, *, worktree_path: Path, upstream_ref: str) -> str:
        return self.git(["rebase", upstream_ref], cwd=worktree_path)

    def try_rebase(self, *, worktree_path: Path, upstream_ref: str) -> bool:
        proc = subprocess.run(
            ["git", "rebase", upstream_ref],
            cwd=str(worktree_path),
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0


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

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as session:
            yield session

    async def aclose(self) -> None:
        await self.engine.dispose()


@dataclass(slots=True)
class ScenarioApp:
    app: App
    client: httpx.AsyncClient
    _lifespan: AbstractAsyncContextManager[Any]

    @classmethod
    async def open(cls, app: App) -> "ScenarioApp":
        lifespan_cm = app.router.lifespan_context(app)
        await lifespan_cm.__aenter__()  # type: ignore[call-arg]
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        return cls(app=app, client=client, _lifespan=lifespan_cm)

    async def aclose(self) -> None:
        await self.client.aclose()
        await self._lifespan.__aexit__(None, None, None)


class _FakeWebSocket:
    client_state = WebSocketState.CONNECTED

    def __init__(self, *, max_payloads: int = 64) -> None:
        self._payloads = deque(maxlen=max_payloads)
        self.sent_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return list(self._payloads)

    async def send_json(self, payload: dict[str, Any]) -> None:
        json.dumps(payload)
        self._payloads.append(payload)
        self.sent_queue.put_nowait(payload)


@dataclass(frozen=True, slots=True)
class ScenarioDaemonConnection:
    host_key: str
    websocket: _FakeWebSocket
    sender_task: asyncio.Task[None]

    async def recv(self, *, timeout_s: float = 1.0) -> dict[str, Any]:
        return await asyncio.wait_for(
            self.websocket.sent_queue.get(), timeout=timeout_s
        )

    async def aclose(self) -> None:
        self.sender_task.cancel()
        try:
            await self.sender_task
        except asyncio.CancelledError:
            pass


@dataclass(frozen=True, slots=True)
class ScenarioDaemon:
    registry: DaemonConnectionRegistry
    event_hub: JsonWebSocketHub

    @classmethod
    def from_app(cls, app: App) -> "ScenarioDaemon":
        return cls(registry=app.state.daemon_connections, event_hub=app.state.event_hub)

    async def connect(
        self,
        *,
        host_key: str,
        attached_repos: list[dict[str, str]] | None = None,
        capabilities: dict[str, Any] | None = None,
        display_name: str | None = None,
    ) -> ScenarioDaemonConnection:
        websocket = _FakeWebSocket()
        queue = await self.registry.register(
            host_key,
            websocket,  # type: ignore[arg-type]
            attached_repos=attached_repos or [],
            capabilities=capabilities or {},
            display_name=display_name,
        )
        sender_task = asyncio.create_task(
            self.event_hub.sender_loop(websocket, queue)  # type: ignore[arg-type]
        )
        return ScenarioDaemonConnection(
            host_key=host_key, websocket=websocket, sender_task=sender_task
        )


@dataclass(slots=True)
class Scenario:
    ctx: RepoContext
    repo: ScenarioRepo
    db: ScenarioDB
    app: ScenarioApp
    daemon: ScenarioDaemon
    host_key: str

    async def aclose(self) -> None:
        # NOTE: Daemon connections are owned by the app runtime and will be torn down
        # with the app lifespan; scenarios should explicitly close any connections
        # they open via `ScenarioDaemon.connect`.
        await self.app.aclose()
        await self.db.aclose()
