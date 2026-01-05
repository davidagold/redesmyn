from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select

from redesmyn.context import RepoContext, build_repo_context
from redesmyn.db import Epic, Repository, Task, create_engine, create_sessionmaker
from redesmyn.domain.enums import TaskState
from redesmyn.host_identity import HostIdentity, host_identity_path
from redesmyn.orchestrator import init_repo

from tests.scenarios.scenario import ScenarioRepo


def _pick_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]
    finally:
        sock.close()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _unique_suffix() -> str:
    return uuid4().hex[:8]


def _write_host_identity(*, ctx: RepoContext, host_key: str) -> None:
    host_identity_path(ctx).write_text(
        HostIdentity(host_key=host_key, display_name="Redesmyn E2E").model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class SeededEpic:
    id: int
    slug: str


async def _seed_epic_with_tasks(ctx: RepoContext) -> SeededEpic:
    suffix = _unique_suffix()
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            repo = await session.scalar(
                select(Repository).where(Repository.repo_root == str(ctx.repo_root))
            )
            if repo is None:
                raise RuntimeError("Repository not initialized")

            epic = Epic(
                repository_id=repo.id,
                name="E2E Epic",
                slug=f"e2e-epic-{suffix}",
                root_branch=repo.default_branch,
            )
            session.add(epic)
            await session.flush()

            parent = Task(
                epic_id=epic.id,
                title="Parent task",
                body="# Parent task\n\nThis is seeded test data.\n",
                branch_name=f"e2e-parent-{suffix}",
                state=TaskState.Done,
            )
            session.add(parent)
            await session.flush()

            child = Task(
                epic_id=epic.id,
                title="Child task",
                body="# Child task\n\nThis is seeded test data.\n",
                branch_name=f"e2e-child-{suffix}",
                parent_task_id=parent.id,
                state=TaskState.InProgress,
            )
            session.add(child)
            await session.commit()
            return SeededEpic(id=epic.id, slug=epic.slug)
    finally:
        await engine.dispose()


def _wait_for_api(base_url: str, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/v1/status", timeout=1.0)
            if resp.status_code == 200:
                return
            last_error = f"status {resp.status_code}"
        except Exception as e:
            last_error = str(e)
        time.sleep(0.1)
    raise RuntimeError(f"API did not become ready ({last_error})")


@dataclass(frozen=True, slots=True)
class E2EServer:
    base_url: str
    proc: subprocess.Popen[str]
    epic_slug: str
    epic_id: int

    def terminate(self) -> None:
        if self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


@pytest.fixture
def e2e_server(tmp_path: Path) -> Iterator[E2EServer]:
    repo = ScenarioRepo.init(tmp_path)
    ctx = build_repo_context(repo_root=repo.repo_root, worktree_root=_project_root())
    ctx.state_dir.mkdir(parents=True, exist_ok=True)
    _write_host_identity(ctx=ctx, host_key="e2e-host-key")

    asyncio.run(init_repo(ctx, migrate=True))
    seeded_epic = asyncio.run(_seed_epic_with_tasks(ctx))

    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_project_root())
    env["REDESMYN_REPO_ROOT"] = str(ctx.repo_root)
    env["REDESMYN_WORKTREE_ROOT"] = str(ctx.worktree_root)
    env["REDESMYN_DB_PATH"] = str(ctx.db_path)
    env["REDESMYN_ENABLE_REPO_OBSERVER"] = "0"
    env["REDESMYN_ENABLE_AGENT_MONITOR"] = "0"
    env["REDESMYN_RUNNER_MODE"] = "local"

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "uvicorn",
        "redesmyn.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_project_root()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    server = E2EServer(
        base_url=base_url,
        proc=proc,
        epic_slug=seeded_epic.slug,
        epic_id=seeded_epic.id,
    )
    try:
        try:
            _wait_for_api(base_url, timeout_s=20.0)
        except Exception:
            server.terminate()
            output = proc.stdout.read() if proc.stdout is not None else ""
            raise RuntimeError("e2e server did not become ready:\n" + output) from None
        yield server
    finally:
        server.terminate()
        if proc.stdout is not None:
            proc.stdout.close()


@pytest.fixture
def e2e_base_url(e2e_server: E2EServer) -> str:
    return e2e_server.base_url


@pytest.fixture
def e2e_epic_slug(e2e_server: E2EServer) -> str:
    return e2e_server.epic_slug


@pytest.fixture
def e2e_epic_id(e2e_server: E2EServer) -> int:
    return e2e_server.epic_id
