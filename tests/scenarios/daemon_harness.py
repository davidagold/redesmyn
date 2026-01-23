from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import asynccontextmanager, suppress
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.api import daemon_ws
from redesmyn.background_tasks import BackgroundTaskManager
from redesmyn.daemon_runtime import DaemonRuntime, DaemonRuntimeConfig
from redesmyn.db import DaemonCommand, Repository
from redesmyn.domain.enums import CommandState
from redesmyn.host_identity import HostIdentity, host_identity_path
from redesmyn.schemas.core import (
    EpicGraphResponse,
    EpicResponse,
    MergeRunSummaryResponse,
)
from redesmyn.settings import RedesmynSettings
from redesmyn.ws_protocol import ServerCommand

from tests.helpers.ws import InProcessWebSocket
from tests.scenarios.scenario import Scenario


_MAX_CAPTURED_FRAMES = 200

T = TypeVar("T")


@dataclass(slots=True)
class InProcessControlPlaneClient:
    client: Any

    async def list_epics(self) -> list[EpicResponse]:
        resp = await self.client.get("/v1/epics")
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        return [EpicResponse.model_validate(item) for item in payload]

    async def epic_graph(self, epic_slug: str) -> EpicGraphResponse:
        resp = await self.client.get(f"/v1/epics/{epic_slug}/graph")
        resp.raise_for_status()
        return EpicGraphResponse.model_validate(resp.json())

    async def find_merge_run(self, run_id: str) -> MergeRunSummaryResponse | None:
        try:
            epics = await self.list_epics()
        except Exception:
            return None
        for epic in epics:
            try:
                graph = await self.epic_graph(epic.slug)
            except Exception:
                continue
            for run in graph.merge_runs:
                if run.run_id == run_id:
                    return run
        return None


@dataclass(slots=True)
class _WebSocketBridge:
    server_ws: InProcessWebSocket
    daemon_to_server: asyncio.Queue[dict[str, Any]]
    server_to_daemon: asyncio.Queue[dict[str, Any]]
    _incoming_text: asyncio.Queue[str]
    daemon_to_server_history: deque[dict[str, Any]]
    server_to_daemon_history: deque[dict[str, Any]]

    async def send(self, data: str | bytes) -> None:
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8", errors="replace")
            except Exception:
                return

        try:
            parsed = json.loads(data)
        except Exception:
            return
        if not isinstance(parsed, dict):
            return

        self.daemon_to_server_history.append(parsed)
        self.daemon_to_server.put_nowait(parsed)
        self.server_ws.send_to_server(parsed)

    async def recv(self) -> str:
        return await self._incoming_text.get()

    async def pump_server_to_daemon(self) -> None:
        while True:
            payload = await self.server_ws.outgoing_queue.get()
            self.server_to_daemon_history.append(payload)
            self.server_to_daemon.put_nowait(payload)
            self._incoming_text.put_nowait(json.dumps(payload))


@dataclass(slots=True)
class DaemonRuntimeHarness:
    scenario: Scenario
    daemon: DaemonRuntime
    api: InProcessControlPlaneClient
    _server_ws: InProcessWebSocket
    _bridge: _WebSocketBridge
    _send_queue: asyncio.Queue[dict[str, Any]]
    _command_queue: asyncio.Queue[ServerCommand]
    _background: BackgroundTaskManager
    _tasks: list[asyncio.Task[None]]
    _daemon_message_buffer: list[dict[str, Any]]
    _server_message_buffer: list[dict[str, Any]]
    _daemon_message_lock: asyncio.Lock
    _server_message_lock: asyncio.Lock

    @classmethod
    @asynccontextmanager
    async def open_ctx(
        cls,
        scenario: Scenario,
        *,
        host_key: str,
    ) -> AsyncIterator["DaemonRuntimeHarness"]:
        harness = await cls.open(scenario, host_key=host_key)
        try:
            yield harness
        except Exception as exc:
            if "DaemonRuntimeHarness diagnostics:" not in str(exc):
                print(harness.format_diagnostics())
            raise
        finally:
            await harness.aclose()

    @classmethod
    async def open(cls, scenario: Scenario, *, host_key: str) -> "DaemonRuntimeHarness":
        _write_host_identity(scenario, host_key=host_key)
        workspace_id, repo_id = await _repo_key(scenario)

        config = DaemonRuntimeConfig(
            control_plane_url="http://test",
            token="dev",
            workspace_id=workspace_id,
            repo_id=repo_id,
            poll_interval_s=9999.0,
            heartbeat_interval_s=9999.0,
        )
        settings = RedesmynSettings(
            repo_root=scenario.ctx.repo_root,
            worktree_root=scenario.ctx.worktree_root,
            db_path=scenario.ctx.db_path,
            runner_mode="remote",
            enable_repo_observer=False,
            enable_agent_monitor=False,
        )
        daemon = DaemonRuntime(scenario.ctx, settings, config)

        server_ws = InProcessWebSocket(app=scenario.app.app)
        server_task: asyncio.Task[None] = asyncio.create_task(
            daemon_ws(server_ws, token="dev")
        )

        bridge = _WebSocketBridge(
            server_ws=server_ws,
            daemon_to_server=asyncio.Queue(),
            server_to_daemon=asyncio.Queue(),
            _incoming_text=asyncio.Queue(),
            daemon_to_server_history=deque(maxlen=_MAX_CAPTURED_FRAMES),
            server_to_daemon_history=deque(maxlen=_MAX_CAPTURED_FRAMES),
        )
        pump_task: asyncio.Task[None] = asyncio.create_task(
            bridge.pump_server_to_daemon()
        )

        send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        command_queue: asyncio.Queue[ServerCommand] = asyncio.Queue(maxsize=128)
        api = InProcessControlPlaneClient(client=scenario.app.client)

        sender_task = asyncio.create_task(daemon._sender_loop(bridge, send_queue))
        receiver_task = asyncio.create_task(
            daemon._receiver_loop(bridge, command_queue)
        )
        background = BackgroundTaskManager()
        command_task = asyncio.create_task(
            daemon._command_loop(api, command_queue, send_queue, background=background)
        )

        server_message_buffer: list[dict[str, Any]] = []
        await bridge.send(daemon.build_hello().model_dump_json())
        await _wait_for_server_message_type(
            bridge.server_to_daemon,
            expected="hello_ack",
            timeout_s=2.0,
            buffer=server_message_buffer,
        )

        return cls(
            scenario=scenario,
            daemon=daemon,
            api=api,
            _server_ws=server_ws,
            _bridge=bridge,
            _send_queue=send_queue,
            _command_queue=command_queue,
            _background=background,
            _tasks=[server_task, pump_task, sender_task, receiver_task, command_task],
            _daemon_message_buffer=[],
            _server_message_buffer=server_message_buffer,
            _daemon_message_lock=asyncio.Lock(),
            _server_message_lock=asyncio.Lock(),
        )

    async def aclose(self) -> None:
        await self._server_ws.close()

        for task in self._tasks:
            if task.done():
                continue
            task.cancel()

        await self._background.cancel_and_await()

        with suppress(Exception):
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def heartbeat_tick(self) -> None:
        await self.daemon.heartbeat_tick(self._send_queue)

    async def telemetry_tick(self) -> None:
        await self.daemon.telemetry_tick(self.api, self._send_queue)

    async def create_and_send_command(
        self,
        *,
        command_type: str,
        workspace_id: str,
        repo_id: str,
        payload: dict[str, Any],
    ) -> int:
        async with self.scenario.db.session() as session:
            row = DaemonCommand(
                host_key=self.daemon.host_key,
                command_type=command_type,
                workspace_id=workspace_id,
                repo_id=repo_id,
                data=payload,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            command_id = row.id

        ws_command = ServerCommand(
            command_id=command_id,
            command_type=command_type,
            workspace_id=workspace_id,
            repo_id=repo_id,
            data=payload,
        )
        await self.scenario.app.app.state.daemon_connections.send(
            self.daemon.host_key,
            {"type": "command", "command": ws_command.model_dump()},
        )
        return command_id

    def format_diagnostics(self, *, limit: int = 25) -> str:
        return "\n".join(
            [
                "DaemonRuntimeHarness diagnostics:",
                *self._format_frame_section(
                    "daemon->server (recent)",
                    list(self._bridge.daemon_to_server_history),
                    limit=limit,
                ),
                *self._format_frame_section(
                    "server->daemon (recent)",
                    list(self._bridge.server_to_daemon_history),
                    limit=limit,
                ),
                *self._format_frame_section(
                    "daemon->server (buffered)",
                    self._daemon_message_buffer,
                    limit=limit,
                ),
                *self._format_frame_section(
                    "server->daemon (buffered)",
                    self._server_message_buffer,
                    limit=limit,
                ),
            ]
        )

    def _format_frame_section(
        self, title: str, frames: list[dict[str, Any]], *, limit: int
    ) -> list[str]:
        lines = [f"{title}: {len(frames)}"]
        for frame in frames[-limit:]:
            lines.append(f"  {self._format_frame_one_line(frame)}")
        return lines

    def _format_frame_one_line(self, frame: dict[str, Any]) -> str:
        try:
            return json.dumps(frame, sort_keys=True)
        except Exception:
            return repr(frame)

    def _format_timeout_error(
        self,
        *,
        expectation: str,
        timeout_s: float,
        direction: str,
    ) -> AssertionError:
        return AssertionError(
            "\n".join(
                [
                    f"Timed out waiting for {expectation} (timeout={timeout_s:.3f}s, direction={direction}).",
                    self.format_diagnostics(),
                ]
            )
        )

    async def _wait_for_message(
        self,
        *,
        direction: str,
        queue: asyncio.Queue[dict[str, Any]],
        buffer: list[dict[str, Any]],
        lock: asyncio.Lock,
        match: Callable[[dict[str, Any]], bool],
        expectation: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        async with lock:
            idx = next((i for i, msg in enumerate(buffer) if match(msg)), None)
            if idx is not None:
                return buffer.pop(idx)

            deadline = asyncio.get_running_loop().time() + timeout_s
            while True:
                remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=remaining)
                except TimeoutError:
                    raise self._format_timeout_error(
                        expectation=expectation,
                        timeout_s=timeout_s,
                        direction=direction,
                    ) from None
                if match(msg):
                    return msg
                buffer.append(msg)

    async def wait_for_daemon_message(
        self,
        *,
        expected_type: str,
        timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        return await self._wait_for_message(
            direction="daemon->server",
            queue=self._bridge.daemon_to_server,
            buffer=self._daemon_message_buffer,
            lock=self._daemon_message_lock,
            match=lambda msg: msg.get("type") == expected_type,
            expectation=f"daemon message type={expected_type!r}",
            timeout_s=timeout_s,
        )

    async def wait_for_server_message(
        self,
        *,
        expected_type: str,
        timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        return await self._wait_for_message(
            direction="server->daemon",
            queue=self._bridge.server_to_daemon,
            buffer=self._server_message_buffer,
            lock=self._server_message_lock,
            match=lambda msg: msg.get("type") == expected_type,
            expectation=f"server message type={expected_type!r}",
            timeout_s=timeout_s,
        )

    async def wait_for_command_state(
        self,
        *,
        command_id: int,
        state: CommandState,
        timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        expected = state.value if hasattr(state, "value") else str(state)
        return await self._wait_for_message(
            direction="daemon->server",
            queue=self._bridge.daemon_to_server,
            buffer=self._daemon_message_buffer,
            lock=self._daemon_message_lock,
            match=lambda msg: (
                msg.get("type") == "command_ack"
                and msg.get("command_id") == command_id
                and msg.get("state") == expected
            ),
            expectation=f"command_ack command_id={command_id} state={expected!r}",
            timeout_s=timeout_s,
        )

    async def wait_for_event(
        self,
        *,
        event_type: str,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        def _match(msg: dict[str, Any]) -> bool:
            if msg.get("type") != "event" or msg.get("event_type") != event_type:
                return False
            if predicate is not None and not predicate(msg):
                return False
            return True

        return await self._wait_for_message(
            direction="daemon->server",
            queue=self._bridge.daemon_to_server,
            buffer=self._daemon_message_buffer,
            lock=self._daemon_message_lock,
            match=_match,
            expectation=f"event event_type={event_type!r}",
            timeout_s=timeout_s,
        )

    async def wait_for_db_state(
        self,
        *,
        fetch: Callable[[AsyncSession], Awaitable[T | None]],
        predicate: Callable[[T], bool],
        timeout_s: float = 2.0,
        expectation: str,
    ) -> T:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            async with self.scenario.db.session() as session:
                value = await fetch(session)
            if value is not None and predicate(value):
                return value
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(
                    "\n".join(
                        [
                            f"Timed out waiting for DB state: {expectation} (timeout={timeout_s:.3f}s).",
                            self.format_diagnostics(),
                        ]
                    )
                )
            await asyncio.sleep(0)


def _write_host_identity(scenario: Scenario, *, host_key: str) -> None:
    host_identity_path(scenario.ctx).write_text(
        HostIdentity(host_key=host_key, display_name="Redesmyn Tests").model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )


async def _repo_key(scenario: Scenario) -> tuple[str, str]:
    async with scenario.db.session() as session:
        repo = await session.scalar(
            select(Repository).where(
                Repository.repo_root == str(scenario.ctx.repo_root)
            )
        )
        if repo is None:
            raise RuntimeError("Scenario repository row missing")
        return repo.workspace_id, repo.repo_id


async def _wait_for_server_message_type(
    queue: asyncio.Queue[dict[str, Any]],
    *,
    expected: str,
    timeout_s: float,
    buffer: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        msg = await asyncio.wait_for(queue.get(), timeout=remaining)
        if msg.get("type") == expected:
            return msg
        if buffer is not None:
            buffer.append(msg)
