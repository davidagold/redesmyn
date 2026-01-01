from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from types import TracebackType
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import httpx

from redesmyn.agent_runtime import has_tmux, load_or_create_runner_host_config
from redesmyn.context import RepoContext
from redesmyn.db.models import HostCapabilities
from redesmyn.domain.enums import AgentStatus, CommandState
from redesmyn.repo import branch_exists, git_is_ancestor
from redesmyn.repo_observer import RepoObserverState, observe_repo
from redesmyn.schemas.core import EpicGraphResponse, EpicResponse, NodeResponse
from redesmyn.settings import RedesmynSettings
from redesmyn.ws_protocol import (
    DaemonCommandAck,
    DaemonEvent,
    DaemonHeartbeat,
    DaemonHello,
    RepoKey,
    ServerCommand,
)


@dataclass(slots=True)
class Backoff:
    initial_s: float = 0.5
    maximum_s: float = 30.0
    factor: float = 1.7
    jitter_fraction: float = 0.2
    _current_s: float = 0.5

    def __post_init__(self) -> None:
        self._current_s = self.initial_s

    def reset(self) -> None:
        self._current_s = self.initial_s

    def next_delay_s(self) -> float:
        delay = self._current_s
        self._current_s = min(self.maximum_s, self._current_s * self.factor)
        jitter = delay * self.jitter_fraction
        return max(0.0, delay + random.uniform(-jitter, jitter))


def _ws_url_for_control_plane(*, control_plane_url: str, token: str) -> str:
    parsed = urlparse(control_plane_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"control plane URL must start with http:// or https:// (got {control_plane_url!r})"
        )
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"token": token})
    ws = parsed._replace(
        scheme=scheme,
        path="/v1/daemon/ws",
        query=query,
        fragment="",
    )
    return urlunparse(ws)


def _drop_oldest(queue: asyncio.Queue[dict[str, Any]]) -> None:
    try:
        queue.get_nowait()
    except asyncio.QueueEmpty:
        return


@dataclass(frozen=True, slots=True)
class DaemonRuntimeConfig:
    control_plane_url: str
    token: str
    workspace_id: str
    repo_id: str
    poll_interval_s: float = 1.0
    heartbeat_interval_s: float = 5.0


class ControlPlaneClient:
    def __init__(self, *, control_plane_url: str, timeout_s: float = 5.0) -> None:
        self._client = httpx.AsyncClient(base_url=control_plane_url, timeout=timeout_s)

    async def __aenter__(self) -> "ControlPlaneClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_epics(self) -> list[EpicResponse]:
        resp = await self._client.get("/v1/epics")
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        return [EpicResponse.model_validate(item) for item in payload]

    async def epic_graph(self, epic_slug: str) -> EpicGraphResponse:
        resp = await self._client.get(f"/v1/epics/{epic_slug}/graph")
        resp.raise_for_status()
        return EpicGraphResponse.model_validate(resp.json())


class DaemonRuntime:
    def __init__(
        self,
        ctx: RepoContext,
        settings: RedesmynSettings,
        config: DaemonRuntimeConfig,
    ) -> None:
        self._ctx = ctx
        self._settings = settings
        self._config = config

        host_config = load_or_create_runner_host_config(ctx)
        self._daemon_id = host_config.host_key
        self._host = host_config.display_name

        self._attached_repos = [
            RepoKey(workspace_id=config.workspace_id, repo_id=config.repo_id)
        ]

        capabilities = HostCapabilities(
            tmux_available=has_tmux(),
            supports_path_shim=True,
        )
        self._capabilities = capabilities.model_dump(mode="python")

        self._observer_state = RepoObserverState()
        self._last_stack_in_sync_by_node_id: dict[int, bool | None] = {}

    async def run_forever(self) -> None:
        backoff = Backoff()
        while True:
            try:
                await self._run_connected()
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(backoff.next_delay_s())

    async def _run_connected(self) -> None:
        import websockets

        ws_url = _ws_url_for_control_plane(
            control_plane_url=self._config.control_plane_url,
            token=self._config.token,
        )

        async with ControlPlaneClient(
            control_plane_url=self._config.control_plane_url
        ) as api:
            async with websockets.connect(ws_url, ping_interval=None) as websocket:
                # Treat each websocket session as a fresh "epoch" so we can resync
                # derived projections on reconnect by emitting a baseline.
                self._observer_state = RepoObserverState()
                self._last_stack_in_sync_by_node_id = {}

                hello = DaemonHello(
                    daemon_id=self._daemon_id,
                    host=self._host,
                    capabilities=self._capabilities,
                    attached_repos=self._attached_repos,
                )
                await websocket.send(hello.model_dump_json())

                send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
                command_queue: asyncio.Queue[ServerCommand] = asyncio.Queue(maxsize=128)

                tasks = [
                    asyncio.create_task(self._sender_loop(websocket, send_queue)),
                    asyncio.create_task(self._receiver_loop(websocket, command_queue)),
                    asyncio.create_task(self._heartbeat_loop(send_queue)),
                    asyncio.create_task(self._telemetry_loop(api, send_queue)),
                    asyncio.create_task(self._command_loop(command_queue, send_queue)),
                ]
                try:
                    done, pending = await asyncio.wait(
                        set(tasks), return_when=asyncio.FIRST_EXCEPTION
                    )
                    for task in done:
                        exc = task.exception()
                        if exc is not None:
                            raise exc
                finally:
                    for task in tasks:
                        task.cancel()

    async def _sender_loop(
        self, websocket: Any, send_queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        while True:
            payload = await send_queue.get()
            await websocket.send(json.dumps(payload))

    async def _receiver_loop(
        self, websocket: Any, command_queue: asyncio.Queue[ServerCommand]
    ) -> None:
        while True:
            raw = await websocket.recv()
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8", errors="replace")
                except Exception:
                    continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "command":
                cmd_raw = msg.get("command")
                if not isinstance(cmd_raw, dict):
                    continue
                try:
                    cmd = ServerCommand.model_validate(cmd_raw)
                except Exception:
                    continue
                if command_queue.full():
                    try:
                        command_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    command_queue.put_nowait(cmd)
                except asyncio.QueueFull:
                    pass

    async def _heartbeat_loop(self, send_queue: asyncio.Queue[dict[str, Any]]) -> None:
        while True:
            heartbeat = DaemonHeartbeat(attached_repos=self._attached_repos)
            await self._enqueue(send_queue, heartbeat.model_dump(mode="python"))
            await asyncio.sleep(self._config.heartbeat_interval_s)

    async def _telemetry_loop(
        self, api: ControlPlaneClient, send_queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        while True:
            try:
                emit_baseline = not self._observer_state.initialized
                await self._emit_repo_telemetry(
                    api, send_queue, emit_baseline=emit_baseline
                )
                self._observer_state.initialized = True
            except Exception:
                pass
            await asyncio.sleep(self._config.poll_interval_s)

    async def _emit_repo_telemetry(
        self,
        api: ControlPlaneClient,
        send_queue: asyncio.Queue[dict[str, Any]],
        *,
        emit_baseline: bool,
    ) -> None:
        try:
            epics = await api.list_epics()
        except Exception:
            epics = []

        nodes: list[NodeResponse] = []
        active_agent_id_by_node_id: dict[int, int] = {}
        upstream_by_node_id: dict[int, str | None] = {}

        for epic in epics:
            graph = await api.epic_graph(epic.slug)
            nodes.extend(graph.nodes)
            agents_by_id = {a.id: a for a in graph.agents}
            nodes_by_id = {n.id: n for n in graph.nodes}
            for node in graph.nodes:
                if node.agent_id is not None:
                    agent = agents_by_id.get(node.agent_id)
                    if agent is not None and agent.status in {
                        AgentStatus.Running,
                        AgentStatus.Blocked,
                    }:
                        active_agent_id_by_node_id[node.id] = node.agent_id

                if node.parent_node_id is None:
                    upstream_by_node_id[node.id] = epic.root_branch
                else:
                    parent = nodes_by_id.get(node.parent_node_id)
                    upstream_by_node_id[node.id] = (
                        parent.branch_name if parent is not None else None
                    )

        if not nodes:
            return

        events = observe_repo(
            self._ctx,
            nodes,
            self._observer_state,
            emit_baseline=emit_baseline,
            active_agent_id_by_node_id=active_agent_id_by_node_id,
        )
        for event in events:
            await self._send_event(
                send_queue,
                event_type=event.event_type,
                data=dict(event.data),
            )

        missing: object = object()
        for node in nodes:
            upstream = upstream_by_node_id.get(node.id)
            value: bool | None
            if upstream is None:
                value = None
            elif not branch_exists(self._ctx.repo_root, node.branch_name):
                value = None
            elif not branch_exists(self._ctx.repo_root, upstream):
                value = None
            else:
                value = git_is_ancestor(self._ctx.repo_root, upstream, node.branch_name)

            prev = self._last_stack_in_sync_by_node_id.get(node.id, missing)
            if prev is not missing and prev == value:
                continue
            self._last_stack_in_sync_by_node_id[node.id] = value
            if prev is missing and value is None and not emit_baseline:
                continue
            await self._send_event(
                send_queue,
                event_type="node.stack_in_sync",
                data={
                    "node_id": node.id,
                    "branch_name": node.branch_name,
                    "upstream_ref": upstream,
                    "stack_in_sync": value,
                },
            )

    async def _command_loop(
        self,
        command_queue: asyncio.Queue[ServerCommand],
        send_queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        while True:
            cmd = await command_queue.get()
            await self._ack_command(send_queue, cmd.command_id, CommandState.Running)
            try:
                await self._execute_command(cmd, send_queue)
            except Exception as exc:
                await self._ack_command(
                    send_queue,
                    cmd.command_id,
                    CommandState.Failed,
                    data={"error": str(exc)},
                )
                continue
            await self._ack_command(
                send_queue,
                cmd.command_id,
                CommandState.Succeeded,
            )

    async def _execute_command(
        self, cmd: ServerCommand, send_queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        if cmd.command_type == "daemon.ping":
            return

        if cmd.command_type == "daemon.emit":
            await self._send_event(
                send_queue,
                event_type="daemon.event",
                data=cmd.data,
            )
            return

        raise ValueError(f"Unknown command_type: {cmd.command_type!r}")

    async def _ack_command(
        self,
        send_queue: asyncio.Queue[dict[str, Any]],
        command_id: int,
        state: CommandState,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        ack = DaemonCommandAck(
            command_id=command_id,
            state=state,
            data=data or {},
        )
        await self._enqueue(send_queue, ack.model_dump(mode="python"))

    async def _send_event(
        self,
        send_queue: asyncio.Queue[dict[str, Any]],
        *,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        event = DaemonEvent(
            workspace_id=self._config.workspace_id,
            repo_id=self._config.repo_id,
            event_type=event_type,
            data=data,
        )
        await self._enqueue(send_queue, event.model_dump(mode="python"))

    async def _enqueue(
        self, send_queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]
    ) -> None:
        if send_queue.full():
            _drop_oldest(send_queue)
        try:
            send_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def run_daemon(
    ctx: RepoContext, settings: RedesmynSettings, config: DaemonRuntimeConfig
) -> None:
    daemon = DaemonRuntime(ctx, settings, config)
    await daemon.run_forever()
