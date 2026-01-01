from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState


def _drop_oldest(queue: asyncio.Queue[dict[str, Any]]) -> None:
    try:
        queue.get_nowait()
    except asyncio.QueueEmpty:
        return


@dataclass(frozen=True, slots=True)
class _WsClient:
    websocket: WebSocket
    send_queue: asyncio.Queue[dict[str, Any]]


class JsonWebSocketHub:
    def __init__(self, *, per_client_queue_size: int = 256) -> None:
        self._per_client_queue_size = per_client_queue_size
        self._clients: dict[WebSocket, _WsClient] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> asyncio.Queue[dict[str, Any]]:
        await websocket.accept()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._per_client_queue_size
        )
        async with self._lock:
            self._clients[websocket] = _WsClient(websocket=websocket, send_queue=queue)
        return queue

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.pop(websocket, None)

    async def publish(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients.values())
        for client in clients:
            if client.websocket.client_state != WebSocketState.CONNECTED:
                continue
            if client.send_queue.full():
                _drop_oldest(client.send_queue)
            try:
                client.send_queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def sender_loop(
        self,
        websocket: WebSocket,
        send_queue: asyncio.Queue[dict[str, Any]],
        *,
        on_send_error: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        while True:
            payload = await send_queue.get()
            try:
                await websocket.send_json(payload)
            except Exception as exc:
                if on_send_error is not None:
                    await on_send_error(exc)
                return


class DaemonConnectionRegistry:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # NOTE: This is a per-process runtime registry.
        #
        # If/when the control plane runs with multiple server instances, "connected"
        # presence cannot be derived from in-memory state alone. We'll need either:
        # - sticky routing so presence queries hit the instance holding the WebSocket, or
        # - a shared distributed store (e.g. Redis) for connection presence, or
        # - a durable projection + leases with a single writer.
        self._connections: dict[str, _DaemonConnection] = {}

    async def register(
        self,
        host_key: str,
        websocket: WebSocket,
        *,
        attached_repos: list[dict[str, str]],
        capabilities: dict[str, Any],
        display_name: str | None,
    ) -> asyncio.Queue[dict[str, Any]]:
        now = datetime.now(UTC)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._connections[host_key] = _DaemonConnection(
                websocket=websocket,
                send_queue=queue,
                connected_at=now,
                last_seen_at=now,
                attached_repos=attached_repos,
                capabilities=capabilities,
                display_name=display_name,
            )
        return queue

    async def unregister(self, host_key: str) -> None:
        async with self._lock:
            self._connections.pop(host_key, None)

    async def is_connected(self, host_key: str) -> bool:
        async with self._lock:
            conn = self._connections.get(host_key)
        return (
            conn is not None and conn.websocket.client_state == WebSocketState.CONNECTED
        )

    async def send(self, host_key: str, payload: dict[str, Any]) -> bool:
        async with self._lock:
            conn = self._connections.get(host_key)
        if conn is None or conn.websocket.client_state != WebSocketState.CONNECTED:
            return False
        if conn.send_queue.full():
            _drop_oldest(conn.send_queue)
        try:
            conn.send_queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            return False

    async def note_heartbeat(
        self,
        host_key: str,
        *,
        attached_repos: list[dict[str, str]] | None = None,
    ) -> _DaemonPresenceSnapshot | None:
        now = datetime.now(UTC)
        async with self._lock:
            conn = self._connections.get(host_key)
            if conn is None:
                return None
            conn.last_seen_at = now
            if attached_repos is not None:
                conn.attached_repos = attached_repos
            return _DaemonPresenceSnapshot(
                connected_at=conn.connected_at,
                last_seen_at=conn.last_seen_at,
                attached_repos=list(conn.attached_repos),
                capabilities=dict(conn.capabilities),
                display_name=conn.display_name,
                connected=conn.websocket.client_state == WebSocketState.CONNECTED,
            )

    async def snapshot(self) -> dict[str, _DaemonPresenceSnapshot]:
        async with self._lock:
            items = list(self._connections.items())
        out: dict[str, _DaemonPresenceSnapshot] = {}
        for host_key, conn in items:
            if conn.websocket.client_state != WebSocketState.CONNECTED:
                continue
            out[host_key] = _DaemonPresenceSnapshot(
                connected_at=conn.connected_at,
                last_seen_at=conn.last_seen_at,
                attached_repos=list(conn.attached_repos),
                capabilities=dict(conn.capabilities),
                display_name=conn.display_name,
                connected=True,
            )
        return out


@dataclass(slots=True)
class _DaemonConnection:
    websocket: WebSocket
    send_queue: asyncio.Queue[dict[str, Any]]
    connected_at: datetime
    last_seen_at: datetime
    attached_repos: list[dict[str, str]]
    capabilities: dict[str, Any]
    display_name: str | None


@dataclass(frozen=True, slots=True)
class _DaemonPresenceSnapshot:
    connected: bool
    connected_at: datetime
    last_seen_at: datetime
    attached_repos: list[dict[str, str]]
    capabilities: dict[str, Any]
    display_name: str | None
