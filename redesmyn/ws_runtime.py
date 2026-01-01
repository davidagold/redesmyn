from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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
        self._connections: dict[str, _WsClient] = {}

    async def register(
        self, daemon_id: str, websocket: WebSocket
    ) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._connections[daemon_id] = _WsClient(
                websocket=websocket, send_queue=queue
            )
        return queue

    async def unregister(self, daemon_id: str) -> None:
        async with self._lock:
            self._connections.pop(daemon_id, None)

    async def is_connected(self, daemon_id: str) -> bool:
        async with self._lock:
            conn = self._connections.get(daemon_id)
        return (
            conn is not None and conn.websocket.client_state == WebSocketState.CONNECTED
        )

    async def send(self, daemon_id: str, payload: dict[str, Any]) -> bool:
        async with self._lock:
            conn = self._connections.get(daemon_id)
        if conn is None or conn.websocket.client_state != WebSocketState.CONNECTED:
            return False
        if conn.send_queue.full():
            _drop_oldest(conn.send_queue)
        try:
            conn.send_queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            return False
