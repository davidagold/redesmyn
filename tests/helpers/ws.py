from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any

from starlette.websockets import WebSocketDisconnect, WebSocketState

_CLOSE_SENTINEL = object()


class JsonQueueWebSocket:
    """
    Minimal websocket test double that asserts `send_json()` payloads are
    JSON-serializable and records them in an awaitable queue.
    """

    client_state = WebSocketState.CONNECTED

    def __init__(self, *, max_payloads: int = 64) -> None:
        self._payloads = deque(maxlen=max_payloads)
        self.sent_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return list(self._payloads)

    async def accept(self) -> None:  # pragma: no cover - interface parity
        return

    async def send_json(self, payload: dict[str, Any]) -> None:
        json.dumps(payload)
        self._payloads.append(payload)
        self.sent_queue.put_nowait(payload)


class InProcessWebSocket:
    """
    In-process websocket harness used to drive WS handlers directly without
    binding ports.

    - `send_to_server()` pushes a JSON payload for `receive_json()`.
    - `recv_from_server()` returns payloads sent via `send_json()`.
    """

    client_state: WebSocketState = WebSocketState.CONNECTING

    def __init__(self, *, app) -> None:
        self.scope = {"app": app}
        self._incoming: asyncio.Queue[object] = asyncio.Queue()
        self._outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def accept(self) -> None:
        self.client_state = WebSocketState.CONNECTED

    async def close(self, code: int = 1000) -> None:
        _ = code
        self.client_state = WebSocketState.DISCONNECTED
        try:
            self._incoming.put_nowait(_CLOSE_SENTINEL)
        except asyncio.QueueFull:
            pass

    async def receive_json(self) -> object:
        msg = await self._incoming.get()
        if msg is _CLOSE_SENTINEL:
            raise WebSocketDisconnect(code=1000)
        return msg

    async def send_json(self, payload: dict[str, Any]) -> None:
        json.dumps(payload)
        self._outgoing.put_nowait(payload)

    def send_to_server(self, payload: dict[str, Any]) -> None:
        self._incoming.put_nowait(payload)

    async def recv_from_server(self, *, timeout_s: float = 1.0) -> dict[str, Any]:
        return await asyncio.wait_for(self._outgoing.get(), timeout=timeout_s)
