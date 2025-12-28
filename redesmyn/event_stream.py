from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import WebSocket
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.db import Epic, Event, Node
from redesmyn.schemas.core import EventResponse

WS_PROTOCOL_VERSION = 1


@dataclass(slots=True)
class EventStreamSubscription:
    epic: str | None
    after_id: int


async def current_last_event_id(session: AsyncSession) -> int:
    row = await session.scalar(select(Event).order_by(desc(Event.id)).limit(1))
    return row.id if row is not None else 0


async def _resolve_epic_id(session: AsyncSession, epic: str) -> int:
    if epic.isdigit():
        row = await session.get(Epic, int(epic))
    else:
        row = await session.scalar(select(Epic).where(Epic.slug == epic))
    if row is None:
        raise ValueError(f"Unknown epic: {epic}")
    return row.id


async def epic_node_ids(session: AsyncSession, epic: str) -> set[int]:
    epic_id = await _resolve_epic_id(session, epic)
    rows = await session.scalars(select(Node.id).where(Node.epic_id == epic_id))
    return set(rows)


def node_id_from_event(event: Event) -> int | None:
    try:
        payload = event.data
    except Exception:
        return None
    if isinstance(payload, dict):
        node_id = payload.get("node_id")
        if isinstance(node_id, int):
            return node_id
    return None


async def send_hello(websocket: WebSocket, *, last_event_id: int) -> None:
    await websocket.send_json(
        {
            "type": "hello",
            "protocol": WS_PROTOCOL_VERSION,
            "serverTime": datetime.now(UTC).isoformat(),
            "lastEventId": last_event_id,
        }
    )


async def send_event(websocket: WebSocket, event: Event) -> None:
    payload = EventResponse.model_validate(event, from_attributes=True).model_dump(
        by_alias=True
    )
    await websocket.send_json({"type": "event", "event": payload})


async def send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_json({"type": "error", "message": message})


async def send_resync(websocket: WebSocket, reason: str) -> None:
    await websocket.send_json({"type": "resync", "reason": reason})


async def send_pong(websocket: WebSocket) -> None:
    await websocket.send_json(
        {"type": "pong", "serverTime": datetime.now(UTC).isoformat()}
    )


def parse_client_message(raw: str) -> dict[str, object] | None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(msg, dict):
        return None
    return msg


async def run_event_stream(
    websocket: WebSocket,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    epic: str | None,
    after_id: int | None,
    poll_interval_s: float = 0.25,
    batch_size: int = 200,
) -> None:
    async with sessionmaker() as session:
        last_id = await current_last_event_id(session)
        node_ids = await epic_node_ids(session, epic) if epic else None

    active_epic = epic
    subscription = EventStreamSubscription(epic=epic, after_id=after_id or last_id)
    await send_hello(websocket, last_event_id=last_id)

    async def recv_loop() -> None:
        while True:
            raw = await websocket.receive_text()
            msg = parse_client_message(raw)
            if msg is None:
                await send_error(websocket, "Invalid JSON message")
                continue

            msg_type = msg.get("type")
            if msg_type == "ping":
                await send_pong(websocket)
                continue

            if msg_type == "subscribe":
                next_epic = msg.get("epic")
                if next_epic is not None and not isinstance(next_epic, str):
                    await send_error(websocket, "`epic` must be a string or null")
                    continue
                subscription.epic = next_epic

                next_after = msg.get("afterId")
                if next_after is not None:
                    if not isinstance(next_after, int):
                        await send_error(websocket, "`afterId` must be an integer")
                        continue
                    subscription.after_id = next_after
                continue

            await send_error(websocket, f"Unknown message type: {msg_type!r}")

    async def send_loop() -> None:
        nonlocal active_epic, node_ids
        while True:
            async with sessionmaker() as session:
                if subscription.epic is None:
                    active_epic = None
                    node_ids = None
                elif node_ids is None or subscription.epic != active_epic:
                    try:
                        node_ids = await epic_node_ids(session, subscription.epic)
                        active_epic = subscription.epic
                    except ValueError as e:
                        await send_error(websocket, str(e))
                        active_epic = None
                        node_ids = None

                rows = list(
                    await session.scalars(
                        select(Event)
                        .where(Event.id > subscription.after_id)
                        .order_by(Event.id)
                        .limit(batch_size)
                    )
                )

            if not rows:
                await asyncio.sleep(poll_interval_s)
                continue

            for row in rows:
                subscription.after_id = row.id
                node_id = node_id_from_event(row)
                if (
                    node_ids is not None
                    and node_id is not None
                    and node_id not in node_ids
                ):
                    continue
                await send_event(websocket, row)

            if len(rows) < batch_size:
                await asyncio.sleep(poll_interval_s)

    recv_task = asyncio.create_task(recv_loop())
    send_task = asyncio.create_task(send_loop())
    try:
        done, pending = await asyncio.wait(
            {recv_task, send_task}, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc
    finally:
        recv_task.cancel()
        send_task.cancel()
