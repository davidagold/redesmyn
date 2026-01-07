from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket
from pydantic import ValidationError
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.db import Epic, Event, Task
from redesmyn.schemas.core import EventResponse
from redesmyn.ws_runtime import JsonWebSocketHub

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


async def epic_task_ids(session: AsyncSession, epic: str) -> set[int]:
    epic_id = await _resolve_epic_id(session, epic)
    rows = await session.scalars(select(Task.id).where(Task.epic_id == epic_id))
    return set(rows)


def task_id_from_event(event: Event) -> int | None:
    try:
        payload = event.data
    except Exception:
        return None
    if isinstance(payload, dict):
        task_id = payload.get("task_id")
        if isinstance(task_id, int):
            return task_id
    return None


def _task_id_from_hub_payload(payload: dict[str, Any]) -> int | None:
    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    for key in ("task_id", "taskId"):
        task_id = data.get(key)
        if isinstance(task_id, int):
            return task_id
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
    raw_data = event.data if isinstance(event.data, dict) else {}
    payload_dict = {
        "id": event.id,
        "event_type": event.event_type,
        "created_at": event.created_at,
        "data": {**raw_data, "type": event.event_type},
    }
    try:
        payload = EventResponse.model_validate(payload_dict).model_dump(
            by_alias=True, mode="json"
        )
    except ValidationError:
        payload_dict["data"] = {
            "type": "unknown",
            "event_type": event.event_type,
            "data": raw_data,
        }
        payload = EventResponse.model_validate(payload_dict).model_dump(
            by_alias=True, mode="json"
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
    event_hub: JsonWebSocketHub | None = None,
    poll_interval_s: float = 0.25,
    batch_size: int = 200,
) -> None:
    hub_queue: asyncio.Queue[dict[str, Any]] | None = None
    if event_hub is not None:
        hub_queue = await event_hub.connect(websocket)
    else:
        await websocket.accept()

    try:
        async with sessionmaker() as session:
            last_id = await current_last_event_id(session)
            task_ids = await epic_task_ids(session, epic) if epic else None

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

        async def _refresh_task_filter() -> None:
            nonlocal active_epic, task_ids
            if subscription.epic is None:
                active_epic = None
                task_ids = None
                return
            if task_ids is not None and subscription.epic == active_epic:
                return
            async with sessionmaker() as session:
                try:
                    task_ids = await epic_task_ids(session, subscription.epic)
                    active_epic = subscription.epic
                except ValueError as e:
                    await send_error(websocket, str(e))
                    active_epic = None
                    task_ids = None

        async def _poll_db_and_send() -> bool:
            await _refresh_task_filter()
            async with sessionmaker() as session:
                rows = list(
                    await session.scalars(
                        select(Event)
                        .where(Event.id > subscription.after_id)
                        .order_by(Event.id)
                        .limit(batch_size)
                    )
                )
            if not rows:
                return False
            for row in rows:
                subscription.after_id = row.id
                task_id = task_id_from_event(row)
                if (
                    task_ids is not None
                    and task_id is not None
                    and task_id not in task_ids
                ):
                    continue
                await send_event(websocket, row)
            return True

        async def _handle_hub_payload(payload: dict[str, Any]) -> None:
            # Hub-published payloads are treated as a wake-up signal, not the
            # source of truth. We still deliver events from the DB in order so
            # we don't skip earlier IDs if hub notifications arrive out-of-order.
            if payload.get("type") != "event":
                return
            if _task_id_from_hub_payload(payload) is None:
                # Non-task-scoped events still advance the global cursor; fetch
                # from DB so ordering stays consistent.
                await _poll_db_and_send()
                return
            await _poll_db_and_send()

        async def send_loop() -> None:
            while True:
                # DB is the source of truth and provides catch-up. We preferentially
                # deliver hub-published events for low latency, but still poll the DB
                # periodically to ensure we don't miss non-hub emitters.
                if await _poll_db_and_send():
                    continue

                if hub_queue is None:
                    await asyncio.sleep(poll_interval_s)
                    continue

                try:
                    payload = await asyncio.wait_for(
                        hub_queue.get(), timeout=poll_interval_s
                    )
                except asyncio.TimeoutError:
                    continue
                await _handle_hub_payload(payload)

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
    finally:
        if event_hub is not None:
            await event_hub.disconnect(websocket)
