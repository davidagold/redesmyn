from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, cast
from uuid import uuid4

from fastapi import WebSocket
import pytest
from sqlalchemy import select

from redesmyn.api import _append_event, _broadcast_event, daemon_ws
from redesmyn.db import MergeRun, Repository
from redesmyn.domain.enums import MergeRunStatus

from tests.helpers.ws import InProcessWebSocket
from tests.scenarios.scenario import Scenario
from tests.scenarios.seeds.git import seed_merged_parent


async def _recv_until_type(
    ws: InProcessWebSocket, expected: str, *, timeout_s: float = 1.0
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        msg = await ws.recv_from_server(timeout_s=remaining)
        if msg.get("type") == expected:
            return msg


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


@pytest.mark.integration
async def test_daemon_attach_updates_repo_executor_status(
    scenario_without_primary_executor: Scenario,
) -> None:
    scenario = scenario_without_primary_executor
    seeded = await seed_merged_parent(scenario)
    workspace_id, repo_id = await _repo_key(scenario)

    before = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert before.status_code == 200
    assert before.json()["repoExecutor"]["attachedHostKeys"] == []

    await scenario.daemon.connect(
        host_key="daemon-a",
        attached_repos=[{"workspace_id": workspace_id, "repo_id": repo_id}],
    )

    after = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert after.status_code == 200
    assert after.json()["repoExecutor"]["attachedHostKeys"] == ["daemon-a"]


@pytest.mark.integration
async def test_server_routes_commands_to_attached_host_key(
    scenario_without_primary_executor: Scenario,
) -> None:
    scenario = scenario_without_primary_executor
    workspace_id, repo_id = await _repo_key(scenario)

    conn_a = await scenario.daemon.connect(
        host_key="daemon-a",
        attached_repos=[{"workspace_id": workspace_id, "repo_id": repo_id}],
    )
    conn_b = await scenario.daemon.connect(host_key="daemon-b", attached_repos=[])

    response = await scenario.app.client.post(
        "/v1/daemons/daemon-a/commands",
        json={
            "command_type": "test.command",
            "workspace_id": workspace_id,
            "repo_id": repo_id,
            "payload": {"k": "v"},
        },
    )
    assert response.status_code == 200

    delivered = await conn_a.recv(timeout_s=1.0)
    assert delivered["type"] == "command"
    assert delivered["command"]["command_type"] == "test.command"
    assert delivered["command"]["workspace_id"] == workspace_id
    assert delivered["command"]["repo_id"] == repo_id
    assert delivered["command"]["data"] == {"k": "v"}

    with pytest.raises(asyncio.TimeoutError):
        await conn_b.recv(timeout_s=0.2)


@pytest.mark.integration
async def test_merge_run_events_progress_merge_run_state_happy_path(
    scenario_without_primary_executor: Scenario,
) -> None:
    scenario = scenario_without_primary_executor
    seeded = await seed_merged_parent(scenario)
    workspace_id, repo_id = await _repo_key(scenario)
    run_id = uuid4().hex

    async with scenario.db.session() as session:
        session.add(
            MergeRun(
                run_id=run_id,
                epic_id=seeded.epic_id,
                requested_task_id=seeded.child_task_id,
                canonical=True,
                status=MergeRunStatus.Running,
                scope="descendants",
            )
        )
        await session.commit()

    ws = InProcessWebSocket(app=scenario.app.app)
    daemon_task = asyncio.create_task(daemon_ws(ws, token="dev"))
    try:
        ws.send_to_server(
            {
                "type": "hello",
                "host_key": "daemon-a",
                "capabilities": {},
                "attached_repos": [{"workspace_id": workspace_id, "repo_id": repo_id}],
            }
        )
        await _recv_until_type(ws, "hello_ack", timeout_s=2.0)

        ws.send_to_server(
            {
                "type": "event",
                "workspace_id": workspace_id,
                "repo_id": repo_id,
                "event_type": "merge.run",
                "data": {
                    "run_id": run_id,
                    "task_id": seeded.child_task_id,
                    "epic_id": seeded.epic_id,
                    "requested_task_id": seeded.child_task_id,
                    "status": "running",
                    "operation": "merge",
                },
            }
        )
        await _recv_until_type(ws, "event_ack", timeout_s=2.0)

        ws.send_to_server(
            {
                "type": "event",
                "workspace_id": workspace_id,
                "repo_id": repo_id,
                "event_type": "merge.run",
                "data": {
                    "run_id": run_id,
                    "task_id": seeded.child_task_id,
                    "epic_id": seeded.epic_id,
                    "requested_task_id": seeded.child_task_id,
                    "status": "succeeded",
                    "operation": "merge",
                },
            }
        )
        await _recv_until_type(ws, "event_ack", timeout_s=2.0)

        async with scenario.db.session() as session:
            row = await session.scalar(
                select(MergeRun).where(MergeRun.run_id == run_id)
            )
            assert row is not None
            assert row.host_key == "daemon-a"
            assert row.status == MergeRunStatus.Succeeded
    finally:
        await ws.close()
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(daemon_task, timeout=2.0)
        if not daemon_task.done():
            daemon_task.cancel()
            with suppress(asyncio.CancelledError):
                await daemon_task


@pytest.mark.integration
async def test_ws_event_stream_payload_is_json_serializable(
    scenario: Scenario,
) -> None:
    app = scenario.app.app
    event_hub = app.state.event_hub

    ws = InProcessWebSocket(app=app)
    websocket = cast(WebSocket, ws)
    send_queue = await event_hub.connect(websocket)  # registers ws for publish()
    sender_task = asyncio.create_task(event_hub.sender_loop(websocket, send_queue))
    try:
        async with scenario.db.session() as session:
            event = await _append_event(
                session,
                event_type="daemon.connected",
                data={"host_key": "daemon-a", "connection_id": 123},
            )
        await _broadcast_event(app, event)

        msg = await _recv_until_type(ws, "event", timeout_s=2.0)
        created_at = msg["event"]["createdAt"]
        assert isinstance(created_at, str)
    finally:
        sender_task.cancel()
        with suppress(asyncio.CancelledError):
            await sender_task
        await event_hub.disconnect(websocket)
