from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy import select
from starlette.websockets import WebSocketState

from redesmyn.api import _append_event, _broadcast_event
from redesmyn.db import MergeRun
from redesmyn.domain.enums import MergeRunStatus

from tests.scenarios.scenario import Scenario
from tests.scenarios.variants import seed_merged_parent, seed_running_agent


class _JsonCapturingWebSocket:
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.sent: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def accept(self) -> None:  # pragma: no cover - part of WS interface
        return

    async def send_json(self, payload: dict[str, object]) -> None:
        json.dumps(payload)
        await self.sent.put(payload)


@pytest.mark.integration
async def test_unhandled_exception_returns_500_with_request_id_header_and_detail(
    scenario: Scenario,
) -> None:
    async def boom() -> None:
        raise RuntimeError("boom")

    scenario.app.app.add_api_route("/v1/__test__/boom", boom, methods=["GET"])

    response = await scenario.app.client.get("/v1/__test__/boom")
    assert response.status_code == 500

    request_id = response.headers.get("x-request-id")
    assert request_id
    assert request_id in response.json()["detail"]


@pytest.mark.integration
async def test_event_payloads_are_json_serializable(scenario: Scenario) -> None:
    app = scenario.app.app
    websocket = _JsonCapturingWebSocket()

    send_queue = await app.state.event_hub.connect(websocket)  # type: ignore[arg-type]
    sender_task = asyncio.create_task(
        app.state.event_hub.sender_loop(websocket, send_queue)  # type: ignore[arg-type]
    )
    try:
        async with scenario.db.session() as session:
            event = await _append_event(session, event_type="test.event", data={})
        await _broadcast_event(app, event)

        payload = await asyncio.wait_for(websocket.sent.get(), timeout=1.0)
        assert payload.get("type") == "event"
    finally:
        sender_task.cancel()
        await asyncio.gather(sender_task, return_exceptions=True)
        await app.state.event_hub.disconnect(websocket)  # type: ignore[arg-type]


@pytest.mark.integration
async def test_merge_creates_merge_run_record(scenario: Scenario) -> None:
    seeded = await seed_merged_parent(scenario)
    run_id = "test-merge-run-id"

    response = await scenario.app.client.post(
        f"/v1/tasks/{seeded.child_task_id}/merge",
        json={"run_id": run_id},
    )
    assert response.status_code == 200, response.text

    async with scenario.db.session() as session:
        run = await session.scalar(select(MergeRun).where(MergeRun.run_id == run_id))
        assert run is not None
        assert run.epic_id == seeded.epic_id
        assert run.requested_task_id == seeded.child_task_id
        assert run.status in {MergeRunStatus.Running, MergeRunStatus.Succeeded}


@pytest.mark.integration
async def test_merge_returns_409_with_running_agents_prefix_contract(
    scenario: Scenario,
) -> None:
    seeded = await seed_running_agent(scenario)

    response = await scenario.app.client.post(
        f"/v1/tasks/{seeded.child_task_id}/merge",
        json={},
    )
    assert response.status_code == 409

    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert detail.startswith("RUNNING_AGENTS:")


@pytest.mark.integration
async def test_canonical_merge_without_primary_executor_returns_503_guidance(
    scenario_without_primary_executor: Scenario,
) -> None:
    response = await scenario_without_primary_executor.app.client.post(
        "/v1/tasks/123/merge",
        json={},
    )
    assert response.status_code == 503

    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert "Start a daemon and attach this repo" in detail
