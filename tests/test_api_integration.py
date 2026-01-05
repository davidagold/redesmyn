from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import select

from redesmyn.agent_runtime import has_tmux, tmux_session_name_for_task
from redesmyn.api import _append_event, _broadcast_event
from redesmyn.db import AgentSession
from redesmyn.db import MergeRun
from redesmyn.domain.enums import AgentStatus
from redesmyn.domain.enums import MergeRunStatus

from tests.helpers.ws import JsonQueueWebSocket
from tests.scenarios.scenario import Scenario
from tests.scenarios.variants import seed_merged_parent, seed_running_agent


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
    websocket = JsonQueueWebSocket()

    send_queue = await app.state.event_hub.connect(websocket)  # type: ignore[arg-type]
    sender_task = asyncio.create_task(
        app.state.event_hub.sender_loop(websocket, send_queue)  # type: ignore[arg-type]
    )
    try:
        async with scenario.db.session() as session:
            event = await _append_event(session, event_type="test.event", data={})
        await _broadcast_event(app, event)

        payload = await asyncio.wait_for(websocket.sent_queue.get(), timeout=1.0)
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
async def test_restack_creates_merge_run_record(scenario: Scenario) -> None:
    seeded = await seed_merged_parent(scenario)
    run_id = "test-restack-run-id"

    response = await scenario.app.client.post(
        f"/v1/tasks/{seeded.child_task_id}/restack",
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
async def test_merge_rejects_invalid_scope_returns_422(scenario: Scenario) -> None:
    seeded = await seed_merged_parent(scenario)
    response = await scenario.app.client.post(
        f"/v1/tasks/{seeded.child_task_id}/merge",
        json={"scope": "nope"},
    )
    assert response.status_code == 422


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


@pytest.mark.integration
async def test_start_and_restart_agent_persists_db_state(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not has_tmux():
        pytest.skip("tmux is required for agent start/restart integration tests")

    seeded = await seed_merged_parent(scenario)
    prefix = f"rn-test-{uuid4().hex[:10]}"
    monkeypatch.setenv("REDESMYN_TMUX_SESSION_PREFIX", prefix)

    harness = "sh -lc 'printf \"\\n> \"; sleep 30'"
    tmux_name = tmux_session_name_for_task(task_id=seeded.child_task_id)

    try:
        start = await scenario.app.client.post(
            f"/v1/tasks/{seeded.child_task_id}/agent/start",
            json={"harness": harness, "detach": True},
        )
        assert start.status_code == 200, start.text
        start_body = start.json()
        assert start_body["taskId"] == seeded.child_task_id
        assert start_body["agentSessionId"]
        assert start_body["agentStatus"] == AgentStatus.Running.value
        assert start_body["attach"]["type"] == "tmux"
        assert start_body["attach"]["session"] == tmux_name

        async with scenario.db.session() as session:
            started_session = await session.get(
                AgentSession, int(start_body["agentSessionId"])
            )
            assert started_session is not None
            assert started_session.task_id == seeded.child_task_id
            assert started_session.status == AgentStatus.Running
            assert started_session.ended_at is None

        restart = await scenario.app.client.post(
            f"/v1/tasks/{seeded.child_task_id}/agent/restart",
            json={"detach": True},
        )
        assert restart.status_code == 200, restart.text
        restart_body = restart.json()
        assert int(restart_body["agentSessionId"]) != int(start_body["agentSessionId"])
        assert restart_body["agentStatus"] == AgentStatus.Running.value
        assert restart_body["attach"]["type"] == "tmux"
        assert restart_body["attach"]["session"] == tmux_name
    finally:
        await scenario.app.client.post(f"/v1/tasks/{seeded.child_task_id}/agent/stop")
