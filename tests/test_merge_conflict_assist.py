from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

import redesmyn.merge_conflict_assist as conflict_assist
from redesmyn.db import AgentSession, Event, MergeRun
from redesmyn.domain.enums import AgentStatus, AgentTurnState, MergeRunStatus
from redesmyn.merge_conflict_assist import MergeConflictAssistSupervisor

from tests.scenarios.scenario import Scenario


@dataclass(slots=True)
class _FakeSender:
    sent: list[tuple[int, str]]
    ok: bool = True

    async def send(self, *, agent_session: AgentSession, text: str) -> bool:
        self.sent.append((agent_session.id, text))
        return self.ok


@dataclass(slots=True)
class _FakeResumer:
    resumed: list[str]
    ok: bool = True

    async def resume(self, *, run: MergeRun) -> bool:
        self.resumed.append(run.run_id)
        return self.ok


async def _load_blocked_merge_run(scenario: Scenario) -> MergeRun:
    async with scenario.db.session() as session:
        run = await session.scalar(
            select(MergeRun).order_by(MergeRun.id.desc()).limit(1)
        )
        assert run is not None
        return run


@pytest.mark.integration
async def test_conflict_assist_sends_remediation_once_agent_ready(
    scenario_with_conflicted_merge_run: Scenario,
) -> None:
    scenario = scenario_with_conflicted_merge_run
    run = await _load_blocked_merge_run(scenario)
    assert run.blocked_task_id is not None
    assert run.blocked_branch_name is not None
    assert run.blocked_worktree_path is not None

    async with scenario.db.session() as session:
        agent = AgentSession(
            task_id=run.blocked_task_id,
            status=AgentStatus.Running,
            agent_capabilities={
                "can_send_text": True,
                "can_detect_ready_for_input": True,
                "can_detect_turn_complete": True,
            },
            agent_semantic_status={"turn_state": AgentTurnState.Ready},
        )
        session.add(agent)
        await session.commit()

    sender = _FakeSender(sent=[])
    resumer = _FakeResumer(resumed=[])
    supervisor = MergeConflictAssistSupervisor(
        sender=sender,
        resumer=resumer,
        delivery_timeout=timedelta(seconds=30),
        overall_timeout=timedelta(minutes=5),
    )

    now = datetime.now(UTC)
    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now)
        await session.commit()

    assert sender.sent
    _, message = sender.sent[-1]
    assert "rebase conflict" in message.lower()
    assert run.blocked_branch_name in message
    assert run.blocked_worktree_path in message
    assert not resumer.resumed


@pytest.mark.integration
async def test_conflict_assist_requires_turn_complete_after_delivery(
    scenario_with_conflicted_merge_run: Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = scenario_with_conflicted_merge_run
    run = await _load_blocked_merge_run(scenario)
    assert run.blocked_task_id is not None

    monkeypatch.setattr(
        conflict_assist,
        "_worktree_is_clean_for_resume",
        lambda _: (True, None),
    )

    sender = _FakeSender(sent=[])
    resumer = _FakeResumer(resumed=[])
    supervisor = MergeConflictAssistSupervisor(
        sender=sender,
        resumer=resumer,
        delivery_timeout=timedelta(seconds=30),
        overall_timeout=timedelta(minutes=5),
    )

    async with scenario.db.session() as session:
        agent = AgentSession(
            task_id=run.blocked_task_id,
            status=AgentStatus.Running,
            agent_capabilities={
                "can_send_text": True,
                "can_detect_ready_for_input": True,
                "can_detect_turn_complete": True,
            },
            agent_semantic_status={"turn_state": AgentTurnState.Ready},
        )
        session.add(agent)
        await session.commit()
        await session.refresh(agent)

        # Existing turn-complete event should not satisfy the post-delivery gate.
        session.add(
            Event(
                event_type="agent.turn_completed",
                data={"task_id": run.blocked_task_id, "agent_session_id": agent.id},
            )
        )
        await session.commit()

    now = datetime.now(UTC)
    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now)
        await session.commit()

    assert sender.sent

    async with scenario.db.session() as session:
        run_row = await session.get(MergeRun, run.id)
        assert run_row is not None
        run_row.status = MergeRunStatus.Resumable
        await session.commit()

    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now + timedelta(seconds=1))
        await session.commit()

    # Still no resume: no new post-delivery turn completion.
    assert not resumer.resumed

    async with scenario.db.session() as session:
        agent = await session.scalar(
            select(AgentSession)
            .where(AgentSession.task_id == run.blocked_task_id)
            .order_by(AgentSession.id.desc())
            .limit(1)
        )
        assert agent is not None
        session.add(
            Event(
                event_type="agent.turn_completed",
                data={"task_id": run.blocked_task_id, "agent_session_id": agent.id},
            )
        )
        await session.commit()

    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now + timedelta(seconds=2))
        await session.commit()

    assert resumer.resumed == [run.run_id]


@pytest.mark.integration
async def test_conflict_assist_times_out_if_agent_never_ready(
    scenario_with_conflicted_merge_run: Scenario,
) -> None:
    scenario = scenario_with_conflicted_merge_run
    run = await _load_blocked_merge_run(scenario)
    assert run.blocked_task_id is not None

    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=run.blocked_task_id,
                status=AgentStatus.Running,
                agent_capabilities={
                    "can_send_text": True,
                    "can_detect_ready_for_input": True,
                    "can_detect_turn_complete": True,
                },
                agent_semantic_status={"turn_state": AgentTurnState.Busy},
            )
        )
        await session.commit()

    sender = _FakeSender(sent=[])
    resumer = _FakeResumer(resumed=[])
    supervisor = MergeConflictAssistSupervisor(
        sender=sender,
        resumer=resumer,
        delivery_timeout=timedelta(seconds=1),
        overall_timeout=timedelta(minutes=5),
    )

    now = datetime.now(UTC)
    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now)
        await session.commit()
        await supervisor.tick(session, now=now + timedelta(seconds=2))
        await session.commit()

    snapshot = supervisor.snapshot(run_id=run.run_id)
    assert snapshot is not None
    assert snapshot.state == "timed_out"
    assert not sender.sent
    assert not resumer.resumed
