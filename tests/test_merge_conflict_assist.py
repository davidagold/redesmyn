from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

import redesmyn.merge_conflict_assist as conflict_assist
from redesmyn.db import AgentSession, Event, MergeRun
from redesmyn.domain.enums import AgentInterfaceMode, AgentStatus, MergeRunStatus
from redesmyn.merge_conflict_assist import MergeConflictAssistSupervisor

from tests.scenarios.scenario import Scenario


@dataclass(slots=True)
class _FakeLauncher:
    launched: list[dict[str, object]]
    agent_session_id: int
    error: Exception | None = None

    async def launch(
        self,
        *,
        task_id: int,
        prompt: str,
        detach: bool,
        idempotency_key: str,
    ) -> int:
        self.launched.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "detach": detach,
                "idempotency_key": idempotency_key,
            }
        )
        if self.error is not None:
            raise self.error
        return self.agent_session_id


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
async def test_conflict_assist_launches_structured_continuation_turn(
    scenario_with_conflicted_merge_run: Scenario,
) -> None:
    scenario = scenario_with_conflicted_merge_run
    run = await _load_blocked_merge_run(scenario)
    assert run.blocked_task_id is not None
    assert run.blocked_branch_name is not None
    assert run.blocked_worktree_path is not None

    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=run.blocked_task_id,
                status=AgentStatus.Stopped,
                agent_interface_mode=AgentInterfaceMode.Structured,
                resolved_launch_configuration={"argv": ["codex", "exec", "--json"]},
                external_session_ref={"type": "codex_thread", "thread_id": "th_123"},
            )
        )
        await session.commit()

    launcher = _FakeLauncher(launched=[], agent_session_id=4242)
    resumer = _FakeResumer(resumed=[])
    supervisor = MergeConflictAssistSupervisor(
        launcher=launcher,
        resumer=resumer,
        delivery_timeout=timedelta(seconds=30),
        overall_timeout=timedelta(minutes=5),
    )

    now = datetime.now(UTC)
    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now)
        await session.commit()

    assert launcher.launched
    launched = launcher.launched[-1]
    message = str(launched["prompt"])
    assert "rebase conflict" in message.lower()
    assert run.blocked_branch_name in message
    assert run.blocked_worktree_path in message
    assert launched["detach"] is True
    assert launched["idempotency_key"] == f"merge_conflict_assist:{run.run_id}"
    assert not resumer.resumed


@pytest.mark.integration
async def test_conflict_assist_requires_turn_complete_for_continuation_session(
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

    launcher = _FakeLauncher(launched=[], agent_session_id=4242)
    resumer = _FakeResumer(resumed=[])
    supervisor = MergeConflictAssistSupervisor(
        launcher=launcher,
        resumer=resumer,
        delivery_timeout=timedelta(seconds=30),
        overall_timeout=timedelta(minutes=5),
    )

    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=run.blocked_task_id,
                status=AgentStatus.Stopped,
                agent_interface_mode=AgentInterfaceMode.Structured,
                resolved_launch_configuration={"argv": ["codex", "exec", "--json"]},
                external_session_ref={"type": "codex_thread", "thread_id": "th_123"},
            )
        )
        await session.commit()

    now = datetime.now(UTC)
    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now)
        await session.commit()

    assert launcher.launched

    async with scenario.db.session() as session:
        run_row = await session.get(MergeRun, run.id)
        assert run_row is not None
        run_row.status = MergeRunStatus.Resumable
        await session.commit()

    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now + timedelta(seconds=1))
        await session.commit()

    # Still no resume: continuation turn has not completed.
    assert not resumer.resumed

    async with scenario.db.session() as session:
        session.add(
            Event(
                event_type="agent.turn_completed",
                data={"task_id": run.blocked_task_id, "agent_session_id": 4242},
            )
        )
        await session.commit()

    async with scenario.db.session() as session:
        await supervisor.tick(session, now=now + timedelta(seconds=2))
        await session.commit()

    assert resumer.resumed == [run.run_id]


@pytest.mark.integration
async def test_conflict_assist_times_out_if_turn_never_launches(
    scenario_with_conflicted_merge_run: Scenario,
) -> None:
    scenario = scenario_with_conflicted_merge_run
    run = await _load_blocked_merge_run(scenario)
    assert run.blocked_task_id is not None

    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=run.blocked_task_id,
                status=AgentStatus.Stopped,
                agent_interface_mode=AgentInterfaceMode.Structured,
                resolved_launch_configuration={"argv": ["codex", "exec", "--json"]},
                external_session_ref={"type": "codex_thread", "thread_id": "th_123"},
            )
        )
        await session.commit()

    launcher = _FakeLauncher(
        launched=[],
        agent_session_id=4242,
        error=RuntimeError("still running"),
    )
    resumer = _FakeResumer(resumed=[])
    supervisor = MergeConflictAssistSupervisor(
        launcher=launcher,
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
    assert launcher.launched
    assert not resumer.resumed
