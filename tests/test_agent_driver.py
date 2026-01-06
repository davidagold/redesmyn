from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from redesmyn.agent_driver import TmuxSupervisor, supervise_once
from redesmyn.agent_interface.v0 import (
    AgentCapabilities,
    AgentEvent,
    AgentSemanticStatus,
    ExternalSessionClaude,
    ExternalSessionCodex,
    ExternalSessionNone,
)
from redesmyn.api import create_app
from redesmyn.agent_runtime import tmux_session_name_for_task
from redesmyn.context import build_repo_context
from redesmyn.db import AgentSession, Epic, Event, Repository, Task
from redesmyn.domain.enums import AgentStatus, AgentTurnState
from redesmyn.host_identity import HostIdentity, host_identity_path
from redesmyn.orchestrator import init_repo
from redesmyn.settings import RedesmynSettings

from tests.scenarios.scenario import Scenario, ScenarioApp, ScenarioRepo


@dataclass(slots=True)
class _FakeTmux(TmuxSupervisor):
    sessions: set[str]
    pipe_calls: list[tuple[str, str]]

    def list_sessions(self, *, timeout_s: float) -> set[str]:
        _ = timeout_s
        return set(self.sessions)

    def pipe_pane_to_log(
        self, *, session_name: str, log_path: str, timeout_s: float
    ) -> None:
        _ = timeout_s
        self.pipe_calls.append((session_name, log_path))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).touch(exist_ok=True)


@dataclass(slots=True)
class _FakeBackend:
    _capabilities: AgentCapabilities
    _semantic_status: AgentSemanticStatus
    _external_session_ref: ExternalSessionNone | ExternalSessionCodex | ExternalSessionClaude
    consume_calls: list[str]

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    @property
    def semantic_status(self) -> AgentSemanticStatus:
        return self._semantic_status

    @property
    def external_session_ref(
        self,
    ) -> ExternalSessionNone | ExternalSessionCodex | ExternalSessionClaude:
        return self._external_session_ref

    def consume_output(self, text: str) -> list[AgentEvent]:
        self.consume_calls.append(text)
        if "READY" in text:
            self._semantic_status = AgentSemanticStatus(turn_state=AgentTurnState.Ready)
        if "BUSY" in text:
            self._semantic_status = AgentSemanticStatus(turn_state=AgentTurnState.Busy)
        if "CODEX_THREAD=" in text:
            thread_id = text.split("CODEX_THREAD=", 1)[1].split()[0].strip()
            self._external_session_ref = ExternalSessionCodex(thread_id=thread_id)
        if "CLAUDE_SESSION=" in text:
            session_id = text.split("CLAUDE_SESSION=", 1)[1].split()[0].strip()
            self._external_session_ref = ExternalSessionClaude(session_id=session_id)
        return []


async def _seed_task(scenario: Scenario) -> int:
    async with scenario.db.session() as session:
        repo_row = await session.scalar(
            select(Repository).where(Repository.repo_root == str(scenario.ctx.repo_root))
        )
        if repo_row is None:
            raise RuntimeError("Scenario repository row missing")

        epic = Epic(
            repository_id=repo_row.id,
            name="Test Epic",
            slug=f"test-epic-{uuid4().hex[:8]}",
            root_branch="main",
        )
        session.add(epic)
        await session.flush()

        task = Task(
            epic_id=epic.id,
            title="Task",
        )
        session.add(task)
        await session.commit()
        return task.id


@pytest.mark.integration
async def test_agent_driver_creates_session_and_pipes_log(scenario: Scenario) -> None:
    task_id = await _seed_task(scenario)
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    fake_tmux = _FakeTmux(sessions={tmux_name}, pipe_calls=[])

    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.status == AgentStatus.Running
        assert row.started_at is not None
        assert row.ended_at is None
        assert row.attach["type"] == "tmux"
        assert row.attach["session"] == tmux_name
        assert isinstance(row.attach.get("log_path"), str)

        event = await session.scalar(
            select(Event)
            .where(Event.event_type == "task.agent_session_update")
            .order_by(Event.id.desc())
            .limit(1)
        )
        assert event is not None
        assert event.data.get("task_id") == task_id
        assert event.data.get("agent_session_id") == row.id

    assert fake_tmux.pipe_calls
    assert fake_tmux.pipe_calls[-1][0] == tmux_name


@pytest.mark.integration
async def test_agent_driver_marks_session_error_when_tmux_disappears(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    async with scenario.db.session() as session:
        session.add(AgentSession(task_id=task_id, status=AgentStatus.Running))
        await session.commit()

    fake_tmux = _FakeTmux(sessions=set(), pipe_calls=[])
    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.status == AgentStatus.Error
        assert row.ended_at is not None


@pytest.mark.integration
async def test_agent_driver_cursoring_and_edge_triggered_semantics(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    fake_tmux = _FakeTmux(sessions={tmux_name}, pipe_calls=[])

    backend = _FakeBackend(
        _capabilities=AgentCapabilities(
            can_send_text=True, can_detect_ready_for_input=True
        ),
        _semantic_status=AgentSemanticStatus(),
        _external_session_ref=ExternalSessionNone(),
        consume_calls=[],
    )

    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            backend_factory=lambda *, agent_session: backend,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        log_path = Path(row.attach["log_path"])

    log_path.write_text("READY\n", encoding="utf-8")
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            backend_factory=lambda *, agent_session: backend,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.agent_semantic_status["turn_state"] == AgentTurnState.Ready

        event_count = await session.scalar(
            select(func.count())
            .select_from(Event)
            .where(Event.event_type == "task.agent_session_update")
        )
        assert isinstance(event_count, int)

    log_path.write_text("READY\nREADY\n", encoding="utf-8")
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            backend_factory=lambda *, agent_session: backend,
        )

    assert backend.consume_calls == ["READY\n", "READY\n"]


@pytest.mark.integration
async def test_agent_driver_persists_external_session_ref_without_thrashing(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    fake_tmux = _FakeTmux(sessions={tmux_name}, pipe_calls=[])

    backend = _FakeBackend(
        _capabilities=AgentCapabilities(can_send_text=True, can_resume_by_id=True),
        _semantic_status=AgentSemanticStatus(),
        _external_session_ref=ExternalSessionNone(),
        consume_calls=[],
    )

    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            backend_factory=lambda *, agent_session: backend,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        log_path = Path(row.attach["log_path"])

    log_path.write_text("CODEX_THREAD=th_123\n", encoding="utf-8")
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            backend_factory=lambda *, agent_session: backend,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.external_session_ref["type"] == "codex_thread"
        assert row.external_session_ref["thread_id"] == "th_123"

        before_events = list(
            await session.scalars(
                select(Event.id).where(Event.event_type == "task.agent_session_update")
            )
        )

    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            backend_factory=lambda *, agent_session: backend,
        )

    async with scenario.db.session() as session:
        after_events = list(
            await session.scalars(
                select(Event.id).where(Event.event_type == "task.agent_session_update")
            )
        )
    assert after_events == before_events


@pytest.mark.integration
async def test_agent_driver_kill_switch_disables_background_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = ScenarioRepo.init(tmp_path)
    ctx = build_repo_context(repo_root=repo.repo_root, worktree_root=repo.worktrees_root)
    ctx.state_dir.mkdir(parents=True, exist_ok=True)
    host_identity_path(ctx).write_text(
        HostIdentity(host_key="test-host-key", display_name="Tests").model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )
    await init_repo(ctx, migrate=True)

    monkeypatch.setenv("REDESMYN_NO_AGENT_MONITOR", "1")
    app = create_app(
        settings=RedesmynSettings(
            repo_root=ctx.repo_root,
            worktree_root=ctx.worktree_root,
            db_path=ctx.db_path,
            runner_mode="local",
            enable_repo_observer=False,
            enable_agent_monitor=True,
        )
    )

    scenario_app = await ScenarioApp.open(app)
    try:
        assert not any(
            task.get_name() == "agent_driver"
            for task in scenario_app.app.state.background_tasks
        )
    finally:
        await scenario_app.aclose()
