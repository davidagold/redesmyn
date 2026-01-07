from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from typing import Any, cast

import pytest
from sqlalchemy import func, select

from redesmyn.agent_driver import TmuxSessionSnapshot, TmuxSupervisor, supervise_once
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
from redesmyn.db.models import AttachExternal, AttachTmux
from redesmyn.domain.enums import (
    AgentKind,
    AgentInterfaceMode,
    AgentStatus,
    AgentTurnState,
)
from redesmyn.host_identity import HostIdentity, host_identity_path
from redesmyn.orchestrator import init_repo
from redesmyn.settings import RedesmynSettings
from redesmyn.ws_runtime import JsonWebSocketHub

from tests.scenarios.scenario import Scenario, ScenarioApp, ScenarioRepo
from tests.helpers.ws import JsonQueueWebSocket


@dataclass(slots=True)
class _FakeTmux(TmuxSupervisor):
    sessions: set[str]
    pipe_calls: list[tuple[str, str]]
    snapshot_authoritative: bool = True
    snapshot_error: str | None = None

    def list_sessions(self, *, timeout_s: float) -> TmuxSessionSnapshot:
        _ = timeout_s
        return TmuxSessionSnapshot(
            sessions=set(self.sessions),
            authoritative=self.snapshot_authoritative,
            error=self.snapshot_error,
        )

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
    _external_session_ref: (
        ExternalSessionNone | ExternalSessionCodex | ExternalSessionClaude
    )
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
            select(Repository).where(
                Repository.repo_root == str(scenario.ctx.repo_root)
            )
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
    driver_started_at = datetime.now(UTC)

    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            driver_started_at=driver_started_at,
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
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    driver_started_at = datetime.now(UTC)
    async with scenario.db.session() as session:
        row = AgentSession(
            task_id=task_id,
            status=AgentStatus.Running,
            attach=AttachTmux(
                session=tmux_name, socket_path=None, log_path=None
            ).model_dump(mode="python"),
        )
        session.add(row)
        await session.commit()

    fake_tmux = _FakeTmux(sessions=set(), pipe_calls=[])
    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            driver_started_at=driver_started_at,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.status == AgentStatus.Error
        assert row.ended_at is not None


@pytest.mark.integration
async def test_agent_driver_non_tmux_session_is_not_auto_ended(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    driver_started_at = datetime.now(UTC)
    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=task_id,
                status=AgentStatus.Running,
                attach=AttachExternal(hint="programmatic", log_path=None).model_dump(
                    mode="python"
                ),
            )
        )
        await session.commit()

    fake_tmux = _FakeTmux(sessions=set(), pipe_calls=[])
    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            driver_started_at=driver_started_at,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.status == AgentStatus.Running
        assert row.ended_at is None


@pytest.mark.integration
async def test_agent_driver_cursoring_and_edge_triggered_semantics(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    fake_tmux = _FakeTmux(sessions={tmux_name}, pipe_calls=[])
    driver_started_at = datetime.now(UTC)

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
            driver_started_at=driver_started_at,
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
            driver_started_at=driver_started_at,
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
            driver_started_at=driver_started_at,
        )

    assert backend.consume_calls == ["", "READY\n", "READY\n"]


@pytest.mark.integration
async def test_agent_driver_external_log_tailing_updates_semantics(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    log_path = scenario.ctx.state_dir / "external" / "session.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("READY\n", encoding="utf-8")
    driver_started_at = datetime.now(UTC)

    backend = _FakeBackend(
        _capabilities=AgentCapabilities(
            can_send_text=True, can_detect_ready_for_input=True
        ),
        _semantic_status=AgentSemanticStatus(),
        _external_session_ref=ExternalSessionNone(),
        consume_calls=[],
    )

    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=task_id,
                status=AgentStatus.Running,
                attach=AttachExternal(
                    hint="programmatic",
                    log_path=str(log_path),
                ).model_dump(mode="python"),
            )
        )
        await session.commit()

    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=_FakeTmux(sessions=set(), pipe_calls=[]),
            backend_factory=lambda *, agent_session: backend,
            driver_started_at=driver_started_at,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.agent_semantic_status["turn_state"] == AgentTurnState.Ready


@pytest.mark.integration
async def test_agent_driver_tmux_list_failure_does_not_end_tmux_sessions(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    driver_started_at = datetime.now(UTC)
    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=task_id,
                status=AgentStatus.Running,
                attach=AttachTmux(
                    session=tmux_name, socket_path=None, log_path=None
                ).model_dump(mode="python"),
            )
        )
        await session.commit()

    fake_tmux = _FakeTmux(
        sessions=set(),
        pipe_calls=[],
        snapshot_authoritative=False,
        snapshot_error="tmux timeout",
    )
    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            driver_started_at=driver_started_at,
        )

    async with scenario.db.session() as session:
        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.status == AgentStatus.Running
        assert row.ended_at is None


@pytest.mark.integration
async def test_agent_driver_persists_external_session_ref_without_thrashing(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    fake_tmux = _FakeTmux(sessions={tmux_name}, pipe_calls=[])
    driver_started_at = datetime.now(UTC)

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
            driver_started_at=driver_started_at,
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
            driver_started_at=driver_started_at,
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
            driver_started_at=driver_started_at,
        )

    async with scenario.db.session() as session:
        after_events = list(
            await session.scalars(
                select(Event.id).where(Event.event_type == "task.agent_session_update")
            )
        )
    assert after_events == before_events


@pytest.mark.integration
async def test_agent_driver_reuses_active_session_row_when_tmux_running(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    tmux_name = tmux_session_name_for_task(task_id=task_id)

    async with scenario.db.session() as session:
        # Simulate a stale/inconsistent DB row: stopped but still "active"
        # according to the `ended_at IS NULL` uniqueness constraint.
        agent_session = AgentSession(task_id=task_id, status=AgentStatus.Stopped)
        session.add(agent_session)
        await session.commit()
        await session.refresh(agent_session)
        session_id = agent_session.id

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
        count = await session.scalar(
            select(func.count(AgentSession.id)).where(AgentSession.task_id == task_id)
        )
        assert count == 1
        row = await session.get(AgentSession, session_id)
        assert row is not None
        assert row.status == AgentStatus.Running
        assert row.ended_at is None
        assert row.attach.get("type") == "tmux"


@pytest.mark.integration
async def test_agent_driver_persists_semantic_events_and_preview_from_structured_codex_log(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    log_path = scenario.ctx.state_dir / "external" / "codex-structured.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    driver_started_at = datetime.now(UTC)

    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=task_id,
                status=AgentStatus.Running,
                agent_kind=AgentKind.Codex,
                agent_interface_mode=AgentInterfaceMode.Structured,
                attach=AttachExternal(
                    hint="structured",
                    log_path=str(log_path),
                ).model_dump(mode="python"),
            )
        )
        await session.commit()

    event_hub = JsonWebSocketHub()
    websocket = JsonQueueWebSocket()
    send_queue = await event_hub.connect(websocket)  # type: ignore[arg-type]

    runtime: dict[int, object] = {}
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=_FakeTmux(sessions=set(), pipe_calls=[]),
            event_hub=event_hub,
            driver_started_at=driver_started_at,
        )

    log_path.write_text(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"th_123"}',
                '{"type":"turn.started","turn_id":"tu_1"}',
                '{"type":"assistant.message","role":"assistant","text":"Hello world"}',
                '{"type":"turn.completed","turn_id":"tu_1"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=_FakeTmux(sessions=set(), pipe_calls=[]),
            event_hub=event_hub,
            driver_started_at=driver_started_at,
        )

    async with scenario.db.session() as session:
        events = list(
            await session.scalars(
                select(Event.event_type).where(
                    Event.event_type.in_(
                        [
                            "agent.turn_started",
                            "agent.assistant_message",
                            "agent.turn_completed",
                        ]
                    )
                )
            )
        )
        assert "agent.turn_started" in events
        assert "agent.assistant_message" in events
        assert "agent.turn_completed" in events

        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.agent_preview.get("last_assistant_message_preview") == "Hello world"
        assert row.agent_preview.get("last_message_turn_id") == "tu_1"
        assert row.agent_preview.get("last_assistant_message_at") is not None

    drained: list[dict[str, object]] = []
    while not send_queue.empty():
        drained.append(send_queue.get_nowait())
    found = False
    for payload in drained:
        if payload.get("type") != "event":
            continue
        event = payload.get("event")
        if not isinstance(event, dict):
            continue
        event_dict = cast(dict[str, Any], event)
        if event_dict.get("eventType") == "agent.assistant_message":
            found = True
            break
    assert found


@pytest.mark.integration
async def test_agent_driver_drains_structured_log_when_tmux_exits(
    scenario: Scenario,
) -> None:
    task_id = await _seed_task(scenario)
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    log_path = scenario.ctx.state_dir / "external" / "codex-tmux-exit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                '{"type":"thread.started","thread_id":"th_123"}',
                '{"type":"turn.started","turn_id":"tu_1"}',
                '{"type":"assistant.message","role":"assistant","text":"Hello world"}',
                '{"type":"turn.completed","turn_id":"tu_1"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    async with scenario.db.session() as session:
        session.add(
            AgentSession(
                task_id=task_id,
                status=AgentStatus.Running,
                agent_kind=AgentKind.Codex,
                agent_interface_mode=AgentInterfaceMode.Structured,
                attach=AttachTmux(
                    session=tmux_name, socket_path=None, log_path=str(log_path)
                ).model_dump(mode="python"),
            )
        )
        await session.commit()

    fake_tmux = _FakeTmux(sessions=set(), pipe_calls=[])
    runtime: dict[int, object] = {}
    driver_started_at = datetime.now(UTC)
    async with scenario.db.session() as session:
        await supervise_once(
            scenario.ctx,
            session,
            runtime_by_session_id=runtime,  # type: ignore[arg-type]
            tmux=fake_tmux,
            driver_started_at=driver_started_at,
        )

    async with scenario.db.session() as session:
        events = list(
            await session.scalars(
                select(Event.event_type).where(
                    Event.event_type.in_(
                        [
                            "agent.turn_started",
                            "agent.assistant_message",
                            "agent.turn_completed",
                            "task.agent_session_update",
                        ]
                    )
                )
            )
        )
        assert "agent.turn_started" in events
        assert "agent.assistant_message" in events
        assert "agent.turn_completed" in events

        row = await session.scalar(
            select(AgentSession).where(AgentSession.task_id == task_id)
        )
        assert row is not None
        assert row.status == AgentStatus.Stopped
        assert row.ended_at is not None
        assert row.agent_preview.get("last_assistant_message_preview") == "Hello world"


@pytest.mark.integration
async def test_agent_driver_kill_switch_disables_background_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = ScenarioRepo.init(tmp_path)
    ctx = build_repo_context(
        repo_root=repo.repo_root, worktree_root=repo.worktrees_root
    )
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
            for task in scenario_app.app.state.background_tasks.tasks
        )
    finally:
        await scenario_app.aclose()
