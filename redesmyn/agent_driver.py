from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import structlog
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.agent_kind import resolve_agent_backend
from redesmyn.agent_interface.v0 import (
    AgentAssistantMessageEvent,
    AgentBackend,
    AgentEvent,
    AgentTurnCompletedEvent,
    AgentTurnStartedEvent,
    ExternalSessionClaude,
    ExternalSessionCodex,
    ExternalSessionNone,
    ExternalSessionRef,
)
from redesmyn.agent_runtime import (
    agent_session_exit_code_path,
    agent_session_log_path,
    has_tmux,
    tmux_session_name_for_task,
)
from redesmyn.context import RepoContext
from redesmyn.db import AgentSession, Event, Task
from redesmyn.db.models import (
    AgentPreview,
    AttachExternal,
    AttachInfo,
    AttachNone,
    AttachTmux,
)
from redesmyn.domain.enums import (
    AgentInterfaceMode,
    AgentSessionRuntimeKind,
    AgentStatus,
)
from redesmyn.schemas.core import EventResponse
from redesmyn.ws_runtime import JsonWebSocketHub

log = structlog.get_logger("redesmyn.agent_driver")


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _read_exit_code(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return int(raw.splitlines()[0].strip())
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class TmuxSessionSnapshot:
    sessions: set[str]
    authoritative: bool
    error: str | None = None


class TmuxSupervisor(Protocol):
    def list_sessions(self, *, timeout_s: float) -> TmuxSessionSnapshot: ...

    def pipe_pane_to_log(
        self,
        *,
        session_name: str,
        log_path: str,
        timeout_s: float,
    ) -> None: ...


class SubprocessTmuxSupervisor:
    def list_sessions(self, *, timeout_s: float) -> TmuxSessionSnapshot:
        proc = subprocess.run(
            ["tmux", "list-sessions", "-F", "#S"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            if "no server running" in stderr or "failed to connect to server" in stderr:
                return TmuxSessionSnapshot(sessions=set(), authoritative=True)
            return TmuxSessionSnapshot(
                sessions=set(),
                authoritative=False,
                error=stderr or f"tmux list-sessions exited {proc.returncode}",
            )
        return TmuxSessionSnapshot(
            sessions={
                line.strip() for line in proc.stdout.splitlines() if line.strip()
            },
            authoritative=True,
        )

    def pipe_pane_to_log(
        self,
        *,
        session_name: str,
        log_path: str,
        timeout_s: float,
    ) -> None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = f"cat >> {shlex.quote(log_path)}"
        subprocess.run(
            ["tmux", "pipe-pane", "-t", session_name, "-o", cmd],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )


class AgentBackendFactory(Protocol):
    def __call__(self, *, agent_session: AgentSession) -> AgentBackend: ...


def _default_backend_factory(*, agent_session: AgentSession) -> AgentBackend:
    return resolve_agent_backend(agent_session=agent_session)


def _task_id_from_tmux_session(name: str) -> int | None:
    prefix = os.environ.get("REDESMYN_TMUX_SESSION_PREFIX", "rn-a").strip()
    if not prefix:
        prefix = "rn-a"
    marker = prefix + "-"
    if not name.startswith(marker):
        return None
    suffix = name.removeprefix(marker)
    if not suffix.isdigit():
        return None
    return int(suffix)


def _read_log_incremental(
    *,
    log_path: Path,
    cursor: int,
    max_read_bytes: int,
) -> tuple[str | None, int]:
    try:
        with log_path.open("rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if cursor > size:
                    cursor = size
                f.seek(cursor, os.SEEK_SET)
            except OSError:
                cursor = 0
            data = f.read(max_read_bytes)
            cursor = f.tell()
    except OSError:
        return None, cursor
    if not data:
        return None, cursor
    return data.decode("utf-8", errors="replace"), cursor


@dataclass(slots=True)
class _SessionRuntime:
    backend: AgentBackend
    cursor: int
    skip_persist_until_log_activity: bool = False
    recent_event_keys: deque[str] = field(default_factory=lambda: deque(maxlen=256))


def _log_file_size(log_path: Path) -> int:
    try:
        return log_path.stat().st_size
    except OSError:
        return 0


def _should_skip_log_history(
    *, agent_session: AgentSession, driver_started_at: datetime
) -> bool:
    started_at = agent_session.started_at
    if started_at is None:
        return False
    return _as_utc(started_at) < _as_utc(driver_started_at)


def _runtime_kind_from_attach(attach: AttachInfo) -> AgentSessionRuntimeKind:
    match attach:
        case AttachTmux():
            return AgentSessionRuntimeKind.Tmux
        case AttachExternal():
            return AgentSessionRuntimeKind.External
        case AttachNone():
            return AgentSessionRuntimeKind.None_


def _log_path_from_attach(attach: AttachInfo) -> Path | None:
    match attach:
        case AttachTmux(log_path=log_path) | AttachExternal(log_path=log_path):
            return Path(log_path) if log_path else None
        case AttachNone():
            return None


def _normalize_preview_text(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


async def _emit_event(
    session: AsyncSession,
    pending: list[Event],
    *,
    event_type: str,
    data: dict[str, Any],
) -> None:
    row = Event(event_type=event_type, data=data)
    session.add(row)
    await session.flush()
    await session.refresh(row)
    pending.append(row)


def _session_update_event_payload(*, agent_session: AgentSession) -> dict[str, Any]:
    attach_adapter = TypeAdapter(AttachInfo)
    try:
        attach = attach_adapter.validate_python(agent_session.attach)
    except Exception:
        attach = AttachNone()
    return {
        "task_id": agent_session.task_id,
        "agent_session_id": agent_session.id,
        "agent_status": agent_session.status,
        "started_at": (
            agent_session.started_at.isoformat()
            if agent_session.started_at is not None
            else None
        ),
        "ended_at": (
            agent_session.ended_at.isoformat()
            if agent_session.ended_at is not None
            else None
        ),
        "attach": agent_session.attach,
        "runtime_kind": _runtime_kind_from_attach(attach),
        "agent_kind_selection": agent_session.agent_kind_selection,
        "agent_kind": agent_session.agent_kind,
        "agent_interface_mode": agent_session.agent_interface_mode,
        "agent_capabilities": agent_session.agent_capabilities,
        "agent_semantic_status": agent_session.agent_semantic_status,
        "external_session_ref": agent_session.external_session_ref,
        "agent_preview": agent_session.agent_preview,
    }


async def supervise_once(
    ctx: RepoContext,
    session: AsyncSession,
    runtime_by_session_id: dict[int, _SessionRuntime],
    *,
    backend_factory: AgentBackendFactory = _default_backend_factory,
    event_hub: JsonWebSocketHub | None = None,
    tmux: TmuxSupervisor | None = None,
    driver_started_at: datetime,
    timeout_s: float = 1.0,
    max_read_bytes: int = 256 * 1024,
) -> int:
    now = datetime.now(UTC)
    pending_events: list[Event] = []
    attach_adapter = TypeAdapter(AttachInfo)
    preview_adapter = TypeAdapter(AgentPreview)

    tmux_snapshot: TmuxSessionSnapshot | None = None
    if tmux is None and has_tmux():
        tmux = SubprocessTmuxSupervisor()

    if tmux is not None:
        try:
            tmux_snapshot = tmux.list_sessions(timeout_s=timeout_s)
        except Exception as exc:
            tmux_snapshot = TmuxSessionSnapshot(
                sessions=set(),
                authoritative=False,
                error=str(exc),
            )
        if tmux_snapshot.authoritative is False:
            log.warning("tmux.list_sessions.failed", error=tmux_snapshot.error)

    tmux_sessions: set[str] = set()
    running_task_ids: set[int] = set()
    if tmux_snapshot is not None and tmux_snapshot.authoritative:
        tmux_sessions = tmux_snapshot.sessions
        running_task_ids = {
            task_id
            for name in tmux_sessions
            if (task_id := _task_id_from_tmux_session(name)) is not None
        }

    tasks_by_id: dict[int, Task] = {}
    if running_task_ids:
        tasks = list(
            await session.scalars(select(Task).where(Task.id.in_(running_task_ids)))
        )
        tasks_by_id = {t.id: t for t in tasks}

    # "Active" sessions are defined by `ended_at IS NULL`. We intentionally do
    # not filter by `status` here because we may need to reconcile/heal
    # inconsistent rows (e.g. stopped sessions with `ended_at IS NULL`) to avoid
    # violating the `uq_agent_sessions_active_task_id` uniqueness constraint.
    active_sessions = list(
        await session.scalars(
            select(AgentSession)
            .where(AgentSession.ended_at.is_(None))
            .order_by(desc(AgentSession.id))
        )
    )
    active_session_by_task_id: dict[int, AgentSession] = {}
    for row in active_sessions:
        if row.task_id in active_session_by_task_id:
            continue
        active_session_by_task_id[row.task_id] = row

    updated = 0

    async def emit_event(event_type: str, data: dict[str, Any]) -> None:
        nonlocal updated
        await _emit_event(
            session,
            pending_events,
            event_type=event_type,
            data=data,
        )
        updated += 1

    async def flush_session_update(agent_session: AgentSession) -> None:
        nonlocal updated
        await emit_event(
            "task.agent_session_update",
            _session_update_event_payload(agent_session=agent_session),
        )

    def external_ref_key(ref: ExternalSessionRef) -> str:
        match ref:
            case ExternalSessionCodex(thread_id=thread_id, turn_id=turn_id):
                return f"codex:{thread_id}:{turn_id or ''}"
            case ExternalSessionClaude(session_id=session_id):
                return f"claude:{session_id}"
            case ExternalSessionNone():
                return "none"

    def agent_event_key(
        *, event_type: str, ref: ExternalSessionRef, text: str | None
    ) -> str:
        base = f"{event_type}:{external_ref_key(ref)}"
        if text is None:
            return base
        normalized = _normalize_preview_text(text)
        return f"{base}:{hash(normalized)}"

    async def persist_agent_event(
        *,
        agent_session: AgentSession,
        runtime: _SessionRuntime,
        event: AgentEvent,
        ref: ExternalSessionRef,
        now: datetime,
    ) -> None:
        match event:
            case AgentTurnStartedEvent():
                key = agent_event_key(event_type="turn_started", ref=ref, text=None)
                if key in runtime.recent_event_keys:
                    return
                runtime.recent_event_keys.append(key)
                await emit_event(
                    "agent.turn_started",
                    {
                        "task_id": agent_session.task_id,
                        "agent_session_id": agent_session.id,
                        "external_session_ref": ref.model_dump(mode="python"),
                    },
                )
                return
            case AgentTurnCompletedEvent():
                key = agent_event_key(event_type="turn_completed", ref=ref, text=None)
                if key in runtime.recent_event_keys:
                    return
                runtime.recent_event_keys.append(key)
                await emit_event(
                    "agent.turn_completed",
                    {
                        "task_id": agent_session.task_id,
                        "agent_session_id": agent_session.id,
                        "external_session_ref": ref.model_dump(mode="python"),
                    },
                )
                return
            case AgentAssistantMessageEvent():
                raw_text = event.text
                if not raw_text.strip():
                    return
                key = agent_event_key(
                    event_type="assistant_message", ref=ref, text=raw_text
                )
                if key in runtime.recent_event_keys:
                    return
                runtime.recent_event_keys.append(key)

                normalized = _normalize_preview_text(raw_text)
                preview = _truncate(normalized, max_chars=240)
                stored_text = _truncate(normalized, max_chars=4000)

                try:
                    current_preview = preview_adapter.validate_python(
                        agent_session.agent_preview
                    )
                except Exception:
                    current_preview = AgentPreview()

                next_turn_id: str | None = None
                match ref:
                    case ExternalSessionCodex(turn_id=turn_id):
                        next_turn_id = turn_id
                next_preview = current_preview.model_copy(
                    update={
                        "last_assistant_message_preview": preview,
                        "last_assistant_message_at": now,
                        "last_message_turn_id": next_turn_id,
                    }
                )
                next_preview_dump = next_preview.model_dump(mode="json")
                if agent_session.agent_preview != next_preview_dump:
                    agent_session.agent_preview = next_preview_dump
                    # Preview changes should update the session snapshot event too.
                    await flush_session_update(agent_session)

                await emit_event(
                    "agent.assistant_message",
                    {
                        "task_id": agent_session.task_id,
                        "agent_session_id": agent_session.id,
                        "text": stored_text,
                        "preview": preview,
                        "external_session_ref": ref.model_dump(mode="python"),
                    },
                )
                return
            case _:
                return

    async def process_log_text(
        *,
        agent_session: AgentSession,
        runtime: _SessionRuntime,
        log_text: str | None,
    ) -> None:
        # Tick the backend every supervise loop so interpreters can enforce
        # timeouts / degrade-to-unknown behavior even when the log is quiet.
        agent_events = runtime.backend.consume_output(log_text or "")

        # If we're attaching to a pre-existing session (e.g. after server
        # restart), we start tailing from the end of the log and defer
        # persisting updates until we see new log activity.
        if runtime.skip_persist_until_log_activity and not log_text:
            return
        runtime.skip_persist_until_log_activity = False

        # Persist semantic events derived from structured output.
        ref = runtime.backend.external_session_ref
        for agent_event in agent_events:
            await persist_agent_event(
                agent_session=agent_session,
                runtime=runtime,
                event=agent_event,
                ref=ref,
                now=now,
            )

        next_caps = runtime.backend.capabilities.model_dump(mode="python")
        next_semantic_status = runtime.backend.semantic_status.model_dump(mode="python")
        next_external_ref = runtime.backend.external_session_ref.model_dump(
            mode="python"
        )

        changed = False
        if agent_session.agent_capabilities != next_caps:
            agent_session.agent_capabilities = next_caps
            changed = True
        if agent_session.agent_semantic_status != next_semantic_status:
            agent_session.agent_semantic_status = next_semantic_status
            changed = True
        if agent_session.external_session_ref != next_external_ref:
            agent_session.external_session_ref = next_external_ref
            changed = True

        if changed:
            await flush_session_update(agent_session)

    for task_id in running_task_ids:
        if task_id not in tasks_by_id:
            continue
        tmux_name = tmux_session_name_for_task(task_id=task_id)
        if tmux_name not in tmux_sessions:
            continue

        agent_session = active_session_by_task_id.get(task_id)
        changed = False
        if agent_session is None:
            agent_session = AgentSession(
                task_id=task_id,
                status=AgentStatus.Running,
                started_at=now,
                ended_at=None,
            )
            session.add(agent_session)
            await session.flush()
            active_session_by_task_id[task_id] = agent_session
            changed = True
        else:
            if agent_session.status not in {AgentStatus.Running, AgentStatus.Blocked}:
                agent_session.status = AgentStatus.Running
                changed = True
            if agent_session.started_at is None:
                agent_session.started_at = now
                changed = True
            if agent_session.ended_at is not None:
                agent_session.ended_at = None
                changed = True

        try:
            existing_attach = attach_adapter.validate_python(agent_session.attach)
        except Exception:
            existing_attach = AttachNone()
        match existing_attach:
            case AttachTmux(log_path=existing_log_path):
                pass
            case _:
                existing_log_path = None
        log_path = existing_log_path or str(
            agent_session_log_path(ctx, task_id=task_id, session_id=agent_session.id)
        )
        attach = AttachTmux(
            session=tmux_name,
            socket_path=None,
            log_path=log_path,
        )
        attach_dict = attach.model_dump(mode="python")
        if agent_session.attach != attach_dict:
            agent_session.attach = attach_dict
            changed = True

        if tmux is not None:
            tmux.pipe_pane_to_log(
                session_name=tmux_name,
                log_path=log_path,
                timeout_s=timeout_s,
            )

        runtime = runtime_by_session_id.get(agent_session.id)
        if runtime is None:
            skip_history = _should_skip_log_history(
                agent_session=agent_session, driver_started_at=driver_started_at
            )
            runtime = _SessionRuntime(
                backend=backend_factory(agent_session=agent_session),
                cursor=_log_file_size(Path(log_path)) if skip_history else 0,
                skip_persist_until_log_activity=skip_history,
            )
            runtime_by_session_id[agent_session.id] = runtime

        if changed:
            await flush_session_update(agent_session)

    for task_id, agent_session in list(active_session_by_task_id.items()):
        if agent_session.status in {AgentStatus.Stopped, AgentStatus.Error}:
            agent_session.ended_at = now
            await flush_session_update(agent_session)
            runtime_by_session_id.pop(agent_session.id, None)
            continue

        try:
            attach = attach_adapter.validate_python(agent_session.attach)
        except Exception:
            attach = AttachNone()

        runtime_kind = _runtime_kind_from_attach(attach)
        if (
            runtime_kind == AgentSessionRuntimeKind.Tmux
            and tmux_snapshot is not None
            and tmux_snapshot.authoritative
        ):
            match attach:
                case AttachTmux(session=tmux_name, log_path=log_path):
                    pass
                case _:
                    tmux_name = tmux_session_name_for_task(task_id=task_id)
                    log_path = None
            if tmux_name not in tmux_sessions:
                if log_path:
                    drain_runtime = runtime_by_session_id.get(agent_session.id)
                    if drain_runtime is None:
                        skip_history = _should_skip_log_history(
                            agent_session=agent_session,
                            driver_started_at=driver_started_at,
                        )
                        drain_runtime = _SessionRuntime(
                            backend=backend_factory(agent_session=agent_session),
                            cursor=_log_file_size(Path(log_path))
                            if skip_history
                            else 0,
                            skip_persist_until_log_activity=skip_history,
                        )
                        runtime_by_session_id[agent_session.id] = drain_runtime

                    drain_log_path = Path(log_path)
                    while True:
                        drained_text, next_cursor = _read_log_incremental(
                            log_path=drain_log_path,
                            cursor=drain_runtime.cursor,
                            max_read_bytes=max_read_bytes,
                        )
                        drain_runtime.cursor = next_cursor
                        if drained_text is None:
                            break
                        await process_log_text(
                            agent_session=agent_session,
                            runtime=drain_runtime,
                            log_text=drained_text,
                        )

                exit_code: int | None = None
                exit_code_path = agent_session_exit_code_path(
                    ctx, task_id=task_id, session_id=agent_session.id
                )
                if exit_code_path.exists():
                    exit_code = _read_exit_code(exit_code_path)
                    if exit_code is not None:
                        agent_session.exit_code = exit_code

                if exit_code is None:
                    agent_session.status = (
                        AgentStatus.Stopped
                        if agent_session.agent_interface_mode
                        == AgentInterfaceMode.Structured
                        else AgentStatus.Error
                    )
                else:
                    agent_session.status = (
                        AgentStatus.Stopped if exit_code == 0 else AgentStatus.Error
                    )
                agent_session.ended_at = now
                await flush_session_update(agent_session)
                runtime_by_session_id.pop(agent_session.id, None)
                continue

        log_path = _log_path_from_attach(attach)
        if log_path is None:
            continue

        runtime = runtime_by_session_id.get(agent_session.id)
        if runtime is None:
            skip_history = _should_skip_log_history(
                agent_session=agent_session, driver_started_at=driver_started_at
            )
            runtime = _SessionRuntime(
                backend=backend_factory(agent_session=agent_session),
                cursor=_log_file_size(log_path) if skip_history else 0,
                skip_persist_until_log_activity=skip_history,
            )
            runtime_by_session_id[agent_session.id] = runtime

        log_text, next_cursor = _read_log_incremental(
            log_path=log_path,
            cursor=runtime.cursor,
            max_read_bytes=max_read_bytes,
        )
        runtime.cursor = next_cursor

        await process_log_text(
            agent_session=agent_session,
            runtime=runtime,
            log_text=log_text,
        )

    active_session_ids = {
        row.id
        for row in active_session_by_task_id.values()
        if row.ended_at is None
        and row.status in {AgentStatus.Running, AgentStatus.Blocked}
    }
    for session_id in list(runtime_by_session_id):
        if session_id not in active_session_ids:
            runtime_by_session_id.pop(session_id, None)

    if updated:
        await session.commit()
        if event_hub is not None:
            for row in pending_events:
                try:
                    payload = EventResponse.model_validate(
                        row, from_attributes=True
                    ).model_dump(by_alias=True, mode="json")
                except ValidationError:
                    payload = EventResponse.model_validate(
                        {
                            "id": row.id,
                            "event_type": row.event_type,
                            "created_at": row.created_at,
                            "data": {
                                "type": "unknown",
                                "event_type": row.event_type,
                                "data": cast(dict[str, Any], row.data),
                            },
                        }
                    ).model_dump(by_alias=True, mode="json")
                await event_hub.publish({"type": "event", "event": payload})
    return updated


async def run_agent_driver(
    ctx: RepoContext,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    interval_s: float,
    timeout_s: float = 1.0,
    once: bool = False,
    backend_factory: AgentBackendFactory = _default_backend_factory,
    event_hub: JsonWebSocketHub | None = None,
    tmux: TmuxSupervisor | None = None,
) -> None:
    runtime_by_session_id: dict[int, _SessionRuntime] = {}
    driver_started_at = datetime.now(UTC)
    last_error_at: float | None = None
    min_error_interval_s = 10.0
    while True:
        async with sessionmaker() as session:
            try:
                await supervise_once(
                    ctx,
                    session,
                    runtime_by_session_id,
                    backend_factory=backend_factory,
                    event_hub=event_hub,
                    tmux=tmux,
                    driver_started_at=driver_started_at,
                    timeout_s=timeout_s,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                now_s = time.monotonic()
                if (
                    last_error_at is None
                    or (now_s - last_error_at) >= min_error_interval_s
                ):
                    log.exception("tick.failed", error=str(exc))
                    last_error_at = now_s
        if once:
            return
        await asyncio.sleep(interval_s)
