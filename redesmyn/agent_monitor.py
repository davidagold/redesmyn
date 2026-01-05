from __future__ import annotations

import asyncio
import os
import subprocess
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.context import RepoContext
from pydantic import TypeAdapter

from redesmyn.db import AgentSession, Task
from redesmyn.db.models import AttachInfo
from redesmyn.db.models import AttachTmux
from redesmyn.domain.enums import AgentStatus
from redesmyn.agent_runtime import (
    agent_session_log_path,
    has_tmux,
    tmux_session_name_for_task,
)


def _read_tmux_sessions(*, timeout_s: float) -> set[str]:
    proc = subprocess.run(
        ["tmux", "list-sessions", "-F", "#S"],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _tmux_pipe_to_log(*, session_name: str, log_path: str, timeout_s: float) -> None:
    cmd = f"cat >> {shlex.quote(log_path)}"
    subprocess.run(
        ["tmux", "pipe-pane", "-t", session_name, "-o", cmd],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )


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


async def observe_agents_once(
    ctx: RepoContext,
    session: AsyncSession,
    *,
    timeout_s: float,
) -> int:
    if not has_tmux():
        return 0

    sessions = _read_tmux_sessions(timeout_s=timeout_s)
    running_task_ids = {
        task_id
        for name in sessions
        if (task_id := _task_id_from_tmux_session(name)) is not None
    }
    now = datetime.now(UTC)
    updated = 0

    tasks_by_id: dict[int, Task] = {}
    if running_task_ids:
        tasks = list(
            await session.scalars(select(Task).where(Task.id.in_(running_task_ids)))
        )
        tasks_by_id = {t.id: t for t in tasks}

    active_sessions = list(
        await session.scalars(
            select(AgentSession)
            .where(AgentSession.status.in_([AgentStatus.Running, AgentStatus.Blocked]))
            .where(AgentSession.ended_at.is_(None))
            .order_by(desc(AgentSession.id))
        )
    )
    active_session_by_task_id: dict[int, AgentSession] = {}
    for agent_session in active_sessions:
        if agent_session.task_id in active_session_by_task_id:
            continue
        active_session_by_task_id[agent_session.task_id] = agent_session

    for task_id in running_task_ids:
        if task_id not in tasks_by_id:
            continue
        tmux_name = tmux_session_name_for_task(task_id=task_id)
        if tmux_name not in sessions:
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

        existing_attach = TypeAdapter(AttachInfo).validate_python(agent_session.attach)
        existing_log_path = (
            existing_attach.log_path
            if isinstance(existing_attach, AttachTmux)
            else None
        )
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

        _tmux_pipe_to_log(
            session_name=tmux_name,
            log_path=log_path,
            timeout_s=timeout_s,
        )
        if changed:
            updated += 1

    for task_id, agent_session in active_session_by_task_id.items():
        tmux_name = tmux_session_name_for_task(task_id=task_id)
        if tmux_name in sessions:
            continue
        agent_session.status = AgentStatus.Error
        agent_session.ended_at = now
        updated += 1

    if updated:
        await session.commit()
    return updated


async def run_agent_monitor(
    ctx: RepoContext,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    interval_s: float,
    timeout_s: float = 1.0,
    once: bool = False,
) -> None:
    while True:
        async with sessionmaker() as session:
            try:
                await observe_agents_once(ctx, session, timeout_s=timeout_s)
            except Exception:
                pass
        if once:
            return
        await asyncio.sleep(interval_s)
