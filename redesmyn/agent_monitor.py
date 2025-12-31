from __future__ import annotations

import asyncio
import subprocess
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.context import RepoContext
from redesmyn.db import Agent, Task
from redesmyn.db.models import AttachTmux
from redesmyn.domain.enums import AgentStatus
from redesmyn.agent_runtime import agent_log_path, has_tmux, tmux_session_name_for_task


@dataclass(frozen=True, slots=True)
class AgentLivenessRow:
    task: Task
    agent: Agent


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


async def _load_liveness_rows(session: AsyncSession) -> list[AgentLivenessRow]:
    rows = list(
        await session.execute(
            select(Task, Agent)
            .join(Agent, Task.agent_id == Agent.id)
            .where(
                Task.agent_id.is_not(None),
            )
            .order_by(Task.id)
        )
    )
    return [AgentLivenessRow(task=task, agent=agent) for task, agent in rows]


async def observe_agents_once(
    ctx: RepoContext,
    session: AsyncSession,
    *,
    timeout_s: float,
) -> int:
    if not has_tmux():
        return 0

    sessions = _read_tmux_sessions(timeout_s=timeout_s)
    now = datetime.now(UTC)
    updated = 0

    for row in await _load_liveness_rows(session):
        task_id = row.task.id

        tmux_name = tmux_session_name_for_task(task_id=task_id)
        is_running = tmux_name in sessions
        agent = row.agent

        if is_running:
            changed = False
            if agent.status not in {AgentStatus.Running, AgentStatus.Blocked}:
                agent.status = AgentStatus.Running
                changed = True

            if agent.started_at is None:
                agent.started_at = now
                changed = True

            if agent.ended_at is not None:
                agent.ended_at = None
                changed = True

            log_path = agent_log_path(ctx, agent_id=agent.id)
            attach = AttachTmux(
                session=tmux_name,
                socket_path=None,
                log_path=str(log_path),
            )
            attach_dict = attach.model_dump(mode="python")

            if agent.last_seen_at != now:
                agent.last_seen_at = now
                changed = True
            if agent.attach != attach_dict:
                agent.attach = attach_dict
                changed = True

            _tmux_pipe_to_log(
                session_name=tmux_name,
                log_path=str(log_path),
                timeout_s=timeout_s,
            )
            if changed:
                updated += 1
            continue

        if agent.status in {AgentStatus.Running, AgentStatus.Blocked}:
            agent.status = AgentStatus.Error
            agent.last_seen_at = now
            agent.ended_at = now
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
