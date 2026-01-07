from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

import structlog
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.agent_runtime import send_task_agent_text
from redesmyn.db import AgentSession, Event, MergeRun, Repository
from redesmyn.db.models import AttachInfo
from redesmyn.domain.enums import AgentStatus, AgentTurnState, MergeRunStatus
from redesmyn.repo import git_has_in_progress_operation, git_status_porcelain
from redesmyn.repo_executor import RepoExecutor, RepoExecutorTarget
from redesmyn.repo_executor_leases import (
    acquire_or_refresh_primary,
    get_primary_host_key,
)
from redesmyn.repo_identity import RepoKey
from redesmyn.schemas.core import (
    MergeConflictAssistStatusResponse,
    MergeRunResumeRequest,
)

log = structlog.get_logger("redesmyn.merge_conflict_assist")


class AgentTextSender(Protocol):
    async def send(self, *, agent_session: AgentSession, text: str) -> bool: ...


class MergeRunResumer(Protocol):
    async def resume(self, *, run: MergeRun) -> bool: ...


def _parse_attach(agent_session: AgentSession) -> AttachInfo | None:
    try:
        return TypeAdapter(AttachInfo).validate_python(agent_session.attach)
    except ValidationError:
        return None


def _supports_conflict_assist(agent_session: AgentSession) -> bool:
    caps = agent_session.agent_capabilities or {}
    return (
        bool(caps.get("can_send_text"))
        and bool(caps.get("can_detect_ready_for_input"))
        and bool(caps.get("can_detect_turn_complete"))
    )


def _turn_state(agent_session: AgentSession) -> AgentTurnState:
    raw = agent_session.agent_semantic_status or {}
    value = raw.get("turn_state")
    try:
        return AgentTurnState(value)  # type: ignore[arg-type]
    except Exception:
        return AgentTurnState.Unknown


def _is_ready_for_input(agent_session: AgentSession) -> bool:
    return _turn_state(agent_session) in {
        AgentTurnState.Ready,
        AgentTurnState.Completed,
    }


def _rebase_remediation_message(run: MergeRun) -> str | None:
    if run.blocked_step_kind != "rebase":
        return None
    worktree = run.blocked_worktree_path
    branch = run.blocked_branch_name
    if not worktree or not branch:
        return None
    operation = run.operation
    return "\n".join(
        [
            f"We hit a rebase conflict while {'restacking' if operation == 'restack' else 'merging'} the stack.",
            "",
            f"Branch: {branch}",
            f"Worktree: {worktree}",
            "",
            "Please:",
            "- Review and resolve the conflict(s).",
            f"- `cd {worktree}`",
            "- `git status` (to see conflicted files)",
            "- `git add -A`",
            "- `git rebase --continue`",
            "- Repeat until the rebase completes (resolving any further conflicts).",
            "",
            f"Once the rebase finishes and the worktree is clean, let me know so I can resume the {operation} run.",
        ]
    )


def _has_unmerged_entries(status_porcelain: str) -> bool:
    for line in status_porcelain.splitlines():
        if len(line) < 2:
            continue
        xy = line[:2]
        if xy in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}:
            return True
    return False


def _worktree_is_clean_for_resume(worktree_path: str) -> tuple[bool, str | None]:
    path = Path(worktree_path)
    if not path.exists():
        return False, "Blocked worktree path no longer exists."
    if git_has_in_progress_operation(path):
        return False, "Git operation still in progress (rebase/merge/etc.)."
    try:
        status = git_status_porcelain(path)
    except Exception as e:
        return False, f"Failed to read git status: {e}"
    if _has_unmerged_entries(status):
        return False, "Worktree still has unmerged entries."
    if status.strip():
        return False, "Worktree has uncommitted changes."
    return True, None


@dataclass(slots=True)
class _AssistRuntime:
    run_id: str
    created_at: datetime
    delivery_deadline_at: datetime
    overall_deadline_at: datetime
    agent_task_id: int | None = None
    agent_session_id: int | None = None
    sent_at: datetime | None = None
    baseline_turn_completed_event_id: int | None = None
    observed_turn_completed_event_id: int | None = None
    state: Literal[
        "inactive",
        "waiting_for_agent_ready",
        "sent_waiting_for_turn_complete",
        "waiting_for_repo_clean",
        "ready_to_resume",
        "timed_out",
        "unsupported",
        "resumed",
    ] = "inactive"
    detail: str | None = None


class DefaultAgentTextSender:
    async def send(self, *, agent_session: AgentSession, text: str) -> bool:
        if agent_session.task_id is None:
            return False
        attach = _parse_attach(agent_session)
        if attach is None or attach.type != "tmux":
            return False
        return await send_task_agent_text(task_id=agent_session.task_id, text=text)


@dataclass(slots=True)
class DefaultMergeRunResumer:
    sessionmaker: async_sessionmaker[AsyncSession]
    repo_executor: RepoExecutor
    runner_mode: str
    local_host_key: str | None
    repo_root: Path

    async def resume(self, *, run: MergeRun) -> bool:
        run_id = run.run_id
        local_host_key = self.local_host_key if self.runner_mode == "local" else None

        async with self.sessionmaker() as session:
            run_row = await session.scalar(
                select(MergeRun).where(MergeRun.run_id == run_id)
            )
            if run_row is None or run_row.status != MergeRunStatus.Resumable:
                return False

            repo = await session.scalar(
                select(Repository).where(Repository.repo_root == str(self.repo_root))
            )
            if repo is None:
                return False

            repo_key = RepoKey(workspace_id=repo.workspace_id, repo_id=repo.repo_id)
            primary_host_key = await get_primary_host_key(session, repo_key)
            if primary_host_key is None and local_host_key is not None:
                try:
                    if await acquire_or_refresh_primary(
                        session, repo_key, host_key=local_host_key
                    ):
                        await session.commit()
                        primary_host_key = local_host_key
                except Exception:
                    await session.rollback()
                    primary_host_key = None

            target_host_key = run_row.host_key or primary_host_key
            if target_host_key is None:
                return False
            if (
                run_row.canonical
                and primary_host_key is not None
                and target_host_key != primary_host_key
            ):
                return False

            is_local_executor = (
                local_host_key is not None and target_host_key == local_host_key
            )
            target = RepoExecutorTarget(
                repo=repo,
                repo_key=repo_key,
                target_host_key=target_host_key,
                primary_host_key=primary_host_key,
                canonical=run_row.canonical,
                is_local=is_local_executor,
            )

        try:
            await self.repo_executor.resume_merge_run(
                target=target,
                run=run,
                request=MergeRunResumeRequest(allow_running=run.allow_running),
            )
        except Exception:
            return False
        return True


@dataclass(slots=True)
class MergeConflictAssistSupervisor:
    """In-memory conflict assist state machine keyed by merge run id."""

    sender: AgentTextSender
    resumer: MergeRunResumer
    delivery_timeout: timedelta = timedelta(minutes=2)
    overall_timeout: timedelta = timedelta(hours=2)

    _runtime_by_run_id: dict[str, _AssistRuntime] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._runtime_by_run_id = {}

    def snapshot(self, *, run_id: str) -> MergeConflictAssistStatusResponse | None:
        runtime = self._runtime_by_run_id.get(run_id)
        if runtime is None:
            return None

        waiting_on: list[
            Literal["agent_ready", "agent_turn_complete", "repo_clean"]
        ] = []
        if runtime.state == "waiting_for_agent_ready":
            waiting_on.append("agent_ready")
        elif runtime.state == "sent_waiting_for_turn_complete":
            waiting_on.append("agent_turn_complete")
        elif runtime.state == "waiting_for_repo_clean":
            waiting_on.append("repo_clean")

        active = runtime.state in {
            "waiting_for_agent_ready",
            "sent_waiting_for_turn_complete",
            "waiting_for_repo_clean",
            "ready_to_resume",
        }
        return MergeConflictAssistStatusResponse(
            active=active,
            state=runtime.state,
            waiting_on=waiting_on,
            agent_task_id=runtime.agent_task_id,
            agent_session_id=runtime.agent_session_id,
            message_sent_at=runtime.sent_at,
            timeout_at=runtime.overall_deadline_at,
            detail=runtime.detail,
        )

    def ensure_tracking(self, *, run_id: str, now: datetime) -> _AssistRuntime:
        runtime = self._runtime_by_run_id.get(run_id)
        if runtime is not None:
            return runtime
        runtime = _AssistRuntime(
            run_id=run_id,
            created_at=now,
            delivery_deadline_at=now + self.delivery_timeout,
            overall_deadline_at=now + self.overall_timeout,
            state="inactive",
        )
        self._runtime_by_run_id[run_id] = runtime
        return runtime

    def stop_tracking(self, *, run_id: str) -> None:
        self._runtime_by_run_id.pop(run_id, None)

    async def tick(self, session: AsyncSession, *, now: datetime) -> None:
        merge_runs = list(
            await session.scalars(
                select(MergeRun).where(
                    MergeRun.status.in_(
                        [MergeRunStatus.Blocked, MergeRunStatus.Resumable]
                    )
                )
            )
        )

        active_run_ids: set[str] = set()

        for run in merge_runs:
            if run.blocked_step_kind != "rebase":
                continue
            if not run.blocked_task_id:
                continue
            active_run_ids.add(run.run_id)
            runtime = self.ensure_tracking(run_id=run.run_id, now=now)

            if now >= runtime.overall_deadline_at and runtime.state not in {
                "timed_out",
                "resumed",
            }:
                runtime.state = "timed_out"
                runtime.detail = "Timed out waiting for conflict assist completion; use manual resume."
                continue

            candidate_task_ids: list[int] = [run.blocked_task_id]
            if run.requested_task_id not in candidate_task_ids:
                candidate_task_ids.append(run.requested_task_id)

            agent_session: AgentSession | None = None
            runtime.agent_task_id = None
            runtime.agent_session_id = None

            for task_id in candidate_task_ids:
                row = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.task_id == task_id)
                    .order_by(AgentSession.id.desc())
                    .limit(1)
                )
                if row is None or row.status != AgentStatus.Running:
                    continue
                if not _supports_conflict_assist(row):
                    continue
                agent_session = row
                runtime.agent_task_id = task_id
                runtime.agent_session_id = row.id
                break

            if agent_session is None:
                runtime.state = "unsupported"
                runtime.detail = "No running agent session supports conflict assist."
                continue

            message = _rebase_remediation_message(run)
            if not message:
                runtime.state = "unsupported"
                runtime.detail = (
                    "No remediation message available for this blocked run."
                )
                continue

            if runtime.sent_at is None:
                if now >= runtime.delivery_deadline_at:
                    runtime.state = "timed_out"
                    runtime.detail = "Timed out waiting to deliver remediation message; use manual copy."
                    continue
                if not _is_ready_for_input(agent_session):
                    runtime.state = "waiting_for_agent_ready"
                    runtime.detail = None
                    continue

                baseline_event_id: int | None = None
                recent_events = list(
                    await session.scalars(
                        select(Event)
                        .where(Event.event_type == "agent.turn_completed")
                        .order_by(Event.id.desc())
                        .limit(200)
                    )
                )
                for ev in recent_events:
                    if ev.data.get("agent_session_id") == agent_session.id:
                        baseline_event_id = ev.id
                        break

                delivered = await self.sender.send(
                    agent_session=agent_session, text=message
                )
                if not delivered:
                    runtime.state = "waiting_for_agent_ready"
                    runtime.detail = (
                        "Failed to deliver remediation message (no transport)."
                    )
                    continue

                runtime.sent_at = now
                runtime.baseline_turn_completed_event_id = baseline_event_id
                runtime.state = "sent_waiting_for_turn_complete"
                runtime.detail = None
                continue

            baseline = runtime.baseline_turn_completed_event_id or 0
            recent_events = list(
                await session.scalars(
                    select(Event)
                    .where(Event.event_type == "agent.turn_completed")
                    .where(Event.id > baseline)
                    .order_by(Event.id.desc())
                    .limit(200)
                )
            )
            for ev in recent_events:
                if ev.data.get("agent_session_id") == runtime.agent_session_id:
                    runtime.observed_turn_completed_event_id = ev.id
                    break

            if runtime.observed_turn_completed_event_id is None:
                runtime.state = "sent_waiting_for_turn_complete"
                runtime.detail = None
                continue

            if run.status != MergeRunStatus.Resumable:
                runtime.state = "waiting_for_repo_clean"
                runtime.detail = "Waiting for conflicts resolved / worktree unblocked."
                continue

            if not run.blocked_worktree_path:
                runtime.state = "waiting_for_repo_clean"
                runtime.detail = "Blocked worktree path missing."
                continue

            repo_ok, repo_detail = _worktree_is_clean_for_resume(
                run.blocked_worktree_path
            )
            if not repo_ok:
                runtime.state = "waiting_for_repo_clean"
                runtime.detail = repo_detail
                continue

            runtime.state = "ready_to_resume"
            runtime.detail = None
            resumed = await self.resumer.resume(run=run)
            if resumed:
                runtime.state = "resumed"
                runtime.detail = None
            else:
                runtime.state = "waiting_for_repo_clean"
                runtime.detail = "Failed to resume automatically; use manual resume."

        for run_id in list(self._runtime_by_run_id.keys()):
            if run_id not in active_run_ids:
                self.stop_tracking(run_id=run_id)


async def run_merge_conflict_assist(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    supervisor: MergeConflictAssistSupervisor,
    interval_s: float = 1.0,
    once: bool = False,
) -> None:
    while True:
        try:
            now = datetime.now(UTC)
            async with sessionmaker() as session:
                await supervisor.tick(session, now=now)
                await session.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("merge_conflict_assist.tick_failed")

        if once:
            return
        await asyncio.sleep(max(0.05, interval_s))


def make_default_supervisor(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    repo_executor: RepoExecutor,
    runner_mode: str,
    local_host_key: str | None,
    repo_root: Path,
) -> MergeConflictAssistSupervisor:
    return MergeConflictAssistSupervisor(
        sender=DefaultAgentTextSender(),
        resumer=DefaultMergeRunResumer(
            sessionmaker=sessionmaker,
            repo_executor=repo_executor,
            runner_mode=runner_mode,
            local_host_key=local_host_key,
            repo_root=repo_root,
        ),
    )
