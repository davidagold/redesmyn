from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import structlog
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.context import RepoContext
from redesmyn.db.context import open_db
from redesmyn.db.periodic import run_periodic_db_task
from redesmyn.db import (
    AgentSession,
    Epic,
    Event,
    MergeRun,
    Repository,
    Task,
    TaskStackInSyncState,
)
from redesmyn.db.models import GitCommitEventData, WorktreeHealthEventData
from redesmyn.domain.enums import AgentStatus, MergeRunStatus
from redesmyn.host_identity import load_or_create_host_identity
from redesmyn.repo import (
    current_branch,
    git_has_in_progress_operation,
    git_is_ancestor,
)
from redesmyn.repo_executor_leases import get_primary_host_key
from redesmyn.repo_identity import RepoKey
from redesmyn.git_projections import update_git_projections_in_session
from redesmyn.git_subprocess import run_git

log = structlog.get_logger("redesmyn.repo_observer")


@dataclass(slots=True, frozen=True)
class WorktreeObservation:
    worktree_path: str
    exists: bool
    current_branch: str | None
    dirty: bool | None
    branch_mismatch: bool | None


@dataclass(slots=True)
class RepoObserverState:
    last_head_by_task_id: dict[int, str] = field(default_factory=dict)
    last_worktree_by_task_id: dict[int, WorktreeObservation] = field(
        default_factory=dict
    )
    last_git_projections_refresh_at: datetime | None = None
    initialized: bool = False


@dataclass(frozen=True, slots=True)
class RepoObserverEvent:
    event_type: str
    data: dict[str, object]
    created_at: datetime


def default_worktree_path(ctx: RepoContext, *, branch: str) -> Path:
    safe_parts = []
    for part in branch.split("/"):
        if part in {"", ".", ".."}:
            continue
        safe_parts.append(part.replace(":", "_"))
    safe_rel = Path(*safe_parts) if safe_parts else Path(branch.replace(":", "_"))
    return ctx.state_dir / "worktrees" / safe_rel


def _find_existing_worktree_path_for_branch(
    repo_root: Path, *, branch: str
) -> Path | None:
    proc = run_git(["worktree", "list", "--porcelain"], cwd=repo_root, timeout_s=5)
    if proc.returncode != 0:
        return None

    current_path: Path | None = None
    current_branch: str | None = None

    def commit_current() -> Path | None:
        if current_path is None or current_branch is None:
            return None
        if current_branch == branch:
            return current_path
        return None

    for line in proc.stdout.splitlines():
        if not line.strip():
            match = commit_current()
            if match is not None:
                return match
            current_path = None
            current_branch = None
            continue

        if line.startswith("worktree "):
            current_path = Path(line.removeprefix("worktree ").strip())
            continue

        if line.startswith("branch "):
            ref = line.removeprefix("branch ").strip()
            if ref.startswith("refs/heads/"):
                current_branch = ref.removeprefix("refs/heads/")
            else:
                current_branch = ref
            continue

    return commit_current()


def read_branch_heads(repo_root: Path) -> dict[str, str]:
    proc = run_git(["show-ref", "--heads"], cwd=repo_root, timeout_s=5)
    if proc.returncode != 0:
        return {}

    heads: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        sha, ref = parts
        if not ref.startswith("refs/heads/"):
            continue
        branch = ref.removeprefix("refs/heads/")
        if branch:
            heads[branch] = sha
    return heads


@dataclass(slots=True, frozen=True)
class CommitSummary:
    sha: str
    author_name: str | None
    author_email: str | None
    authored_at: str | None
    subject: str | None


def read_commit_summaries(
    repo_root: Path, *, shas: list[str]
) -> dict[str, CommitSummary]:
    unique: list[str] = []
    seen: set[str] = set()
    for sha in shas:
        if not sha or sha in seen:
            continue
        seen.add(sha)
        unique.append(sha)

    if not unique:
        return {}

    fmt = "%H%x1f%an%x1f%ae%x1f%aI%x1f%s"
    proc = run_git(
        ["show", "-s", f"--format={fmt}", *unique], cwd=repo_root, timeout_s=5
    )
    if proc.returncode != 0:
        return {}

    summaries: dict[str, CommitSummary] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 5:
            continue
        sha, author_name, author_email, authored_at, subject = parts
        summaries[sha] = CommitSummary(
            sha=sha,
            author_name=author_name or None,
            author_email=author_email or None,
            authored_at=authored_at or None,
            subject=subject or None,
        )
    return summaries


def worktree_dirty(worktree_path: Path) -> bool | None:
    proc = run_git(["status", "--porcelain"], cwd=worktree_path, timeout_s=5)
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


class _TaskLike(Protocol):
    id: int
    branch_name: str | None
    worktree_path: str | None


def observe_worktree(*, task: _TaskLike, ctx: RepoContext) -> WorktreeObservation:
    if task.branch_name is None:
        raise RuntimeError("Task has no branch_name")

    if task.worktree_path:
        path = Path(task.worktree_path)
    else:
        path = _find_existing_worktree_path_for_branch(
            ctx.repo_root, branch=task.branch_name
        ) or default_worktree_path(ctx, branch=task.branch_name)
    if not path.exists():
        return WorktreeObservation(
            worktree_path=str(path),
            exists=False,
            current_branch=None,
            dirty=None,
            branch_mismatch=None,
        )

    branch = current_branch(cwd=path)
    dirty = worktree_dirty(path)
    mismatch = branch not in {task.branch_name, "HEAD"}
    return WorktreeObservation(
        worktree_path=str(path),
        exists=True,
        current_branch=branch,
        dirty=dirty,
        branch_mismatch=mismatch,
    )


async def observe_once(
    ctx: RepoContext,
    session: AsyncSession,
    state: RepoObserverState,
    *,
    emit_baseline: bool,
) -> int:
    tasks = list(
        await session.scalars(
            select(Task).where(Task.branch_name.is_not(None)).order_by(Task.id)
        )
    )
    if not tasks:
        if not state.initialized:
            state.initialized = True
        return 0

    task_ids = [t.id for t in tasks]
    latest_session_by_task_id: dict[int, AgentSession] = {}
    if task_ids:
        sessions = list(
            await session.scalars(
                select(AgentSession)
                .where(AgentSession.task_id.in_(task_ids))
                .order_by(desc(AgentSession.id))
            )
        )
        for sess in sessions:
            if sess.task_id in latest_session_by_task_id:
                continue
            latest_session_by_task_id[sess.task_id] = sess

    active_agent_session_id_by_task_id: dict[int, int] = {
        task_id: sess.id
        for task_id, sess in latest_session_by_task_id.items()
        if sess.status in {AgentStatus.Running, AgentStatus.Blocked}
        and sess.ended_at is None
    }

    heads = read_branch_heads(ctx.repo_root)

    now = datetime.now(UTC)
    host_key = load_or_create_host_identity(ctx).host_key
    repo = await session.scalar(
        select(Repository).where(Repository.repo_root == str(ctx.repo_root))
    )
    if repo is None:
        return 0
    repo_key = RepoKey(workspace_id=repo.workspace_id, repo_id=repo.repo_id)
    primary_host_key = await get_primary_host_key(session, repo_key, now=now)
    is_primary = primary_host_key == host_key
    events_added = 0
    merge_runs_updated = 0
    tasks_updated = 0
    projections_refreshed = False

    # This function does non-trivial git I/O (merge-base checks, worktree
    # inspection) and also records observations into SQLite. Ensure we do not
    # trigger autoflush mid-loop; flushing early can open a write transaction and
    # hold the single-writer lock long enough to trip the busy timeout in
    # unrelated API requests.
    with session.no_autoflush:
        new_shas: list[str] = []
        new_commits: list[tuple[Task, str]] = []
        for task in tasks:
            if task.branch_name is None:
                continue
            sha = heads.get(task.branch_name)
            if not sha:
                continue
            prev = state.last_head_by_task_id.get(task.id)
            if prev is None:
                state.last_head_by_task_id[task.id] = sha
                if emit_baseline:
                    new_shas.append(sha)
                    new_commits.append((task, sha))
                continue
            if prev == sha:
                continue
            state.last_head_by_task_id[task.id] = sha
            new_shas.append(sha)
            new_commits.append((task, sha))

        summaries = read_commit_summaries(ctx.repo_root, shas=new_shas)
        for task, sha in new_commits:
            summary = summaries.get(sha)
            data = GitCommitEventData(
                task_id=task.id,
                branch_name=task.branch_name or "",
                sha=sha,
                author_name=summary.author_name if summary else None,
                author_email=summary.author_email if summary else None,
                authored_at=summary.authored_at if summary else None,
                subject=summary.subject if summary else None,
                agent_session_id=active_agent_session_id_by_task_id.get(task.id),
            )
            payload = {
                **data.model_dump(mode="python"),
                "workspace_id": repo.workspace_id,
                "repo_id": repo.repo_id,
                "host_key": host_key,
            }
            session.add(
                Event(
                    event_type="git.commit",
                    data=payload,
                    created_at=now,
                )
            )
            events_added += 1

        should_refresh_git_projections = (
            state.last_git_projections_refresh_at is None
            or (now - state.last_git_projections_refresh_at).total_seconds() >= 5.0
        )
        if should_refresh_git_projections:
            try:
                await update_git_projections_in_session(
                    ctx=ctx,
                    session=session,
                    repo=repo,
                    host_key=host_key,
                    now=now,
                    tasks=tasks,
                )
                projections_refreshed = True
                state.last_git_projections_refresh_at = now
            except Exception:
                log.exception("repo_observer.git_projections_failed")

        # Compute stack_in_sync for each task
        epic_ids = {t.epic_id for t in tasks}
        epics_by_id: dict[int, Epic] = {}
        if epic_ids:
            epic_rows = list(
                await session.scalars(select(Epic).where(Epic.id.in_(epic_ids)))
            )
            epics_by_id = {e.id: e for e in epic_rows}

        tasks_by_id: dict[int, Task] = {t.id: t for t in tasks}
        sync_rows = list(
            await session.scalars(
                select(TaskStackInSyncState).where(
                    TaskStackInSyncState.host_key == host_key,
                    TaskStackInSyncState.task_id.in_(task_ids),
                )
            )
        )
        sync_by_task_id = {r.task_id: r for r in sync_rows}

        merged_into_base_by_task_id: dict[int, bool | None] = {}

        def effective_upstream_branch_for_task(task: Task) -> str | None:
            epic = epics_by_id.get(task.epic_id)
            base_branch = epic.root_branch if epic else None
            if base_branch is None:
                return None

            if task.parent_task_id is None:
                return base_branch

            if base_branch not in heads:
                return None

            seen: set[int] = set()
            current_parent_id = task.parent_task_id
            while current_parent_id is not None and current_parent_id not in seen:
                seen.add(current_parent_id)
                parent = tasks_by_id.get(current_parent_id)
                if parent is None or parent.branch_name is None:
                    return base_branch
                if parent.branch_name not in heads:
                    return None

                merged = merged_into_base_by_task_id.get(current_parent_id)
                if (
                    merged is None
                    and current_parent_id not in merged_into_base_by_task_id
                ):
                    try:
                        merged = git_is_ancestor(
                            ctx.repo_root, parent.branch_name, base_branch
                        )
                    except Exception:
                        merged = None
                    merged_into_base_by_task_id[current_parent_id] = merged

                if merged is True:
                    current_parent_id = parent.parent_task_id
                    continue

                return parent.branch_name

            return base_branch

        for task in tasks:
            in_sync: bool | None

            if task.branch_name is None or task.branch_name not in heads:
                in_sync = None
            else:
                upstream = effective_upstream_branch_for_task(task)
                if upstream is None:
                    in_sync = None
                else:
                    try:
                        in_sync = git_is_ancestor(
                            ctx.repo_root, upstream, task.branch_name
                        )
                    except Exception:
                        in_sync = None

            sync_state = sync_by_task_id.get(task.id)
            if sync_state is None:
                sync_state = TaskStackInSyncState(
                    task_id=task.id,
                    host_key=host_key,
                    stack_in_sync=in_sync,
                    observed_at=now,
                    updated_at=now,
                )
                session.add(sync_state)
                sync_by_task_id[task.id] = sync_state
            else:
                sync_state.stack_in_sync = in_sync
                sync_state.observed_at = now
                sync_state.updated_at = now

            if not is_primary:
                continue
            if task.stack_in_sync != in_sync:
                task.stack_in_sync = in_sync
                tasks_updated += 1

        for task in tasks:
            observed = observe_worktree(task=task, ctx=ctx)
            if (
                task.worktree_path is None
                and observed.exists
                and observed.current_branch == task.branch_name
            ):
                task.worktree_path = observed.worktree_path
                tasks_updated += 1

            prev = state.last_worktree_by_task_id.get(task.id)
            if prev is None:
                state.last_worktree_by_task_id[task.id] = observed
                if not emit_baseline:
                    continue
            elif prev == observed:
                continue

            state.last_worktree_by_task_id[task.id] = observed
            data = WorktreeHealthEventData(
                task_id=task.id,
                branch_name=task.branch_name or "",
                worktree_path=observed.worktree_path,
                exists=observed.exists,
                current_branch=observed.current_branch,
                dirty=observed.dirty,
                branch_mismatch=observed.branch_mismatch,
            )
            payload = {
                **data.model_dump(mode="python"),
                "workspace_id": repo.workspace_id,
                "repo_id": repo.repo_id,
                "host_key": host_key,
            }
            session.add(
                Event(
                    event_type="worktree.health",
                    data=payload,
                    created_at=now,
                )
            )
            events_added += 1

        blocked_runs = list(
            await session.scalars(
                select(MergeRun).where(MergeRun.status == MergeRunStatus.Blocked)
            )
        )
        for run in blocked_runs:
            if not run.blocked_worktree_path or not run.blocked_task_id:
                continue
            path = Path(run.blocked_worktree_path)
            if not path.exists():
                continue
            if git_has_in_progress_operation(path):
                continue
            run.status = MergeRunStatus.Resumable
            merge_runs_updated += 1
            session.add(
                Event(
                    event_type="merge.run",
                    data={
                        "run_id": run.run_id,
                        "task_id": run.blocked_task_id,
                        "epic_id": run.epic_id,
                        "requested_task_id": run.requested_task_id,
                        "status": run.status,
                        "operation": run.operation,
                        "workspace_id": repo.workspace_id,
                        "repo_id": repo.repo_id,
                        "host_key": host_key,
                    },
                    created_at=now,
                )
            )
            events_added += 1

    if events_added or tasks_updated or merge_runs_updated or projections_refreshed:
        await session.commit()
    if not state.initialized:
        state.initialized = True
    return events_added


def observe_repo(
    ctx: RepoContext,
    tasks: Sequence[_TaskLike],
    state: RepoObserverState,
    *,
    emit_baseline: bool,
    active_agent_session_id_by_task_id: dict[int, int],
    now: datetime | None = None,
) -> list[RepoObserverEvent]:
    created_at = now or datetime.now(UTC)
    heads = read_branch_heads(ctx.repo_root)
    events: list[RepoObserverEvent] = []

    new_shas: list[str] = []
    new_commits: list[tuple[_TaskLike, str]] = []
    for task in tasks:
        if task.branch_name is None:
            continue
        sha = heads.get(task.branch_name)
        if not sha:
            continue
        prev = state.last_head_by_task_id.get(task.id)
        if prev is None:
            state.last_head_by_task_id[task.id] = sha
            if emit_baseline:
                new_shas.append(sha)
                new_commits.append((task, sha))
            continue
        if prev == sha:
            continue
        state.last_head_by_task_id[task.id] = sha
        new_shas.append(sha)
        new_commits.append((task, sha))

    summaries = read_commit_summaries(ctx.repo_root, shas=new_shas)
    for task, sha in new_commits:
        summary = summaries.get(sha)
        data = GitCommitEventData(
            task_id=task.id,
            branch_name=task.branch_name or "",
            sha=sha,
            author_name=summary.author_name if summary else None,
            author_email=summary.author_email if summary else None,
            authored_at=summary.authored_at if summary else None,
            subject=summary.subject if summary else None,
            agent_session_id=active_agent_session_id_by_task_id.get(task.id),
        )
        events.append(
            RepoObserverEvent(
                event_type="git.commit",
                data=data.model_dump(mode="python"),
                created_at=created_at,
            )
        )

    for task in tasks:
        if task.branch_name is None:
            continue
        observed = observe_worktree(task=task, ctx=ctx)

        prev = state.last_worktree_by_task_id.get(task.id)
        if prev is None:
            state.last_worktree_by_task_id[task.id] = observed
            if not emit_baseline:
                continue
        elif prev == observed:
            continue

        state.last_worktree_by_task_id[task.id] = observed
        data = WorktreeHealthEventData(
            task_id=task.id,
            branch_name=task.branch_name or "",
            worktree_path=observed.worktree_path,
            exists=observed.exists,
            current_branch=observed.current_branch,
            dirty=observed.dirty,
            branch_mismatch=observed.branch_mismatch,
        )
        events.append(
            RepoObserverEvent(
                event_type="worktree.health",
                data=data.model_dump(mode="python"),
                created_at=created_at,
            )
        )

    return events


async def run_repo_observer(
    ctx: RepoContext,
    *,
    interval_s: float,
    emit_baseline: bool,
    once: bool,
) -> None:
    async with open_db(db_path=ctx.db_path, migrate=False) as db:
        state = RepoObserverState()

        async def _step(session: AsyncSession) -> None:
            await observe_once(
                ctx,
                session,
                state,
                emit_baseline=emit_baseline,
            )

        await run_periodic_db_task(
            sessionmaker=db.sessionmaker,
            interval_s=interval_s,
            once=once,
            name="repo_observer",
            step=_step,
            logger=log,
        )
