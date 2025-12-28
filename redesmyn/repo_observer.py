from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.context import RepoContext
from redesmyn.db import Agent, Event, Node
from redesmyn.db.models import GitCommitEventData, WorktreeHealthEventData
from redesmyn.domain.enums import AgentStatus
from redesmyn.repo import current_branch


@dataclass(slots=True, frozen=True)
class WorktreeObservation:
    worktree_path: str
    exists: bool
    current_branch: str | None
    dirty: bool | None
    branch_mismatch: bool | None


@dataclass(slots=True)
class RepoObserverState:
    last_head_by_node_id: dict[int, str] = field(default_factory=dict)
    last_worktree_by_node_id: dict[int, WorktreeObservation] = field(
        default_factory=dict
    )
    initialized: bool = False


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
    proc = _run_git(["worktree", "list", "--porcelain"], cwd=repo_root, timeout_s=5)
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


def _run_git(
    args: list[str], *, cwd: Path, timeout_s: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )


def read_branch_heads(repo_root: Path) -> dict[str, str]:
    proc = _run_git(["show-ref", "--heads"], cwd=repo_root, timeout_s=5)
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
    proc = _run_git(
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
    proc = _run_git(["status", "--porcelain"], cwd=worktree_path, timeout_s=5)
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


def observe_worktree(*, node: Node, ctx: RepoContext) -> WorktreeObservation:
    if node.worktree_path:
        path = Path(node.worktree_path)
    else:
        path = _find_existing_worktree_path_for_branch(
            ctx.repo_root, branch=node.branch_name
        ) or default_worktree_path(ctx, branch=node.branch_name)
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
    mismatch = branch not in {node.branch_name, "HEAD"}
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
    nodes = list(await session.scalars(select(Node).order_by(Node.id)))
    if not nodes:
        if not state.initialized:
            state.initialized = True
        return 0

    agent_ids = [n.agent_id for n in nodes if n.agent_id is not None]
    agents = (
        list(await session.scalars(select(Agent).where(Agent.id.in_(agent_ids))))
        if agent_ids
        else []
    )
    agent_by_id = {a.id: a for a in agents}
    active_agent_id_by_node_id: dict[int, int] = {
        n.id: n.agent_id
        for n in nodes
        if n.agent_id is not None
        and (agent := agent_by_id.get(n.agent_id)) is not None
        and agent.status in {AgentStatus.Running, AgentStatus.Blocked}
    }

    heads = read_branch_heads(ctx.repo_root)

    now = datetime.now(UTC)
    events_added = 0
    nodes_updated = 0

    new_shas: list[str] = []
    new_commits: list[tuple[Node, str]] = []
    for node in nodes:
        sha = heads.get(node.branch_name)
        if not sha:
            continue
        prev = state.last_head_by_node_id.get(node.id)
        if prev is None:
            state.last_head_by_node_id[node.id] = sha
            if emit_baseline:
                new_shas.append(sha)
                new_commits.append((node, sha))
            continue
        if prev == sha:
            continue
        state.last_head_by_node_id[node.id] = sha
        new_shas.append(sha)
        new_commits.append((node, sha))

    summaries = read_commit_summaries(ctx.repo_root, shas=new_shas)
    for node, sha in new_commits:
        summary = summaries.get(sha)
        data = GitCommitEventData(
            node_id=node.id,
            branch_name=node.branch_name,
            sha=sha,
            author_name=summary.author_name if summary else None,
            author_email=summary.author_email if summary else None,
            authored_at=summary.authored_at if summary else None,
            subject=summary.subject if summary else None,
            agent_id=active_agent_id_by_node_id.get(node.id),
        )
        session.add(
            Event(
                event_type="git.commit",
                data=data.model_dump(mode="python"),
                created_at=now,
            )
        )
        events_added += 1

    for node in nodes:
        observed = observe_worktree(node=node, ctx=ctx)
        if (
            node.worktree_path is None
            and observed.exists
            and observed.current_branch == node.branch_name
        ):
            node.worktree_path = observed.worktree_path
            nodes_updated += 1

        prev = state.last_worktree_by_node_id.get(node.id)
        if prev is None:
            state.last_worktree_by_node_id[node.id] = observed
            if not emit_baseline:
                continue
        elif prev == observed:
            continue

        state.last_worktree_by_node_id[node.id] = observed
        data = WorktreeHealthEventData(
            node_id=node.id,
            branch_name=node.branch_name,
            worktree_path=observed.worktree_path,
            exists=observed.exists,
            current_branch=observed.current_branch,
            dirty=observed.dirty,
            branch_mismatch=observed.branch_mismatch,
        )
        session.add(
            Event(
                event_type="worktree.health",
                data=data.model_dump(mode="python"),
                created_at=now,
            )
        )
        events_added += 1

    if events_added or nodes_updated:
        await session.commit()
    if not state.initialized:
        state.initialized = True
    return events_added


async def run_repo_observer(
    ctx: RepoContext,
    *,
    interval_s: float,
    emit_baseline: bool,
    once: bool,
) -> None:
    from redesmyn.db import create_engine, create_sessionmaker

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        state = RepoObserverState()
        while True:
            async with sessionmaker() as session:
                await observe_once(
                    ctx,
                    session,
                    state,
                    emit_baseline=emit_baseline,
                )
            if once:
                return
            await asyncio.sleep(interval_s)
    finally:
        await engine.dispose()
