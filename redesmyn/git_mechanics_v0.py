from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.agent_runtime import ensure_node_worktree
from redesmyn.context import RepoContext
from redesmyn.db import Agent, Epic, Node, Task
from redesmyn.domain.enums import AgentStatus, TaskState
from redesmyn.repo import (
    GitCommandError,
    current_branch,
    git_has_in_progress_operation,
    git_is_ancestor,
    git_merge_ff_only,
    git_rebase,
    git_status_porcelain,
    git_worktree_path_for_branch,
)

MergeCascadeScope = Literal["descendants", "spine"]


class MergePlanError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunningAgentInfo:
    node_id: int
    task_id: int | None
    branch_name: str
    agent_id: int
    agent_name: str
    agent_status: AgentStatus


class MergeBlockedByRunningAgents(MergePlanError):
    def __init__(self, agents: Sequence[RunningAgentInfo]):
        super().__init__("Merge affects running tasks; confirmation required.")
        self.agents = list(agents)


@dataclass(frozen=True, slots=True)
class MergeNodeInfo:
    node_id: int
    parent_node_id: int | None
    branch_name: str
    upstream_ref: str
    worktree_path: Path
    primary_task_id: int | None


MergeStepKind = Literal["rebase", "merge_ff"]


@dataclass(frozen=True, slots=True)
class MergePlanStep:
    kind: MergeStepKind
    node_id: int | None
    task_id: int | None
    branch_name: str
    worktree_path: Path
    upstream_ref: str | None = None
    base_branch: str | None = None


@dataclass(frozen=True, slots=True)
class MergeCascadePlan:
    run_id: str
    epic_id: int
    base_branch: str
    base_worktree: Path
    scope: MergeCascadeScope
    spine_node_ids: tuple[int, ...]
    affected_node_ids: tuple[int, ...]
    nodes: dict[int, MergeNodeInfo]
    steps: tuple[MergePlanStep, ...]
    running_agents: tuple[RunningAgentInfo, ...]


MergeStepPhase = Literal["started", "finished", "failed"]


@dataclass(frozen=True, slots=True)
class MergeRunStepUpdate:
    step_index: int
    step: MergePlanStep
    phase: MergeStepPhase
    blocked: bool = False
    error: str | None = None


async def _load_task_node_epic(
    session: AsyncSession, *, task_id: int
) -> tuple[Task, Node, Epic]:
    task = await session.get(Task, task_id)
    if task is None:
        raise MergePlanError(f"Unknown task id: {task_id}")
    if task.node_id is None:
        raise MergePlanError(
            "Task has no node/branch yet (node_id is null). "
            "Run `rn sync --from local --create-nodes`."
        )
    node = await session.get(Node, task.node_id)
    if node is None:
        raise MergePlanError(f"Task node not found: {task.node_id}")
    epic = await session.get(Epic, task.epic_id)
    if epic is None:
        raise MergePlanError(f"Epic not found: {task.epic_id}")
    return task, node, epic


def _compute_spine_node_ids(
    *, nodes_by_id: dict[int, Node], leaf_node_id: int
) -> list[int]:
    spine: list[int] = []
    cursor: int | None = leaf_node_id
    while cursor is not None:
        spine.append(cursor)
        parent = nodes_by_id.get(cursor)
        if parent is None:
            break
        cursor = parent.parent_node_id
    spine.reverse()
    return spine


def _compute_descendants(
    *, children_by_parent: dict[int | None, list[int]], start_node_ids: Sequence[int]
) -> set[int]:
    seen: set[int] = set()
    queue: list[int] = list(start_node_ids)
    while queue:
        node_id = queue.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for child in children_by_parent.get(node_id, []):
            queue.append(child)
    return seen


def _topo_sort_by_depth(
    *, nodes_by_id: dict[int, Node], node_ids: set[int]
) -> list[int]:
    depth_cache: dict[int, int] = {}

    def depth(node_id: int) -> int:
        if node_id in depth_cache:
            return depth_cache[node_id]
        node = nodes_by_id.get(node_id)
        if node is None or node.parent_node_id is None:
            depth_cache[node_id] = 0
            return 0
        value = depth(node.parent_node_id) + 1
        depth_cache[node_id] = value
        return value

    ordered = sorted(node_ids, key=lambda nid: (depth(nid), nid))
    return ordered


async def build_merge_cascade_plan(
    *,
    ctx: RepoContext,
    sessionmaker: async_sessionmaker[AsyncSession],
    task_id: int,
    run_id: str,
    scope: MergeCascadeScope,
    force: bool,
) -> MergeCascadePlan:
    async with sessionmaker() as session:
        task, leaf_node, epic = await _load_task_node_epic(session, task_id=task_id)
        base_branch = epic.root_branch

        nodes = list(await session.scalars(select(Node).where(Node.epic_id == epic.id)))
        nodes_by_id: dict[int, Node] = {n.id: n for n in nodes}
        children_by_parent: dict[int | None, list[int]] = {}
        for n in nodes:
            children_by_parent.setdefault(n.parent_node_id, []).append(n.id)
        for value in children_by_parent.values():
            value.sort()

        spine_node_ids = _compute_spine_node_ids(
            nodes_by_id=nodes_by_id, leaf_node_id=leaf_node.id
        )

        affected_ids: set[int] = set(spine_node_ids)
        if scope == "descendants":
            affected_ids = _compute_descendants(
                children_by_parent=children_by_parent, start_node_ids=spine_node_ids
            )

        ordered_node_ids = _topo_sort_by_depth(
            nodes_by_id=nodes_by_id, node_ids=affected_ids
        )

        base_worktree = git_worktree_path_for_branch(ctx.repo_root, base_branch)
        if base_worktree is None:
            raise MergePlanError(
                f"No worktree has {base_branch!r} checked out. "
                f"Check out {base_branch} in a worktree and retry."
            )

        # Ensure worktrees exist and persist their paths.
        node_infos: dict[int, MergeNodeInfo] = {}
        for node_id in ordered_node_ids:
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            worktree_path = await ensure_node_worktree(
                session,
                ctx,
                node=node,
                epic=epic,
            )
            upstream_ref = (
                nodes_by_id[node.parent_node_id].branch_name
                if node.parent_node_id is not None
                and node.parent_node_id in nodes_by_id
                else base_branch
            )
            node_infos[node_id] = MergeNodeInfo(
                node_id=node.id,
                parent_node_id=node.parent_node_id,
                branch_name=node.branch_name,
                upstream_ref=upstream_ref,
                worktree_path=worktree_path,
                primary_task_id=node.primary_task_id,
            )

        await session.commit()

        # Merge-ready gating (spine).
        if not force:
            spine_tasks = list(
                await session.scalars(
                    select(Task).where(Task.node_id.in_(spine_node_ids))
                )
            )
            task_by_id = {t.id: t for t in spine_tasks}
            missing_ready: list[str] = []
            for node_id in spine_node_ids:
                node = nodes_by_id.get(node_id)
                if node is None:
                    continue
                primary_id = node.primary_task_id
                candidate = (
                    task_by_id.get(primary_id)
                    if primary_id is not None
                    else next((t for t in spine_tasks if t.node_id == node_id), None)
                )
                if candidate is None:
                    continue
                if candidate.state == TaskState.Done:
                    continue
                if candidate.merge_ready_at is None:
                    missing_ready.append(f"{candidate.id} ({node.branch_name})")
            if missing_ready:
                raise MergePlanError(
                    "Task(s) on the merge spine are not marked ready: "
                    + ", ".join(missing_ready)
                )

        # Running-agent detection for affected set.
        running_agents: list[RunningAgentInfo] = []
        agent_ids = [
            nodes_by_id[nid].agent_id
            for nid in ordered_node_ids
            if nid in nodes_by_id and nodes_by_id[nid].agent_id is not None
        ]
        agents_by_id: dict[int, Agent] = {}
        if agent_ids:
            rows = list(
                await session.scalars(select(Agent).where(Agent.id.in_(agent_ids)))
            )
            agents_by_id = {a.id: a for a in rows}

        tasks_for_affected = list(
            await session.scalars(
                select(Task).where(Task.node_id.in_(ordered_node_ids))
            )
        )
        task_by_node_id: dict[int, Task] = {}
        for t in tasks_for_affected:
            if t.node_id is None:
                continue
            existing = task_by_node_id.get(t.node_id)
            if existing is None:
                task_by_node_id[t.node_id] = t
                continue
            # Prefer merge-ready/primary-ish task, but best-effort only.
            if existing.merge_ready_at is None and t.merge_ready_at is not None:
                task_by_node_id[t.node_id] = t

        for node_id in ordered_node_ids:
            node = nodes_by_id.get(node_id)
            if node is None or node.agent_id is None:
                continue
            agent = agents_by_id.get(node.agent_id)
            if agent is None:
                continue
            if agent.status not in {AgentStatus.Running, AgentStatus.Blocked}:
                continue
            task_row = task_by_node_id.get(node_id)
            running_agents.append(
                RunningAgentInfo(
                    node_id=node_id,
                    task_id=None if task_row is None else task_row.id,
                    branch_name=node.branch_name,
                    agent_id=agent.id,
                    agent_name=agent.display_name,
                    agent_status=agent.status,
                )
            )

    # Hard checks (git) outside the DB session.
    try:
        if current_branch(cwd=base_worktree) != base_branch:
            raise MergePlanError(
                f"Base worktree is not on {base_branch}: {base_worktree}"
            )
        if git_status_porcelain(base_worktree).strip():
            raise MergePlanError(
                f"Base worktree has uncommitted changes: {base_worktree}"
            )
        if git_has_in_progress_operation(base_worktree):
            raise MergePlanError(
                f"Base worktree has an in-progress git operation: {base_worktree}"
            )

        for node_id in ordered_node_ids:
            info = node_infos.get(node_id)
            if info is None:
                continue
            if current_branch(cwd=info.worktree_path) != info.branch_name:
                raise MergePlanError(
                    f"Worktree is not on {info.branch_name} (at {info.worktree_path})"
                )
            if git_status_porcelain(info.worktree_path).strip():
                raise MergePlanError(
                    f"Worktree has uncommitted changes: {info.worktree_path}"
                )
            if git_has_in_progress_operation(info.worktree_path):
                raise MergePlanError(
                    f"Worktree has an in-progress git operation: {info.worktree_path}"
                )
    except GitCommandError as e:
        raise MergePlanError(str(e)) from e

    steps: list[MergePlanStep] = []
    for node_id in ordered_node_ids:
        info = node_infos.get(node_id)
        if info is None:
            continue
        steps.append(
            MergePlanStep(
                kind="rebase",
                node_id=info.node_id,
                task_id=info.primary_task_id,
                branch_name=info.branch_name,
                worktree_path=info.worktree_path,
                upstream_ref=info.upstream_ref,
            )
        )
    for node_id in spine_node_ids:
        info = node_infos.get(node_id)
        if info is None:
            continue
        steps.append(
            MergePlanStep(
                kind="merge_ff",
                node_id=info.node_id,
                task_id=info.primary_task_id,
                branch_name=info.branch_name,
                worktree_path=base_worktree,
                base_branch=base_branch,
            )
        )

    return MergeCascadePlan(
        run_id=run_id,
        epic_id=epic.id,
        base_branch=base_branch,
        base_worktree=base_worktree,
        scope=scope,
        spine_node_ids=tuple(spine_node_ids),
        affected_node_ids=tuple(ordered_node_ids),
        nodes=node_infos,
        steps=tuple(steps),
        running_agents=tuple(running_agents),
    )


async def execute_merge_cascade_plan(
    *,
    ctx: RepoContext,
    sessionmaker: async_sessionmaker[AsyncSession],
    plan: MergeCascadePlan,
    allow_running: bool,
    emit_event: Callable[[dict[str, object]], Awaitable[None]] | None = None,
    update_run: Callable[[MergeRunStepUpdate], Awaitable[None]] | None = None,
    start_at_step_index: int = 0,
) -> None:
    if plan.running_agents and not allow_running:
        raise MergeBlockedByRunningAgents(plan.running_agents)

    if start_at_step_index < 0:
        raise ValueError("start_at_step_index must be >= 0")
    if start_at_step_index > len(plan.steps):
        raise ValueError("start_at_step_index is out of range")

    for step_index, step in enumerate(
        plan.steps[start_at_step_index:], start_at_step_index
    ):
        payload: dict[str, object] = {
            "run_id": plan.run_id,
            "node_id": step.node_id,
            "task_id": step.task_id,
            "kind": step.kind,
            "branch_name": step.branch_name,
        }
        if emit_event is not None:
            await emit_event({**payload, "phase": "started"})
        if update_run is not None:
            await update_run(
                MergeRunStepUpdate(step_index=step_index, step=step, phase="started")
            )
        try:
            if step.kind == "rebase":
                assert step.upstream_ref is not None
                git_rebase(step.worktree_path, step.upstream_ref)
            elif step.kind == "merge_ff":
                git_merge_ff_only(step.worktree_path, step.branch_name)
            else:
                raise RuntimeError(f"Unknown merge step kind: {step.kind!r}")
        except GitCommandError as e:
            blocked = step.kind == "rebase" and git_has_in_progress_operation(
                step.worktree_path
            )
            if update_run is not None:
                await update_run(
                    MergeRunStepUpdate(
                        step_index=step_index,
                        step=step,
                        phase="failed",
                        blocked=blocked,
                        error=str(e),
                    )
                )
            if emit_event is not None:
                await emit_event({**payload, "phase": "failed", "error": str(e)})
            raise
        if emit_event is not None:
            await emit_event({**payload, "phase": "finished"})
        if update_run is not None:
            await update_run(
                MergeRunStepUpdate(step_index=step_index, step=step, phase="finished")
            )

    async with sessionmaker() as session:
        # Mark the merged task's ancestors as complete (best-effort), and clear merge-ready.
        # We intentionally keep this scoped to ancestors because branch refs for unrelated
        # nodes may not reflect updated history in worktree-heavy workflows.
        #
        # Note: plan.spine_node_ids is already root->leaf order.
        rows = list(
            await session.scalars(
                select(Task).where(Task.node_id.in_(plan.spine_node_ids))
            )
        )
        for task_row in rows:
            if task_row.state != TaskState.Done:
                task_row.state = TaskState.Done
            if task_row.merge_ready_at is not None:
                task_row.merge_ready_at = None

        # Also reflect tasks whose branch is now included in the base branch history.
        result = await session.execute(
            select(Task, Node)
            .join(Node, Task.node_id == Node.id)
            .where(Task.epic_id == plan.epic_id)
        )
        for task_row, node_row in result.all():
            if task_row.state == TaskState.Done:
                continue
            if not git_is_ancestor(
                ctx.repo_root, node_row.branch_name, plan.base_branch
            ):
                continue
            task_row.state = TaskState.Done
            if task_row.merge_ready_at is not None:
                task_row.merge_ready_at = None

        await session.commit()


def format_running_agents_confirmation(
    agents: Sequence[RunningAgentInfo],
) -> str:
    lines = ["This operation affects running tasks/agents:"]
    for info in agents:
        label = f"{info.branch_name} ({info.agent_name}, {info.agent_status})"
        if info.task_id is not None:
            label = f"T-{info.task_id} {label}"
        lines.append(f"- {label}")
    lines.append("")
    lines.append("Proceed anyway?")
    return "\n".join(lines)


def format_merge_plan(plan: MergeCascadePlan) -> str:
    lines: list[str] = []
    lines.append(f"Epic base: {plan.base_branch} ({plan.base_worktree})")
    lines.append(f"Scope: {plan.scope}")
    lines.append(f"Spine: {len(plan.spine_node_ids)} node(s)")
    lines.append(f"Affected: {len(plan.affected_node_ids)} node(s)")
    lines.append("")
    for step in plan.steps:
        if step.kind == "rebase":
            lines.append(
                f"rebase {step.branch_name} on {step.upstream_ref} ({step.worktree_path})"
            )
        else:
            lines.append(
                f"merge --ff-only {step.branch_name} -> {plan.base_branch} ({plan.base_worktree})"
            )
    return "\n".join(lines)
