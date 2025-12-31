from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.context import RepoContext
from redesmyn.db import Agent, Epic, Task
from redesmyn.domain.enums import AgentStatus, TaskState
from redesmyn.repo import (
    GitCommandError,
    current_branch,
    git_has_in_progress_operation,
    git_merge_ff_only,
    git_rebase,
    git_status_porcelain,
    git_worktree_path_for_branch,
)

MergeCascadeScope = Literal["descendants", "spine"]
MergeRestackMode = Literal["strict", "merge_then_restack"]


class MergePlanError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunningAgentInfo:
    task_id: int
    branch_name: str
    agent_id: int
    agent_name: str
    agent_status: AgentStatus


class MergeBlockedByRunningAgents(MergePlanError):
    def __init__(self, agents: Sequence[RunningAgentInfo]):
        super().__init__("Merge affects running tasks; confirmation required.")
        self.agents = list(agents)


@dataclass(frozen=True, slots=True)
class MergeTaskInfo:
    task_id: int
    parent_task_id: int | None
    branch_name: str
    upstream_ref: str
    worktree_path: Path


MergeStepKind = Literal["rebase", "merge_ff"]


@dataclass(frozen=True, slots=True)
class MergePlanStep:
    kind: MergeStepKind
    task_id: int
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
    restack_mode: MergeRestackMode
    spine_task_ids: tuple[int, ...]
    affected_task_ids: tuple[int, ...]
    tasks: dict[int, MergeTaskInfo]
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
) -> tuple[Task, Epic]:
    task = await session.get(Task, task_id)
    if task is None:
        raise MergePlanError(f"Unknown task id: {task_id}")
    if task.branch_name is None:
        raise MergePlanError(
            "Task has no branch backing (branch_name is null). "
            "Run `rn sync --from local --create-branches`."
        )
    epic = await session.get(Epic, task.epic_id)
    if epic is None:
        raise MergePlanError(f"Epic not found: {task.epic_id}")
    return task, epic


def _compute_spine_task_ids(
    *, tasks_by_id: dict[int, Task], leaf_task_id: int
) -> list[int]:
    spine: list[int] = []
    cursor: int | None = leaf_task_id
    while cursor is not None:
        spine.append(cursor)
        task = tasks_by_id.get(cursor)
        if task is None:
            break
        cursor = task.parent_task_id
    spine.reverse()
    return spine


def _compute_descendants(
    *, children_by_parent: dict[int | None, list[int]], start_task_ids: Sequence[int]
) -> set[int]:
    seen: set[int] = set()
    queue: list[int] = list(start_task_ids)
    while queue:
        task_id = queue.pop()
        if task_id in seen:
            continue
        seen.add(task_id)
        for child in children_by_parent.get(task_id, []):
            queue.append(child)
    return seen


def _topo_sort_by_depth(
    *, tasks_by_id: dict[int, Task], task_ids: set[int]
) -> list[int]:
    depth_cache: dict[int, int] = {}

    def depth(task_id: int) -> int:
        if task_id in depth_cache:
            return depth_cache[task_id]
        task = tasks_by_id.get(task_id)
        if task is None or task.parent_task_id is None:
            depth_cache[task_id] = 0
            return 0
        value = depth(task.parent_task_id) + 1
        depth_cache[task_id] = value
        return value

    return sorted(task_ids, key=lambda tid: (depth(tid), tid))


async def build_merge_cascade_plan(
    *,
    ctx: RepoContext,
    sessionmaker: async_sessionmaker[AsyncSession],
    task_id: int,
    run_id: str,
    scope: MergeCascadeScope,
    restack_mode: MergeRestackMode,
    force: bool,
) -> MergeCascadePlan:
    async with sessionmaker() as session:
        task, epic = await _load_task_node_epic(session, task_id=task_id)
        base_branch = epic.root_branch

        tasks = list(
            await session.scalars(select(Task).where(Task.epic_id == epic.id))
        )
        tasks_by_id: dict[int, Task] = {t.id: t for t in tasks}
        children_by_parent: dict[int | None, list[int]] = {}
        for t in tasks:
            if t.branch_name is None:
                continue
            children_by_parent.setdefault(t.parent_task_id, []).append(t.id)
        for value in children_by_parent.values():
            value.sort()

        spine_task_ids = _compute_spine_task_ids(
            tasks_by_id=tasks_by_id,
            leaf_task_id=task.id,
        )

        affected_ids: set[int] = set(spine_task_ids)
        if scope == "descendants":
            affected_ids = _compute_descendants(
                children_by_parent=children_by_parent, start_task_ids=spine_task_ids
            )

        ordered_task_ids = _topo_sort_by_depth(
            tasks_by_id=tasks_by_id,
            task_ids=affected_ids,
        )

        base_worktree = git_worktree_path_for_branch(ctx.repo_root, base_branch)
        if base_worktree is None:
            raise MergePlanError(
                f"No worktree has {base_branch!r} checked out. "
                f"Check out {base_branch} in a worktree and retry."
            )

        # Merge-ready gating (spine).
        if not force:
            missing_ready: list[str] = []
            for spine_task_id in spine_task_ids:
                spine_task = tasks_by_id.get(spine_task_id)
                if spine_task is None:
                    continue
                if spine_task.state == TaskState.Done:
                    continue
                if spine_task.merge_ready_at is None:
                    missing_ready.append(
                        f"{spine_task.id} ({spine_task.branch_name or 'no-branch'})"
                    )
            if missing_ready:
                raise MergePlanError(
                    "Task(s) on the merge spine are not marked ready: "
                    + ", ".join(missing_ready)
                )

        active_task_by_id: dict[int, Task] = {
            t.id: t
            for t in tasks_by_id.values()
            if t.branch_name is not None and t.state in {TaskState.InProgress, TaskState.Blocked}
        }

        def resolve_worktree_path(task_row: Task) -> Path | None:
            if task_row.worktree_path:
                candidate = Path(task_row.worktree_path)
                if candidate.exists():
                    return candidate
            if task_row.branch_name is None:
                return None
            return git_worktree_path_for_branch(ctx.repo_root, task_row.branch_name)

        spine_set = set(spine_task_ids)

        # Decide which tasks to rebase based on worktree presence and state.
        task_infos: dict[int, MergeTaskInfo] = {}
        missing_spine: list[str] = []
        missing_active_descendants: list[str] = []
        for candidate_task_id in ordered_task_ids:
            task_row = tasks_by_id.get(candidate_task_id)
            if task_row is None or task_row.branch_name is None:
                continue

            worktree_path = resolve_worktree_path(task_row)
            parent_task = (
                tasks_by_id.get(task_row.parent_task_id)
                if task_row.parent_task_id is not None
                else None
            )
            upstream_ref = (
                parent_task.branch_name
                if parent_task is not None and parent_task.branch_name is not None
                else base_branch
            )

            if candidate_task_id in spine_set:
                if worktree_path is None:
                    missing_spine.append(f"T-{task_row.id} {task_row.branch_name}")
                    continue
                task_infos[candidate_task_id] = MergeTaskInfo(
                    task_id=task_row.id,
                    parent_task_id=task_row.parent_task_id,
                    branch_name=task_row.branch_name,
                    upstream_ref=upstream_ref,
                    worktree_path=worktree_path,
                )
                continue

            # Descendants: do not create worktrees. Only restack tasks that already have them.
            active_task = active_task_by_id.get(candidate_task_id)
            if active_task is not None and worktree_path is None:
                missing_active_descendants.append(
                    f"T-{task_row.id} {task_row.branch_name} ({active_task.state})"
                )
                continue

            if worktree_path is None:
                continue

            task_infos[candidate_task_id] = MergeTaskInfo(
                task_id=task_row.id,
                parent_task_id=task_row.parent_task_id,
                branch_name=task_row.branch_name,
                upstream_ref=upstream_ref,
                worktree_path=worktree_path,
            )

        if missing_spine:
            raise MergePlanError(
                "Missing worktree(s) for merge spine branch(es): "
                + ", ".join(missing_spine)
            )
        if missing_active_descendants:
            raise MergePlanError(
                "Descendant task(s) are active but have no worktree: "
                + ", ".join(missing_active_descendants)
            )

        # Running-agent detection for affected set.
        running_agents: list[RunningAgentInfo] = []
        agent_ids = [
            tasks_by_id[tid].agent_id
            for tid in task_infos
            if tid in tasks_by_id and tasks_by_id[tid].agent_id is not None
        ]
        agents_by_id: dict[int, Agent] = {}
        if agent_ids:
            rows = list(
                await session.scalars(select(Agent).where(Agent.id.in_(agent_ids)))
            )
            agents_by_id = {a.id: a for a in rows}

        for plan_task_id in task_infos:
            task_row = tasks_by_id.get(plan_task_id)
            if task_row is None or task_row.agent_id is None:
                continue
            agent = agents_by_id.get(task_row.agent_id)
            if agent is None:
                continue
            if agent.status not in {AgentStatus.Running, AgentStatus.Blocked}:
                continue
            running_agents.append(
                RunningAgentInfo(
                    task_id=plan_task_id,
                    branch_name=task_row.branch_name or "unknown",
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

        for plan_task_id in ordered_task_ids:
            info = task_infos.get(plan_task_id)
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
    if scope == "descendants" and restack_mode == "merge_then_restack":
        for spine_task_id in spine_task_ids:
            info = task_infos.get(spine_task_id)
            if info is None:
                continue
            steps.append(
                MergePlanStep(
                    kind="rebase",
                    task_id=info.task_id,
                    branch_name=info.branch_name,
                    worktree_path=info.worktree_path,
                    upstream_ref=info.upstream_ref,
                )
            )
    else:
        for plan_task_id in ordered_task_ids:
            info = task_infos.get(plan_task_id)
            if info is None:
                continue
            steps.append(
                MergePlanStep(
                    kind="rebase",
                    task_id=info.task_id,
                    branch_name=info.branch_name,
                    worktree_path=info.worktree_path,
                    upstream_ref=info.upstream_ref,
                )
            )

    for spine_task_id in spine_task_ids:
        info = task_infos.get(spine_task_id)
        if info is None:
            continue
        steps.append(
            MergePlanStep(
                kind="merge_ff",
                task_id=info.task_id,
                branch_name=info.branch_name,
                worktree_path=base_worktree,
                base_branch=base_branch,
            )
        )

    if scope == "descendants" and restack_mode == "merge_then_restack":
        for plan_task_id in ordered_task_ids:
            if plan_task_id in spine_set:
                continue
            info = task_infos.get(plan_task_id)
            if info is None:
                continue
            steps.append(
                MergePlanStep(
                    kind="rebase",
                    task_id=info.task_id,
                    branch_name=info.branch_name,
                    worktree_path=info.worktree_path,
                    upstream_ref=info.upstream_ref,
                )
            )

    return MergeCascadePlan(
        run_id=run_id,
        epic_id=epic.id,
        base_branch=base_branch,
        base_worktree=base_worktree,
        scope=scope,
        restack_mode=restack_mode,
        spine_task_ids=tuple(spine_task_ids),
        affected_task_ids=tuple([tid for tid in ordered_task_ids if tid in task_infos]),
        tasks=task_infos,
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
        # tasks may not reflect updated history in worktree-heavy workflows.
        #
        # Note: plan.spine_task_ids is already root->leaf order.
        rows = list(await session.scalars(select(Task).where(Task.id.in_(plan.spine_task_ids))))
        for task_row in rows:
            if task_row.state != TaskState.Done:
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
    lines.append(f"Restack mode: {plan.restack_mode}")
    lines.append(f"Spine: {len(plan.spine_task_ids)} task(s)")
    lines.append(f"Affected: {len(plan.affected_task_ids)} task(s)")
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
