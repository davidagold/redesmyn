from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal
from urllib.parse import urlencode, urlparse, urlunparse

import httpx
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.agent_runtime import has_tmux
from redesmyn.context import RepoContext
from redesmyn.db import (
    DatabaseMigrationRequiredError,
    DatabaseNotInitializedError,
    create_engine,
    create_sessionmaker,
    init_db,
)
from redesmyn.db.models import HostCapabilities
from redesmyn.domain.enums import AgentStatus, CommandState, MergeRunStatus
from redesmyn.git_mechanics_v0 import (
    MergeBlockedByRunningAgents,
    MergeCascadePlan,
    MergePlanError,
    MergeRunStepUpdate,
    RestackPlan,
    _execute_git_plan_steps,
    build_merge_cascade_plan,
    build_restack_plan,
)
from redesmyn.merge_runs import merge_run_plan_snapshot
from redesmyn.repo import (
    branch_exists,
    git_has_in_progress_operation,
    git_is_ancestor,
)
from redesmyn.repo_observer import RepoObserverState, observe_repo
from redesmyn.schemas.core import (
    EpicGraphResponse,
    EpicResponse,
    MergeRunSummaryResponse,
    TaskResponse,
)
from redesmyn.settings import RedesmynSettings
from redesmyn.host_identity import load_or_create_host_identity
from redesmyn.ws_protocol import (
    DaemonCommandAck,
    DaemonEvent,
    DaemonHeartbeat,
    DaemonHello,
    RepoKey,
    ServerCommand,
)


class DaemonCommandError(RuntimeError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class MergeRunPlanCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: int
    operation: Literal["merge"] = "merge"
    scope: Literal["descendants", "spine"]
    restack_mode: Literal["strict", "merge_then_restack"] = "strict"
    force: bool = False


class RestackPlanCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: int
    operation: Literal["restack"] = "restack"
    scope: Literal["descendants", "spine"]
    # DaemonRepoExecutor.resume_merge_run currently includes these fields even for restack
    # plans; accept them for forwards/backwards compatibility.
    restack_mode: Literal["strict", "merge_then_restack"] = "strict"
    force: bool = False


class MergeRunStartCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: int
    operation: Literal["merge", "restack"]
    scope: Literal["descendants", "spine"]
    restack_mode: Literal["strict", "merge_then_restack"] = "strict"
    allow_running: bool = False
    force: bool = False
    canonical: bool = True


class MergeRunResumeCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    allow_running: bool = False
    canonical: bool = True


@dataclass(slots=True)
class _BlockedMergeRun:
    run_id: str
    epic_id: int
    requested_task_id: int
    operation: Literal["merge", "restack"]
    blocked_step_index: int
    blocked_step_kind: str
    blocked_task_id: int | None
    blocked_branch_name: str | None
    blocked_worktree_path: str
    error: str | None = None
    emitted_resumable: bool = False


@dataclass(slots=True)
class Backoff:
    initial_s: float = 0.5
    maximum_s: float = 30.0
    factor: float = 1.7
    jitter_fraction: float = 0.2
    _current_s: float = 0.5

    def __post_init__(self) -> None:
        self._current_s = self.initial_s

    def reset(self) -> None:
        self._current_s = self.initial_s

    def next_delay_s(self) -> float:
        delay = self._current_s
        self._current_s = min(self.maximum_s, self._current_s * self.factor)
        jitter = delay * self.jitter_fraction
        return max(0.0, delay + random.uniform(-jitter, jitter))


def _ws_url_for_control_plane(*, control_plane_url: str, token: str) -> str:
    parsed = urlparse(control_plane_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"control plane URL must start with http:// or https:// (got {control_plane_url!r})"
        )
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = urlencode({"token": token})
    ws = parsed._replace(
        scheme=scheme,
        path="/v1/daemon/ws",
        query=query,
        fragment="",
    )
    return urlunparse(ws)


def _drop_oldest(queue: asyncio.Queue[dict[str, Any]]) -> None:
    try:
        queue.get_nowait()
    except asyncio.QueueEmpty:
        return


@dataclass(frozen=True, slots=True)
class DaemonRuntimeConfig:
    control_plane_url: str
    token: str
    workspace_id: str
    repo_id: str
    poll_interval_s: float = 1.0
    heartbeat_interval_s: float = 5.0


class ControlPlaneClient:
    def __init__(self, *, control_plane_url: str, timeout_s: float = 5.0) -> None:
        self._client = httpx.AsyncClient(base_url=control_plane_url, timeout=timeout_s)

    async def __aenter__(self) -> "ControlPlaneClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_epics(self) -> list[EpicResponse]:
        resp = await self._client.get("/v1/epics")
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        return [EpicResponse.model_validate(item) for item in payload]

    async def epic_graph(self, epic_slug: str) -> EpicGraphResponse:
        resp = await self._client.get(f"/v1/epics/{epic_slug}/graph")
        resp.raise_for_status()
        return EpicGraphResponse.model_validate(resp.json())

    async def find_merge_run(self, run_id: str) -> MergeRunSummaryResponse | None:
        try:
            epics = await self.list_epics()
        except Exception:
            return None
        for epic in epics:
            try:
                graph = await self.epic_graph(epic.slug)
            except Exception:
                continue
            for run in graph.merge_runs:
                if run.run_id == run_id:
                    return run
        return None


class DaemonRuntime:
    def __init__(
        self,
        ctx: RepoContext,
        settings: RedesmynSettings,
        config: DaemonRuntimeConfig,
    ) -> None:
        self._ctx = ctx
        self._settings = settings
        self._config = config

        host_identity = load_or_create_host_identity(ctx)
        self._daemon_id = host_identity.host_key
        self._host = host_identity.display_name

        self._attached_repos = [
            RepoKey(workspace_id=config.workspace_id, repo_id=config.repo_id)
        ]

        capabilities = HostCapabilities(
            tmux_available=has_tmux(),
            supports_path_shim=True,
        )
        self._capabilities = {
            **capabilities.model_dump(mode="python"),
            "repo_plan_v1": True,
        }

        self._engine = create_engine(ctx.db_path)
        self._sessionmaker: async_sessionmaker[AsyncSession] = create_sessionmaker(
            self._engine
        )
        self._db_ready = False

        self._observer_state = RepoObserverState()
        self._last_stack_in_sync_by_task_id: dict[int, bool | None] = {}
        self._blocked_merge_runs: dict[str, _BlockedMergeRun] = {}

    async def run_forever(self) -> None:
        backoff = Backoff()
        while True:
            try:
                await self._run_connected()
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(backoff.next_delay_s())

    async def _run_connected(self) -> None:
        import websockets

        ws_url = _ws_url_for_control_plane(
            control_plane_url=self._config.control_plane_url,
            token=self._config.token,
        )

        async with ControlPlaneClient(
            control_plane_url=self._config.control_plane_url
        ) as api:
            async with websockets.connect(ws_url, ping_interval=None) as websocket:
                # Treat each websocket session as a fresh "epoch" so we can resync
                # derived projections on reconnect by emitting a baseline.
                self._observer_state = RepoObserverState()
                self._last_stack_in_sync_by_task_id = {}

                hello = DaemonHello(
                    host_key=self._daemon_id,
                    display_name=self._host,
                    capabilities=self._capabilities,
                    attached_repos=self._attached_repos,
                )
                await websocket.send(hello.model_dump_json())

                send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
                command_queue: asyncio.Queue[ServerCommand] = asyncio.Queue(maxsize=128)

                tasks = [
                    asyncio.create_task(self._sender_loop(websocket, send_queue)),
                    asyncio.create_task(self._receiver_loop(websocket, command_queue)),
                    asyncio.create_task(self._heartbeat_loop(send_queue)),
                    asyncio.create_task(self._telemetry_loop(api, send_queue)),
                    asyncio.create_task(
                        self._command_loop(api, command_queue, send_queue)
                    ),
                ]
                try:
                    done, pending = await asyncio.wait(
                        set(tasks), return_when=asyncio.FIRST_EXCEPTION
                    )
                    for task in done:
                        exc = task.exception()
                        if exc is not None:
                            raise exc
                finally:
                    for task in tasks:
                        task.cancel()

    async def _ensure_db_ready(self) -> None:
        if self._db_ready:
            return
        try:
            await init_db(self._engine, migrate=False)
        except (DatabaseNotInitializedError, DatabaseMigrationRequiredError) as e:
            raise DaemonCommandError(str(e)) from e
        self._db_ready = True

    def _require_attached_repo(self, cmd: ServerCommand) -> None:
        workspace_id = cmd.workspace_id
        repo_id = cmd.repo_id
        if not workspace_id or not repo_id:
            raise DaemonCommandError("Command is missing workspace_id/repo_id.")
        if not any(
            r.workspace_id == workspace_id and r.repo_id == repo_id
            for r in self._attached_repos
        ):
            raise DaemonCommandError(
                f"Repo {workspace_id}/{repo_id} is not attached on this daemon."
            )

    async def _emit_resumable_merge_runs(
        self, send_queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        for run_id, run in list(self._blocked_merge_runs.items()):
            if run.emitted_resumable:
                continue
            path = Path(run.blocked_worktree_path)
            if not path.exists():
                continue
            if git_has_in_progress_operation(path):
                continue
            run.emitted_resumable = True
            await self._send_event(
                send_queue,
                event_type="merge.run",
                data={
                    "run_id": run_id,
                    "task_id": run.blocked_task_id or run.requested_task_id,
                    "epic_id": run.epic_id,
                    "requested_task_id": run.requested_task_id,
                    "status": MergeRunStatus.Resumable,
                    "operation": run.operation,
                },
            )

    async def _sender_loop(
        self, websocket: Any, send_queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        while True:
            payload = await send_queue.get()
            await websocket.send(json.dumps(payload))

    async def _receiver_loop(
        self, websocket: Any, command_queue: asyncio.Queue[ServerCommand]
    ) -> None:
        while True:
            raw = await websocket.recv()
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8", errors="replace")
                except Exception:
                    continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "command":
                cmd_raw = msg.get("command")
                if not isinstance(cmd_raw, dict):
                    continue
                try:
                    cmd = ServerCommand.model_validate(cmd_raw)
                except Exception:
                    continue
                if command_queue.full():
                    try:
                        command_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    command_queue.put_nowait(cmd)
                except asyncio.QueueFull:
                    pass

    async def _heartbeat_loop(self, send_queue: asyncio.Queue[dict[str, Any]]) -> None:
        while True:
            heartbeat = DaemonHeartbeat(attached_repos=self._attached_repos)
            await self._enqueue(send_queue, heartbeat.model_dump(mode="python"))
            await asyncio.sleep(self._config.heartbeat_interval_s)

    async def _telemetry_loop(
        self, api: ControlPlaneClient, send_queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        while True:
            try:
                emit_baseline = not self._observer_state.initialized
                await self._emit_repo_telemetry(
                    api, send_queue, emit_baseline=emit_baseline
                )
                await self._emit_resumable_merge_runs(send_queue)
                self._observer_state.initialized = True
            except Exception:
                pass
            await asyncio.sleep(self._config.poll_interval_s)

    async def _emit_repo_telemetry(
        self,
        api: ControlPlaneClient,
        send_queue: asyncio.Queue[dict[str, Any]],
        *,
        emit_baseline: bool,
    ) -> None:
        try:
            epics = await api.list_epics()
        except Exception:
            epics = []

        tasks: list[TaskResponse] = []
        active_agent_id_by_task_id: dict[int, int] = {}
        upstream_by_task_id: dict[int, str | None] = {}

        for epic in epics:
            graph = await api.epic_graph(epic.slug)
            tasks.extend(graph.tasks)
            tasks_by_id = {t.id: t for t in graph.tasks}

            for session in graph.agent_sessions:
                if session.task_id is None:
                    continue
                if (
                    session.status in {AgentStatus.Running, AgentStatus.Blocked}
                    and session.ended_at is None
                ):
                    active_agent_id_by_task_id[session.task_id] = session.agent_id

            for task in graph.tasks:
                if task.parent_task_id is None:
                    upstream_by_task_id[task.id] = epic.root_branch
                else:
                    parent = tasks_by_id.get(task.parent_task_id)
                    upstream_by_task_id[task.id] = (
                        parent.branch_name if parent is not None else None
                    )

        if not tasks:
            return

        events = observe_repo(
            self._ctx,
            tasks,
            self._observer_state,
            emit_baseline=emit_baseline,
            active_agent_id_by_task_id=active_agent_id_by_task_id,
        )
        for event in events:
            await self._send_event(
                send_queue,
                event_type=event.event_type,
                data=dict(event.data),
            )

        missing: object = object()
        for task in tasks:
            upstream = upstream_by_task_id.get(task.id)
            value: bool | None
            if upstream is None:
                value = None
            elif task.branch_name is None:
                value = None
            elif not branch_exists(self._ctx.repo_root, task.branch_name):
                value = None
            elif not branch_exists(self._ctx.repo_root, upstream):
                value = None
            else:
                value = git_is_ancestor(self._ctx.repo_root, upstream, task.branch_name)

            prev = self._last_stack_in_sync_by_task_id.get(task.id, missing)
            if prev is not missing and prev == value:
                continue
            self._last_stack_in_sync_by_task_id[task.id] = value
            if prev is missing and value is None and not emit_baseline:
                continue
            await self._send_event(
                send_queue,
                event_type="node.stack_in_sync",
                data={
                    "task_id": task.id,
                    "branch_name": task.branch_name,
                    "upstream_ref": upstream,
                    "stack_in_sync": value,
                },
            )

    def _spawn_background(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)

        def _consume(task: asyncio.Task[None]) -> None:
            try:
                task.exception()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        task.add_done_callback(_consume)

    async def _run_merge_run_plan(
        self,
        send_queue: asyncio.Queue[dict[str, Any]],
        *,
        run_id: str,
        requested_task_id: int,
        epic_id: int,
        operation: Literal["merge", "restack"],
        allow_running: bool,
        plan: MergeCascadePlan | RestackPlan,
        start_at_step_index: int,
    ) -> None:
        self._blocked_merge_runs.pop(run_id, None)

        plan_snapshot = merge_run_plan_snapshot(plan=plan, operation=operation)
        merge_run_task_id = requested_task_id
        if operation == "merge" and isinstance(plan, MergeCascadePlan):
            if plan.spine_task_ids:
                merge_run_task_id = plan.spine_task_ids[-1]

        await self._send_event(
            send_queue,
            event_type="merge.run",
            data={
                "run_id": run_id,
                "task_id": merge_run_task_id,
                "epic_id": epic_id,
                "requested_task_id": requested_task_id,
                "status": MergeRunStatus.Running,
                "operation": operation,
                "plan_snapshot": plan_snapshot,
            },
        )

        failed_update: MergeRunStepUpdate | None = None

        async def emit_task_merge(payload: dict[str, object]) -> None:
            await self._send_event(
                send_queue,
                event_type="task.merge",
                data={
                    "run_id": run_id,
                    "operation": operation,
                    **payload,
                },
            )

        async def update_run(update: MergeRunStepUpdate) -> None:
            nonlocal failed_update
            if update.phase == "failed":
                failed_update = update

        try:
            await _execute_git_plan_steps(
                steps=plan.steps,
                running_agents=plan.running_agents,
                allow_running=allow_running,
                emit_event=emit_task_merge,
                update_run=update_run,
                start_at_step_index=start_at_step_index,
            )
        except MergeBlockedByRunningAgents as exc:
            self._blocked_merge_runs.pop(run_id, None)
            await self._send_event(
                send_queue,
                event_type="merge.run",
                data={
                    "run_id": run_id,
                    "task_id": merge_run_task_id,
                    "epic_id": epic_id,
                    "requested_task_id": requested_task_id,
                    "status": MergeRunStatus.Failed,
                    "operation": operation,
                    "error": str(exc),
                },
            )
            return
        except Exception as exc:
            status = MergeRunStatus.Failed
            payload: dict[str, object] = {
                "run_id": run_id,
                "task_id": merge_run_task_id,
                "epic_id": epic_id,
                "requested_task_id": requested_task_id,
                "status": status,
                "operation": operation,
                "error": str(exc),
            }
            if failed_update is not None:
                status = (
                    MergeRunStatus.Blocked
                    if failed_update.blocked
                    else MergeRunStatus.Failed
                )
                payload["status"] = status
                payload["task_id"] = failed_update.step.task_id
                payload["blocked_step_index"] = failed_update.step_index
                payload["blocked_step_kind"] = failed_update.step.kind
                payload["blocked_task_id"] = failed_update.step.task_id
                payload["blocked_branch_name"] = failed_update.step.branch_name
                payload["blocked_worktree_path"] = str(failed_update.step.worktree_path)
                payload["error"] = failed_update.error or str(exc)

                if status == MergeRunStatus.Blocked:
                    self._blocked_merge_runs[run_id] = _BlockedMergeRun(
                        run_id=run_id,
                        epic_id=epic_id,
                        requested_task_id=requested_task_id,
                        operation=operation,
                        blocked_step_index=failed_update.step_index,
                        blocked_step_kind=failed_update.step.kind,
                        blocked_task_id=failed_update.step.task_id,
                        blocked_branch_name=failed_update.step.branch_name,
                        blocked_worktree_path=str(failed_update.step.worktree_path),
                        error=failed_update.error,
                    )
            else:
                self._blocked_merge_runs.pop(run_id, None)

            await self._send_event(send_queue, event_type="merge.run", data=payload)
            return

        self._blocked_merge_runs.pop(run_id, None)
        await self._send_event(
            send_queue,
            event_type="merge.run",
            data={
                "run_id": run_id,
                "task_id": merge_run_task_id,
                "epic_id": epic_id,
                "requested_task_id": requested_task_id,
                "status": MergeRunStatus.Succeeded,
                "operation": operation,
            },
        )

    async def _handle_merge_run_start(
        self,
        api: ControlPlaneClient,
        send_queue: asyncio.Queue[dict[str, Any]],
        *,
        payload: MergeRunStartCommand,
    ) -> None:
        del api  # not needed for start
        try:
            await self._ensure_db_ready()
            if payload.operation == "restack":
                plan = await build_restack_plan(
                    ctx=self._ctx,
                    sessionmaker=self._sessionmaker,
                    task_id=payload.task_id,
                    run_id=payload.run_id,
                    scope=payload.scope,
                )
            else:
                plan = await build_merge_cascade_plan(
                    ctx=self._ctx,
                    sessionmaker=self._sessionmaker,
                    task_id=payload.task_id,
                    run_id=payload.run_id,
                    scope=payload.scope,
                    restack_mode=payload.restack_mode,
                    force=payload.force,
                )
        except Exception as exc:
            await self._send_event(
                send_queue,
                event_type="merge.run",
                data={
                    "run_id": payload.run_id,
                    "status": MergeRunStatus.Failed,
                    "operation": payload.operation,
                    "error": str(exc),
                },
            )
            return

        await self._run_merge_run_plan(
            send_queue,
            run_id=payload.run_id,
            requested_task_id=payload.task_id,
            epic_id=plan.epic_id,
            operation=payload.operation,
            allow_running=payload.allow_running,
            plan=plan,
            start_at_step_index=0,
        )

    async def _handle_merge_run_resume(
        self,
        api: ControlPlaneClient,
        send_queue: asyncio.Queue[dict[str, Any]],
        *,
        payload: MergeRunResumeCommand,
    ) -> None:
        run = await api.find_merge_run(payload.run_id)
        if run is None:
            await self._send_event(
                send_queue,
                event_type="merge.run",
                data={
                    "run_id": payload.run_id,
                    "status": MergeRunStatus.Failed,
                    "error": "Merge run not found.",
                },
            )
            return

        try:
            await self._ensure_db_ready()
            if run.operation == "restack":
                plan = await build_restack_plan(
                    ctx=self._ctx,
                    sessionmaker=self._sessionmaker,
                    task_id=run.requested_task_id,
                    run_id=run.run_id,
                    scope=run.scope,
                )
            else:
                plan = await build_merge_cascade_plan(
                    ctx=self._ctx,
                    sessionmaker=self._sessionmaker,
                    task_id=run.requested_task_id,
                    run_id=run.run_id,
                    scope=run.scope,
                    restack_mode=run.restack_mode,
                    force=run.force,
                )
        except Exception as exc:
            await self._send_event(
                send_queue,
                event_type="merge.run",
                data={
                    "run_id": run.run_id,
                    "status": MergeRunStatus.Failed,
                    "operation": run.operation,
                    "error": str(exc),
                },
            )
            return

        start_at_step_index = run.blocked_step_index or 0
        if start_at_step_index >= len(plan.steps):
            start_at_step_index = 0
        if start_at_step_index and run.blocked_step_kind and run.blocked_branch_name:
            step = plan.steps[start_at_step_index]
            if (
                step.kind != run.blocked_step_kind
                or step.branch_name != run.blocked_branch_name
            ):
                start_at_step_index = 0

        await self._run_merge_run_plan(
            send_queue,
            run_id=run.run_id,
            requested_task_id=run.requested_task_id,
            epic_id=run.epic_id,
            operation=run.operation,
            allow_running=payload.allow_running,
            plan=plan,
            start_at_step_index=start_at_step_index,
        )

    async def _command_loop(
        self,
        api: ControlPlaneClient,
        command_queue: asyncio.Queue[ServerCommand],
        send_queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        while True:
            cmd = await command_queue.get()
            await self._ack_command(send_queue, cmd.command_id, CommandState.Running)
            try:
                data = await self._execute_command(cmd, api, send_queue)
            except DaemonCommandError as exc:
                await self._ack_command(
                    send_queue,
                    cmd.command_id,
                    CommandState.Failed,
                    data={"detail": exc.detail},
                )
                continue
            except Exception as exc:
                await self._ack_command(
                    send_queue,
                    cmd.command_id,
                    CommandState.Failed,
                    data={"detail": str(exc)},
                )
                continue
            await self._ack_command(
                send_queue,
                cmd.command_id,
                CommandState.Succeeded,
                data=data,
            )

    async def _execute_command(
        self,
        cmd: ServerCommand,
        api: ControlPlaneClient,
        send_queue: asyncio.Queue[dict[str, Any]],
    ) -> dict[str, Any]:
        if cmd.command_type == "daemon.ping":
            return {}

        if cmd.command_type == "daemon.emit":
            await self._send_event(
                send_queue,
                event_type="daemon.event",
                data=cmd.data,
            )
            return {}

        if cmd.command_type == "repo.merge_run.plan":
            self._require_attached_repo(cmd)
            payload = MergeRunPlanCommand.model_validate(cmd.data)
            await self._ensure_db_ready()
            try:
                plan = await build_merge_cascade_plan(
                    ctx=self._ctx,
                    sessionmaker=self._sessionmaker,
                    task_id=payload.task_id,
                    run_id=payload.run_id,
                    scope=payload.scope,
                    restack_mode=payload.restack_mode,
                    force=payload.force,
                )
            except MergePlanError as e:
                raise DaemonCommandError(str(e)) from e

            plan_snapshot = merge_run_plan_snapshot(plan=plan, operation="merge")
            return {
                "plan_snapshot": plan_snapshot,
                "running_agents": bool(plan.running_agents),
            }

        if cmd.command_type == "repo.restack.plan":
            self._require_attached_repo(cmd)
            payload = RestackPlanCommand.model_validate(cmd.data)
            await self._ensure_db_ready()
            try:
                plan = await build_restack_plan(
                    ctx=self._ctx,
                    sessionmaker=self._sessionmaker,
                    task_id=payload.task_id,
                    run_id=payload.run_id,
                    scope=payload.scope,
                )
            except MergePlanError as e:
                raise DaemonCommandError(str(e)) from e

            plan_snapshot = merge_run_plan_snapshot(plan=plan, operation="restack")
            return {
                "plan_snapshot": plan_snapshot,
                "running_agents": bool(plan.running_agents),
            }

        if cmd.command_type == "repo.merge_run.start":
            self._require_attached_repo(cmd)
            payload = MergeRunStartCommand.model_validate(cmd.data)
            self._spawn_background(
                self._handle_merge_run_start(api, send_queue, payload=payload)
            )
            return {}

        if cmd.command_type == "repo.merge_run.resume":
            self._require_attached_repo(cmd)
            payload = MergeRunResumeCommand.model_validate(cmd.data)
            self._spawn_background(
                self._handle_merge_run_resume(api, send_queue, payload=payload)
            )
            return {}

        raise ValueError(f"Unknown command_type: {cmd.command_type!r}")

    async def _ack_command(
        self,
        send_queue: asyncio.Queue[dict[str, Any]],
        command_id: int,
        state: CommandState,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        ack = DaemonCommandAck(
            command_id=command_id,
            state=state,
            data=data or {},
        )
        await self._enqueue(send_queue, ack.model_dump(mode="python"))

    async def _send_event(
        self,
        send_queue: asyncio.Queue[dict[str, Any]],
        *,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        event = DaemonEvent(
            workspace_id=self._config.workspace_id,
            repo_id=self._config.repo_id,
            event_type=event_type,
            data=data,
        )
        await self._enqueue(send_queue, event.model_dump(mode="python"))

    async def _enqueue(
        self, send_queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]
    ) -> None:
        if send_queue.full():
            _drop_oldest(send_queue)
        try:
            send_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def run_daemon(
    ctx: RepoContext, settings: RedesmynSettings, config: DaemonRuntimeConfig
) -> None:
    daemon = DaemonRuntime(ctx, settings, config)
    await daemon.run_forever()
