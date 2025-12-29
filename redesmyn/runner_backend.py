from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from redesmyn.agent_runtime import (
    StartAgentResult,
    restart_task_agent,
    start_task_agent,
    stop_task_agent,
)
from redesmyn.context import RepoContext

RunnerMode = Literal["local", "remote"]


class RunnerBackend(Protocol):
    mode: RunnerMode

    async def start_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str,
        detach: bool,
    ) -> StartAgentResult: ...

    async def stop_task_agent(
        self,
        *,
        task_id: int,
    ) -> bool: ...

    async def restart_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str | None,
        detach: bool,
    ) -> StartAgentResult: ...


class RunnerBackendError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class LocalRunnerBackend:
    """Runner backend for local-dev where API host == runner host."""

    ctx: RepoContext
    mode: RunnerMode = "local"

    async def start_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str,
        detach: bool,
    ) -> StartAgentResult:
        return await start_task_agent(
            self.ctx,
            task_id=task_id,
            harness_command=harness_command,
            detach=detach,
        )

    async def stop_task_agent(self, *, task_id: int) -> bool:
        return await stop_task_agent(self.ctx, task_id=task_id)

    async def restart_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str | None,
        detach: bool,
    ) -> StartAgentResult:
        return await restart_task_agent(
            self.ctx,
            task_id=task_id,
            harness_command=harness_command,
            detach=detach,
        )


@dataclass(frozen=True, slots=True)
class RemoteRunnerBackend:
    """Runner backend for cloud/remote mode (daemon executes commands).

    Not implemented yet. This placeholder exists so API handlers can be written
    against a stable interface, and the implementation can be swapped in once
    the daemon/control-plane split lands.
    """

    mode: RunnerMode = "remote"

    async def start_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str,
        detach: bool,
    ) -> StartAgentResult:
        raise RunnerBackendError(
            "Runner backend is remote; start via the daemon is not implemented yet.",
            status_code=501,
        )

    async def stop_task_agent(self, *, task_id: int) -> bool:
        raise RunnerBackendError(
            "Runner backend is remote; stop via the daemon is not implemented yet.",
            status_code=501,
        )

    async def restart_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str | None,
        detach: bool,
    ) -> StartAgentResult:
        raise RunnerBackendError(
            "Runner backend is remote; restart via the daemon is not implemented yet.",
            status_code=501,
        )


def make_runner_backend(*, mode: RunnerMode, ctx: RepoContext) -> RunnerBackend:
    if mode == "local":
        return LocalRunnerBackend(ctx=ctx)
    if mode == "remote":
        return RemoteRunnerBackend()
    raise ValueError(f"Unknown runner mode: {mode!r}")
