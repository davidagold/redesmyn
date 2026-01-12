from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from redesmyn.domain.enums import AgentKindSelection
from redesmyn.agent_runtime import (
    StartAgentResult,
    restart_task_agent,
    run_task_agent_resume_by_id_turn,
    send_task_agent_text,
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
        agent_kind_selection: AgentKindSelection,
        detach: bool,
        prelude_override: str | None = None,
        initial_prompt: str | None = None,
    ) -> StartAgentResult: ...

    async def stop_task_agent(
        self,
        *,
        task_id: int,
    ) -> bool: ...

    async def send_task_agent_text(
        self,
        *,
        task_id: int,
        text: str,
        submit: bool = True,
        interrupt: bool = False,
    ) -> None: ...

    async def restart_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str | None,
        agent_kind_selection_override: AgentKindSelection | None,
        default_agent_kind_selection: AgentKindSelection,
        detach: bool,
        prelude_override: str | None = None,
    ) -> StartAgentResult: ...

    async def run_task_agent_resume_by_id_turn(
        self,
        *,
        task_id: int,
        prompt: str,
        detach: bool,
        idempotency_key: str | None = None,
        resume_session_id: int | None = None,
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
        agent_kind_selection: AgentKindSelection,
        detach: bool,
        prelude_override: str | None = None,
        initial_prompt: str | None = None,
    ) -> StartAgentResult:
        return await start_task_agent(
            self.ctx,
            task_id=task_id,
            harness_command=harness_command,
            detach=detach,
            prelude_override=prelude_override,
            agent_kind_selection=agent_kind_selection,
            initial_prompt=initial_prompt,
        )

    async def stop_task_agent(self, *, task_id: int) -> bool:
        return await stop_task_agent(self.ctx, task_id=task_id)

    async def send_task_agent_text(
        self,
        *,
        task_id: int,
        text: str,
        submit: bool = True,
        interrupt: bool = False,
    ) -> None:
        await send_task_agent_text(
            self.ctx,
            task_id=task_id,
            text=text,
            submit=submit,
            interrupt=interrupt,
        )

    async def restart_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str | None,
        agent_kind_selection_override: AgentKindSelection | None,
        default_agent_kind_selection: AgentKindSelection,
        detach: bool,
        prelude_override: str | None = None,
    ) -> StartAgentResult:
        return await restart_task_agent(
            self.ctx,
            task_id=task_id,
            harness_command=harness_command,
            detach=detach,
            prelude_override=prelude_override,
            agent_kind_selection_override=agent_kind_selection_override,
            default_agent_kind_selection=default_agent_kind_selection,
        )

    async def run_task_agent_resume_by_id_turn(
        self,
        *,
        task_id: int,
        prompt: str,
        detach: bool,
        idempotency_key: str | None = None,
        resume_session_id: int | None = None,
    ) -> StartAgentResult:
        return await run_task_agent_resume_by_id_turn(
            self.ctx,
            task_id=task_id,
            prompt=prompt,
            detach=detach,
            idempotency_key=idempotency_key,
            resume_session_id=resume_session_id,
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
        agent_kind_selection: AgentKindSelection,
        detach: bool,
        prelude_override: str | None = None,
        initial_prompt: str | None = None,
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

    async def send_task_agent_text(
        self,
        *,
        task_id: int,
        text: str,
        submit: bool = True,
        interrupt: bool = False,
    ) -> None:
        raise RunnerBackendError(
            "Runner backend is remote; interactive messaging via the daemon is not implemented yet.",
            status_code=501,
        )

    async def restart_task_agent(
        self,
        *,
        task_id: int,
        harness_command: str | None,
        agent_kind_selection_override: AgentKindSelection | None,
        default_agent_kind_selection: AgentKindSelection,
        detach: bool,
        prelude_override: str | None = None,
    ) -> StartAgentResult:
        raise RunnerBackendError(
            "Runner backend is remote; restart via the daemon is not implemented yet.",
            status_code=501,
        )

    async def run_task_agent_resume_by_id_turn(
        self,
        *,
        task_id: int,
        prompt: str,
        detach: bool,
        idempotency_key: str | None = None,
        resume_session_id: int | None = None,
    ) -> StartAgentResult:
        raise RunnerBackendError(
            "Runner backend is remote; resume-by-id turns via the daemon are not implemented yet.",
            status_code=501,
        )


def make_runner_backend(*, mode: RunnerMode, ctx: RepoContext) -> RunnerBackend:
    if mode == "local":
        return LocalRunnerBackend(ctx=ctx)
    if mode == "remote":
        return RemoteRunnerBackend()
    raise ValueError(f"Unknown runner mode: {mode!r}")
