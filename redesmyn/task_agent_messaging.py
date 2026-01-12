from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from redesmyn.agent_interface.v0 import AgentSemanticStatus, ExternalSessionRef
from redesmyn.agent_kind import resolve_agent_kind
from redesmyn.agent_runtime import infer_interface_mode_from_argv
from redesmyn.context import RepoContext
from redesmyn.db import AgentSession, Task
from redesmyn.domain.enums import (
    AgentInterfaceMode,
    AgentKind,
    AgentStatus,
    AgentTurnState,
    TaskAgentMessageConflictAction,
)
from redesmyn.orchestration_config import load_orchestration_defaults
from redesmyn.runner_backend import RunnerBackend, RunnerBackendError

_EXTERNAL_SESSION_REF_ADAPTER = TypeAdapter(ExternalSessionRef)
_SEMANTIC_STATUS_ADAPTER = TypeAdapter(AgentSemanticStatus)


class TaskAgentMessageError(RuntimeError):
    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


TaskAgentMessageDelivery = Literal[
    "structured_started",
    "structured_resumed",
    "interactive_started",
    "interactive_sent",
]


@dataclass(frozen=True, slots=True)
class TaskAgentMessageResult:
    agent_session_id: int
    agent_interface_mode: AgentInterfaceMode
    delivery: TaskAgentMessageDelivery
    warnings: tuple[str, ...] = ()


def _effective_on_conflict(
    *,
    on_conflict: TaskAgentMessageConflictAction,
    interrupt: bool | None,
) -> TaskAgentMessageConflictAction:
    if on_conflict != TaskAgentMessageConflictAction.Fail:
        return on_conflict
    if interrupt is True:
        return TaskAgentMessageConflictAction.InterruptTurn
    return TaskAgentMessageConflictAction.Fail


_CONFLICT_PREFIX_TURN_IN_PROGRESS = (
    "[task_agent_message_conflict:structured_turn_in_progress]"
)
_CONFLICT_PREFIX_SESSION_CONFLICT = (
    "[task_agent_message_conflict:structured_session_conflict]"
)


def _external_session_key(
    external_session_ref_raw: object,
) -> tuple[Literal["codex_thread", "claude_session"], str] | None:
    try:
        external_session_ref = _EXTERNAL_SESSION_REF_ADAPTER.validate_python(
            external_session_ref_raw
        )
    except ValidationError:
        return None

    if external_session_ref.type == "codex_thread":
        return ("codex_thread", external_session_ref.thread_id)
    if external_session_ref.type == "claude_session":
        return ("claude_session", external_session_ref.session_id)
    return None


def _key_matches_kind(
    key: tuple[Literal["codex_thread", "claude_session"], str],
    *,
    agent_kind: AgentKind,
) -> bool:
    key_type, _ = key
    match agent_kind:
        case AgentKind.Codex:
            return key_type == "codex_thread"
        case AgentKind.ClaudeCode:
            return key_type == "claude_session"
        case _:
            return False


def _looks_busy(row: AgentSession) -> bool:
    if row.ended_at is not None:
        return False
    if row.status not in {AgentStatus.Running, AgentStatus.Blocked}:
        return False
    try:
        status = _SEMANTIC_STATUS_ADAPTER.validate_python(row.agent_semantic_status)
    except ValidationError:
        return False
    return status.turn_state == AgentTurnState.Busy


def _as_message_error(exc: Exception) -> TaskAgentMessageError:
    if isinstance(exc, TaskAgentMessageError):
        return exc
    if isinstance(exc, RunnerBackendError):
        return TaskAgentMessageError(exc.detail, status_code=exc.status_code)
    message = str(exc)
    if "tmux session is active for this task" in message:
        return TaskAgentMessageError(message, status_code=409)
    return TaskAgentMessageError(message or "Failed to send message.", status_code=400)


async def send_task_agent_message(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    runner_backend: RunnerBackend,
    ctx: RepoContext,
    task_id: int,
    message: str,
    on_conflict: TaskAgentMessageConflictAction,
    interrupt: bool | None,
    preferred_interface_mode: Literal["auto"] | AgentInterfaceMode,
) -> TaskAgentMessageResult:
    trimmed = message.strip()
    if not trimmed:
        raise TaskAgentMessageError("Message is empty.", status_code=400)

    try:
        defaults = load_orchestration_defaults(ctx)
    except RuntimeError as exc:
        raise TaskAgentMessageError(
            "Orchestration defaults are unavailable; configure `harness.command`.",
            status_code=400,
        ) from exc

    harness_command = defaults.harness.command
    if harness_command is None:
        raise TaskAgentMessageError(
            "No harness command configured; set `harness.command` first.",
            status_code=400,
        )

    try:
        argv = shlex.split(harness_command)
    except ValueError as exc:
        raise TaskAgentMessageError(
            f"Invalid harness command: {exc}", status_code=400
        ) from exc
    if not argv:
        raise TaskAgentMessageError(
            "Harness command is empty; set `harness.command` first.",
            status_code=400,
        )

    resolved_agent_kind = resolve_agent_kind(defaults.harness.agent_kind, argv)
    inferred_mode = infer_interface_mode_from_argv(
        argv=argv, resolved_agent_kind=resolved_agent_kind
    )
    desired_mode = (
        inferred_mode
        if preferred_interface_mode == "auto"
        else preferred_interface_mode
    )
    if preferred_interface_mode != "auto" and desired_mode != inferred_mode:
        raise TaskAgentMessageError(
            (
                f"Preferred interface mode {desired_mode.value!r} does not match the configured "
                f"harness command ({inferred_mode.value!r}). Update `harness.command` or use auto."
            ),
            status_code=400,
        )

    if desired_mode == AgentInterfaceMode.Structured and inferred_mode != desired_mode:
        raise TaskAgentMessageError(
            (
                "Structured messaging requires a structured harness command "
                "(e.g. Codex: `codex exec --json ...`; Claude: `claude --print --output-format stream-json ...`)."
            ),
            status_code=400,
        )

    async with sessionmaker() as session:
        task = await session.get(Task, task_id)
        if task is None:
            raise TaskAgentMessageError("Task not found.", status_code=404)

        recent_sessions = list(
            await session.scalars(
                select(AgentSession)
                .where(AgentSession.task_id == task.id)
                .order_by(desc(AgentSession.id))
                .limit(50)
            )
        )

    compatible_sessions = [
        row
        for row in recent_sessions
        if row.agent_kind == resolved_agent_kind
        and row.agent_interface_mode == desired_mode
    ]

    effective_on_conflict = _effective_on_conflict(
        on_conflict=on_conflict, interrupt=interrupt
    )

    if desired_mode == AgentInterfaceMode.Structured:
        active_any = next(
            (
                row
                for row in recent_sessions
                if row.ended_at is None
                and row.status in {AgentStatus.Running, AgentStatus.Blocked}
            ),
            None,
        )
        resumable_session = next(
            (
                row
                for row in compatible_sessions
                if (key := _external_session_key(row.external_session_ref)) is not None
                and _key_matches_kind(key, agent_kind=resolved_agent_kind)
            ),
            None,
        )
        if resumable_session is not None:
            if (
                effective_on_conflict
                == TaskAgentMessageConflictAction.StopSessionAndStartNew
            ):
                raise TaskAgentMessageError(
                    (
                        "Stop-and-start-new is not supported when a resumable structured session exists. "
                        "Use on_conflict=interrupt_turn instead."
                    ),
                    status_code=400,
                )
            external_key = _external_session_key(resumable_session.external_session_ref)
            if external_key is not None:
                in_progress = next(
                    (
                        row
                        for row in recent_sessions
                        if row.agent_interface_mode == AgentInterfaceMode.Structured
                        and row.ended_at is None
                        and _looks_busy(row)
                        and _external_session_key(row.external_session_ref)
                        == external_key
                    ),
                    None,
                )
            else:
                in_progress = None

            if (
                in_progress is not None
                and effective_on_conflict == TaskAgentMessageConflictAction.Fail
            ):
                raise TaskAgentMessageError(
                    (
                        f"{_CONFLICT_PREFIX_TURN_IN_PROGRESS} "
                        "Agent turn in progress. Interrupt the current turn before sending a new structured message."
                    ),
                    status_code=409,
                )
            if (
                in_progress is not None
                and effective_on_conflict
                == TaskAgentMessageConflictAction.InterruptTurn
            ):
                try:
                    await runner_backend.stop_task_agent(task_id=task_id)
                except (RunnerBackendError, RuntimeError) as exc:
                    raise _as_message_error(exc) from exc

            try:
                result = await runner_backend.run_task_agent_resume_by_id_turn(
                    task_id=task_id,
                    prompt=trimmed,
                    detach=defaults.harness.detach,
                    idempotency_key=None,
                    resume_session_id=resumable_session.id,
                )
            except (RunnerBackendError, RuntimeError) as exc:
                raise _as_message_error(exc) from exc
            return TaskAgentMessageResult(
                agent_session_id=result.agent_session.id,
                agent_interface_mode=AgentInterfaceMode.Structured,
                delivery="structured_resumed",
                warnings=result.warnings,
            )

        if (
            active_any is not None
            and effective_on_conflict == TaskAgentMessageConflictAction.InterruptTurn
        ):
            raise TaskAgentMessageError(
                (
                    "Cannot interrupt a structured turn when no resumable structured session exists. "
                    "Use on_conflict=stop_session_and_start_new instead."
                ),
                status_code=400,
            )
        if (
            active_any is not None
            and effective_on_conflict == TaskAgentMessageConflictAction.Fail
        ):
            raise TaskAgentMessageError(
                (
                    f"{_CONFLICT_PREFIX_SESSION_CONFLICT} "
                    "A (non-resumable) agent session is running for this task. "
                    "Stop it and start a new structured session to send this message?"
                ),
                status_code=409,
            )
        if (
            active_any is not None
            and effective_on_conflict
            == TaskAgentMessageConflictAction.StopSessionAndStartNew
        ):
            try:
                await runner_backend.stop_task_agent(task_id=task_id)
            except (RunnerBackendError, RuntimeError) as exc:
                raise _as_message_error(exc) from exc

        try:
            result = await runner_backend.start_task_agent(
                task_id=task_id,
                harness_command=harness_command,
                agent_kind_selection=defaults.harness.agent_kind,
                detach=defaults.harness.detach,
                prelude_override=None,
                initial_prompt=trimmed,
            )
        except (RunnerBackendError, RuntimeError, ValueError) as exc:
            raise _as_message_error(exc) from exc
        if not result.started:
            session_row = result.agent_session
            if session_row.ended_at is not None:
                if (
                    session_row.status == AgentStatus.Stopped
                    and session_row.exit_code == 0
                ):
                    return TaskAgentMessageResult(
                        agent_session_id=session_row.id,
                        agent_interface_mode=session_row.agent_interface_mode,
                        delivery="structured_started",
                        warnings=result.warnings,
                    )
                raise TaskAgentMessageError(
                    (
                        result.warnings[0]
                        if result.warnings
                        else "Structured harness exited immediately."
                    ),
                    status_code=400,
                )

            raise TaskAgentMessageError(
                (
                    f"{_CONFLICT_PREFIX_SESSION_CONFLICT} "
                    "An agent session is currently running for this task. "
                    "Stop it and start a new structured session to send this message?"
                ),
                status_code=409,
            )
        return TaskAgentMessageResult(
            agent_session_id=result.agent_session.id,
            agent_interface_mode=result.agent_session.agent_interface_mode,
            delivery="structured_started",
            warnings=result.warnings,
        )

    active_interactive = next(
        (
            row
            for row in compatible_sessions
            if row.ended_at is None
            and row.status in {AgentStatus.Running, AgentStatus.Blocked}
        ),
        None,
    )
    active_any = next(
        (
            row
            for row in recent_sessions
            if row.ended_at is None
            and row.status in {AgentStatus.Running, AgentStatus.Blocked}
        ),
        None,
    )

    if active_interactive is None and active_any is not None:
        if active_any.agent_interface_mode == AgentInterfaceMode.Structured:
            if (
                effective_on_conflict
                == TaskAgentMessageConflictAction.StopSessionAndStartNew
            ):
                try:
                    await runner_backend.stop_task_agent(task_id=task_id)
                except (RunnerBackendError, RuntimeError) as exc:
                    raise _as_message_error(exc) from exc
                active_any = None
            else:
                raise TaskAgentMessageError(
                    (
                        f"{_CONFLICT_PREFIX_SESSION_CONFLICT} "
                        "An incompatible structured agent session is currently running for this task. "
                        "Stop it and start a new interactive session to send this message?"
                    ),
                    status_code=409,
                )

        if (
            active_any is not None
            and effective_on_conflict
            == TaskAgentMessageConflictAction.StopSessionAndStartNew
        ):
            try:
                await runner_backend.stop_task_agent(task_id=task_id)
            except (RunnerBackendError, RuntimeError) as exc:
                raise _as_message_error(exc) from exc
            active_any = None

        if active_any is not None:
            try:
                await runner_backend.send_task_agent_text(
                    task_id=task_id,
                    text=trimmed,
                    interrupt=effective_on_conflict
                    == TaskAgentMessageConflictAction.InterruptTurn,
                    submit=True,
                )
            except (RunnerBackendError, RuntimeError) as exc:
                raise _as_message_error(exc) from exc
            return TaskAgentMessageResult(
                agent_session_id=active_any.id,
                agent_interface_mode=active_any.agent_interface_mode,
                delivery="interactive_sent",
                warnings=(
                    "Sent message to the currently running agent session (it does not match the configured harness command).",
                ),
            )

    if active_interactive is None:
        try:
            result = await runner_backend.start_task_agent(
                task_id=task_id,
                harness_command=harness_command,
                agent_kind_selection=defaults.harness.agent_kind,
                detach=defaults.harness.detach,
                prelude_override=None,
                initial_prompt=trimmed,
            )
        except (RunnerBackendError, RuntimeError, ValueError) as exc:
            raise _as_message_error(exc) from exc
        if not result.started:
            try:
                await runner_backend.send_task_agent_text(
                    task_id=task_id,
                    text=trimmed,
                    interrupt=effective_on_conflict
                    == TaskAgentMessageConflictAction.InterruptTurn,
                    submit=True,
                )
            except (RunnerBackendError, RuntimeError) as exc:
                raise _as_message_error(exc) from exc
            return TaskAgentMessageResult(
                agent_session_id=result.agent_session.id,
                agent_interface_mode=AgentInterfaceMode.Interactive,
                delivery="interactive_sent",
                warnings=result.warnings,
            )
        return TaskAgentMessageResult(
            agent_session_id=result.agent_session.id,
            agent_interface_mode=result.agent_session.agent_interface_mode,
            delivery="interactive_started",
            warnings=result.warnings,
        )

    try:
        await runner_backend.send_task_agent_text(
            task_id=task_id,
            text=trimmed,
            interrupt=effective_on_conflict
            == TaskAgentMessageConflictAction.InterruptTurn,
            submit=True,
        )
    except (RunnerBackendError, RuntimeError) as exc:
        raise _as_message_error(exc) from exc
    return TaskAgentMessageResult(
        agent_session_id=active_interactive.id,
        agent_interface_mode=AgentInterfaceMode.Interactive,
        delivery="interactive_sent",
        warnings=(),
    )
