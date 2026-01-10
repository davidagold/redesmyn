from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from redesmyn.agent_interface.v0 import (
    ExternalSessionClaude,
    ExternalSessionCodex,
    ExternalSessionRef,
)
from redesmyn.domain.enums import AgentKind


class StructuredTurnTransportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResumeByIdTurn:
    argv: list[str]
    stdin_prompt: str


_EXTERNAL_SESSION_REF_ADAPTER = TypeAdapter(ExternalSessionRef)


def _name(token: str) -> str:
    return Path(token).name.lower()


def _find_executable_index(argv: list[str], *, names: set[str]) -> int | None:
    for idx, token in enumerate(argv):
        if not token or token.startswith("-"):
            continue
        if _name(token) in names:
            return idx
    return None


def _split_flags_with_values(
    tokens: list[str], *, flags_with_values: set[str]
) -> tuple[list[str], list[str]]:
    """Split `tokens` into [options...] + [rest...], best-effort.

    This is intentionally conservative: we only treat a flag as consuming a
    value when it is in `flags_with_values`. Everything else is treated as
    a no-value flag.
    """

    options: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--":
            options.append(token)
            return options, tokens[idx + 1 :]
        if not token.startswith("-"):
            return options, tokens[idx:]
        options.append(token)
        if token in flags_with_values and (idx + 1) < len(tokens):
            options.append(tokens[idx + 1])
            idx += 2
            continue
        idx += 1
    return options, []


def build_resume_by_id_turn(
    *,
    base_argv: list[str],
    agent_kind: AgentKind,
    external_session_ref_raw: object,
    prompt: str,
) -> ResumeByIdTurn:
    """Build a structured "resume-by-id" turn invocation.

    This is the transport used by downstream automation (e.g. conflict assist)
    when an agent session supports resuming an existing conversation by an
    external id.
    """

    try:
        external_session_ref = _EXTERNAL_SESSION_REF_ADAPTER.validate_python(
            external_session_ref_raw
        )
    except ValidationError as exc:
        raise StructuredTurnTransportError(
            "Invalid external_session_ref; cannot build resume-by-id turn."
        ) from exc

    if external_session_ref.type == "none":
        raise StructuredTurnTransportError(
            "Missing external resume handle; cannot build resume-by-id turn."
        )

    prompt_text = prompt if prompt.endswith("\n") else prompt + "\n"

    match agent_kind:
        case AgentKind.Codex:
            if not isinstance(external_session_ref, ExternalSessionCodex):
                raise StructuredTurnTransportError(
                    "External resume handle is not a Codex thread id."
                )
            return ResumeByIdTurn(
                argv=_build_codex_exec_resume_argv(
                    base_argv, session_id=external_session_ref.thread_id
                ),
                stdin_prompt=prompt_text,
            )
        case AgentKind.ClaudeCode:
            if not isinstance(external_session_ref, ExternalSessionClaude):
                raise StructuredTurnTransportError(
                    "External resume handle is not a Claude session id."
                )
            return ResumeByIdTurn(
                argv=_build_claude_resume_argv(
                    base_argv, session_id=external_session_ref.session_id
                ),
                stdin_prompt=prompt_text,
            )
        case _:
            raise StructuredTurnTransportError(
                f"Agent kind {agent_kind.value!r} does not support resume-by-id turns."
            )


def _build_codex_exec_resume_argv(
    base_argv: list[str], *, session_id: str
) -> list[str]:
    idx = _find_executable_index(base_argv, names={"codex"})
    if idx is None:
        raise StructuredTurnTransportError("Unable to find Codex executable in argv.")
    try:
        exec_idx = base_argv.index("exec", idx + 1)
    except ValueError as exc:
        raise StructuredTurnTransportError("Codex argv is missing `exec`.") from exc

    prefix = list(base_argv[: exec_idx + 1])
    rest = list(base_argv[exec_idx + 1 :])
    if "--json" not in rest:
        raise StructuredTurnTransportError(
            "Codex argv must include `--json` for structured resume-by-id turns."
        )

    codex_exec_flags_with_values = {
        "-c",
        "--config",
        "-i",
        "--image",
        "-m",
        "--model",
        "--local-provider",
        "-s",
        "--sandbox",
        "-p",
        "--profile",
        "-C",
        "--cd",
        "--add-dir",
        "--output-schema",
        "--color",
        "-o",
        "--output-last-message",
        "--enable",
        "--disable",
    }
    options, _ = _split_flags_with_values(
        rest, flags_with_values=codex_exec_flags_with_values
    )
    return [*prefix, *options, "resume", session_id, "-"]


def _build_claude_resume_argv(base_argv: list[str], *, session_id: str) -> list[str]:
    idx = _find_executable_index(base_argv, names={"claude", "claude-code"})
    if idx is None:
        raise StructuredTurnTransportError("Unable to find Claude executable in argv.")

    tokens_after = list(base_argv[idx + 1 :])
    # Preserve only the option-shaped portion of the argv. For `--print` mode we
    # send the prompt over stdin, so positional tokens are treated as an
    # ambiguous prompt/command and dropped.
    claude_flags_with_values = {
        "--output-format",
        "--input-format",
        "--json-schema",
        "--max-budget-usd",
        "--allowedTools",
        "--allowed-tools",
        "--tools",
        "--disallowedTools",
        "--disallowed-tools",
        "--mcp-config",
        "--system-prompt",
        "--append-system-prompt",
        "--permission-mode",
        "--model",
        "--agent",
        "--betas",
        "--fallback-model",
        "--settings",
        "--add-dir",
        "--session-id",
        "--agents",
        "--setting-sources",
        "--plugin-dir",
    }
    options, _ = _split_flags_with_values(
        tokens_after, flags_with_values=claude_flags_with_values
    )

    # Drop continuation flags; we are explicitly resuming by id.
    cleaned: list[str] = []
    skip_next = False
    for token in options:
        if skip_next:
            skip_next = False
            continue
        if token in {"-c", "--continue"}:
            continue
        if token in {"-r", "--resume"}:
            skip_next = True
            continue
        cleaned.append(token)

    if "--print" not in cleaned and "-p" not in cleaned:
        raise StructuredTurnTransportError(
            "Claude argv must include `--print` for structured resume-by-id turns."
        )
    if (
        not ("--output-format" in cleaned and "stream-json" in cleaned)
        and "--output-format=stream-json" not in cleaned
    ):
        raise StructuredTurnTransportError(
            "Claude argv must include `--output-format stream-json` for structured turns."
        )

    insert_at = len(cleaned)
    for idx_opt, token in enumerate(cleaned):
        if token in {"--print", "-p"}:
            insert_at = idx_opt + 1
            break
    return [
        *base_argv[: idx + 1],
        *cleaned[:insert_at],
        "--resume",
        session_id,
        *cleaned[insert_at:],
    ]
