from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from redesmyn.domain.enums import AgentKind, AgentKindSelection


def _name(token: str) -> str:
    return Path(token).name.lower()


def _direct_kind_from_name(name: str) -> AgentKind | None:
    if name == "codex":
        return AgentKind.Codex
    if name in {"claude", "claude-code"}:
        return AgentKind.ClaudeCode
    return None


def infer_agent_kind_from_argv(argv: list[str]) -> AgentKind:
    """Best-effort inference of agent kind from argv.

    - If argv[0] is codex/claude, treat it as that agent.
    - If argv[0] is a common wrapper (uv/npx/etc), scan the post-wrapper tokens.
    - Otherwise, default to Generic (avoid false positives like `echo codex`).
    """
    if not argv:
        return AgentKind.Generic

    name0 = _name(argv[0])
    direct = _direct_kind_from_name(name0)
    if direct is not None:
        return direct

    tokens: Iterable[str] = ()
    if name0 == "uv":
        if "run" in argv[1:]:
            idx = argv.index("run", 1)
            tokens = argv[idx + 1 :]
        else:
            tokens = argv[1:]
    elif name0 in {"npx", "bunx"}:
        tokens = argv[1:]
    elif name0 == "npm":
        if len(argv) > 1 and argv[1] in {"exec", "x"}:
            tokens = argv[2:]
    elif name0 == "pnpm":
        if len(argv) > 1 and argv[1] in {"dlx", "exec"}:
            tokens = argv[2:]
    elif name0 == "yarn":
        if len(argv) > 1 and argv[1] in {"dlx", "exec", "run"}:
            tokens = argv[2:]

    for token in tokens:
        if not token or token.startswith("-"):
            continue
        kind = _direct_kind_from_name(_name(token))
        if kind is not None:
            return kind

    return AgentKind.Generic


def agent_kind_from_external_session_ref(
    external_session_ref: dict[str, Any] | None,
) -> AgentKind | None:
    if not external_session_ref:
        return None
    match external_session_ref.get("type"):
        case "codex_thread":
            return AgentKind.Codex
        case "claude_session":
            return AgentKind.ClaudeCode
        case _:
            return None


def resolve_agent_kind(
    selection: AgentKindSelection,
    argv: list[str],
    *,
    external_session_ref_hint: dict[str, Any] | None = None,
) -> AgentKind:
    if selection == AgentKindSelection.Codex:
        return AgentKind.Codex
    if selection == AgentKindSelection.ClaudeCode:
        return AgentKind.ClaudeCode
    if selection == AgentKindSelection.Generic:
        return AgentKind.Generic

    # Auto: prefer a persisted external resume handle when present to avoid
    # flapping across restarts.
    if selection == AgentKindSelection.Auto:
        hinted = agent_kind_from_external_session_ref(external_session_ref_hint)
        if hinted is not None:
            return hinted
        return infer_agent_kind_from_argv(argv)

    raise ValueError(f"Unknown agent kind selection: {selection!r}")
