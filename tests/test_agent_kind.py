from __future__ import annotations

import pytest

from redesmyn.agent_kind import infer_agent_kind_from_argv, resolve_agent_kind
from redesmyn.domain.enums import AgentKind, AgentKindSelection


@pytest.mark.unit
def test_infer_agent_kind_from_argv_direct_binaries() -> None:
    assert infer_agent_kind_from_argv(["codex", "exec", "ls"]) == AgentKind.Codex
    assert infer_agent_kind_from_argv(["claude", "-p", "hi"]) == AgentKind.ClaudeCode


@pytest.mark.unit
def test_infer_agent_kind_from_argv_tolerates_wrappers() -> None:
    assert infer_agent_kind_from_argv(["uv", "run", "codex"]) == AgentKind.Codex
    assert (
        infer_agent_kind_from_argv(["uv", "run", "--quiet", "codex"]) == AgentKind.Codex
    )
    assert (
        infer_agent_kind_from_argv(["npx", "-y", "@anthropic-ai/claude-code"])
        == AgentKind.ClaudeCode
    )
    assert (
        infer_agent_kind_from_argv(["npm", "exec", "--", "claude"])
        == AgentKind.ClaudeCode
    )


@pytest.mark.unit
def test_infer_agent_kind_from_argv_avoids_false_positives() -> None:
    assert infer_agent_kind_from_argv(["echo", "codex"]) == AgentKind.Generic


@pytest.mark.unit
def test_resolve_agent_kind_uses_external_session_ref_hint_in_auto_mode() -> None:
    assert (
        resolve_agent_kind(
            AgentKindSelection.Auto,
            ["bash"],
            external_session_ref_hint={"type": "codex_thread", "thread_id": "th_123"},
        )
        == AgentKind.Codex
    )
    assert (
        resolve_agent_kind(
            AgentKindSelection.Generic,
            ["codex"],
            external_session_ref_hint={"type": "codex_thread", "thread_id": "th_123"},
        )
        == AgentKind.Generic
    )
