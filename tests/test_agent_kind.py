from __future__ import annotations

from dataclasses import dataclass

import pytest

from redesmyn.agent_kind import (
    infer_agent_kind_from_argv,
    resolve_agent_backend,
    resolve_agent_kind,
)
from redesmyn.agent_interface.v0 import ExternalSessionCodex
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


@dataclass(frozen=True, slots=True)
class _FakeAgentSession:
    agent_kind: AgentKind
    agent_kind_selection: AgentKindSelection
    external_session_ref: dict[str, object]
    resolved_launch_configuration: dict[str, object] | None = None


@pytest.mark.unit
def test_resolve_agent_backend_uses_registered_codex_builder() -> None:
    backend = resolve_agent_backend(
        agent_session=_FakeAgentSession(
            agent_kind=AgentKind.Codex,
            agent_kind_selection=AgentKindSelection.Codex,
            external_session_ref={"type": "codex_thread", "thread_id": "th_123"},
        )
    )
    from redesmyn.agent_interface.codex import CodexAgent

    assert isinstance(backend, CodexAgent)
    codex_ref = ExternalSessionCodex.model_validate(
        backend.external_session_ref.model_dump(mode="python")
    )
    assert codex_ref.thread_id == "th_123"
