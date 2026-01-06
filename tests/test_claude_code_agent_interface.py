from __future__ import annotations

import pytest

from redesmyn.agent_interface.claude_code import ClaudeCodeAgent
from redesmyn.agent_kind import resolve_agent_backend
from redesmyn.agent_interface.v0 import ExternalSessionClaude
from redesmyn.db import AgentSession
from redesmyn.domain.enums import AgentKind, AgentTurnState


@pytest.mark.unit
def test_claude_code_stream_json_updates_session_id_and_turn_state() -> None:
    agent = ClaudeCodeAgent()

    agent.consume_output(
        '{"type":"system","subtype":"init","session_id":"sess_123"}\n'
        '{"type":"user","session_id":"sess_123"}\n'
        '{"type":"assistant","session_id":"sess_123"}\n'
    )
    assert agent.capabilities.can_stream_semantic_events is True
    assert agent.capabilities.can_detect_turn_complete is True
    assert agent.capabilities.can_detect_ready_for_input is True
    assert agent.capabilities.can_resume_by_id is True
    assert isinstance(agent.external_session_ref, ExternalSessionClaude)
    assert agent.external_session_ref.session_id == "sess_123"
    assert agent.semantic_status.turn_state == AgentTurnState.Busy

    agent.consume_output(
        '{"type":"result","subtype":"success","session_id":"sess_123"}\n'
    )
    assert agent.semantic_status.turn_state == AgentTurnState.Ready


@pytest.mark.unit
def test_claude_code_stream_json_handles_chunked_lines() -> None:
    agent = ClaudeCodeAgent()

    agent.consume_output('{"type":"system","subtype":"init","session_id":"s')
    assert agent.capabilities.can_stream_semantic_events is False

    agent.consume_output(
        'ess_abc"}\n{"type":"result","subtype":"success","session_id":"sess_abc"}\n'
    )
    assert isinstance(agent.external_session_ref, ExternalSessionClaude)
    assert agent.external_session_ref.session_id == "sess_abc"
    assert agent.capabilities.can_resume_by_id is True
    assert agent.semantic_status.turn_state == AgentTurnState.Ready


@pytest.mark.unit
def test_claude_code_json_output_format_parses_multiline_object() -> None:
    agent = ClaudeCodeAgent()

    agent.consume_output('{\n  "type": "result",\n')
    assert agent.semantic_status.turn_state == AgentTurnState.Unknown

    agent.consume_output(
        '  "subtype": "success",\n  "session_id": "sess_multiline"\n}\n'
    )
    assert isinstance(agent.external_session_ref, ExternalSessionClaude)
    assert agent.external_session_ref.session_id == "sess_multiline"
    assert agent.capabilities.can_resume_by_id is True
    assert agent.semantic_status.turn_state == AgentTurnState.Ready


@pytest.mark.unit
def test_claude_code_backend_builder_seeds_external_session_ref() -> None:
    session = AgentSession(
        task_id=1,
        agent_kind=AgentKind.ClaudeCode,
        external_session_ref={"type": "claude_session", "session_id": "sess_seeded"},
    )
    backend = resolve_agent_backend(agent_session=session)
    assert isinstance(backend.external_session_ref, ExternalSessionClaude)
    assert backend.external_session_ref.session_id == "sess_seeded"
    assert backend.capabilities.can_resume_by_id is True
    assert backend.capabilities.can_continue_in_cwd is True
    assert backend.capabilities.can_detect_turn_complete is False


@pytest.mark.unit
def test_claude_code_can_resume_by_id_false_until_session_id_present() -> None:
    agent = ClaudeCodeAgent()
    assert agent.capabilities.can_resume_by_id is False

    agent.consume_output('{"type":"system","subtype":"init"}\n')
    assert agent.capabilities.can_stream_semantic_events is True
    assert agent.capabilities.can_detect_turn_complete is True
    assert agent.capabilities.can_detect_ready_for_input is True
    assert agent.capabilities.can_resume_by_id is False

    agent.consume_output('{"type":"system","subtype":"init","session_id":"sess_now"}\n')
    assert agent.capabilities.can_resume_by_id is True


@pytest.mark.unit
def test_claude_code_ignores_non_json_lines_before_stream_json() -> None:
    agent = ClaudeCodeAgent()
    agent.consume_output(
        "noise line\n"
        '{"type":"system","subtype":"init","session_id":"sess_noise"}\n'
        '{"type":"result","subtype":"success","session_id":"sess_noise"}\n'
    )
    assert isinstance(agent.external_session_ref, ExternalSessionClaude)
    assert agent.external_session_ref.session_id == "sess_noise"
    assert agent.semantic_status.turn_state == AgentTurnState.Ready
