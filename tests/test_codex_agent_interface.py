from __future__ import annotations

import pytest

from redesmyn.agent_interface.codex import CodexAgent
from redesmyn.agent_interface.v0 import (
    AgentAssistantMessageEvent,
    AgentTurnCompletedEvent,
    AgentTurnStartedEvent,
    ExternalSessionCodex,
)
from redesmyn.domain.enums import AgentTurnState


@pytest.mark.unit
def test_codex_agent_parses_structured_turn_events() -> None:
    now_s = 0.0

    def clock() -> float:
        return now_s

    agent = CodexAgent(clock_s=clock, max_busy_s=5.0)
    assert agent.capabilities.can_stream_semantic_events is False
    assert agent.capabilities.can_detect_turn_complete is False
    assert agent.capabilities.can_resume_by_id is False

    events = agent.consume_output('{"type":"thread.started","thread_id":"th_123"}\n')
    assert events == []
    assert agent.capabilities.can_stream_semantic_events is True
    assert agent.capabilities.can_detect_turn_complete is False
    assert agent.capabilities.can_resume_by_id is True
    codex_ref = ExternalSessionCodex.model_validate(
        agent.external_session_ref.model_dump(mode="python")
    )
    assert codex_ref.thread_id == "th_123"
    assert codex_ref.turn_id is None

    events = agent.consume_output('{"type":"turn.started","turn_id":"tu_1"}\n')
    assert any(isinstance(e, AgentTurnStartedEvent) for e in events)
    assert agent.capabilities.can_detect_turn_complete is True
    assert agent.semantic_status.turn_state == AgentTurnState.Busy
    codex_ref = ExternalSessionCodex.model_validate(
        agent.external_session_ref.model_dump(mode="python")
    )
    assert codex_ref.turn_id == "tu_1"

    events = agent.consume_output('{"type":"turn.completed","turn_id":"tu_1"}\n')
    assert any(isinstance(e, AgentTurnCompletedEvent) for e in events)
    assert agent.semantic_status.turn_state == AgentTurnState.Completed
    codex_ref = ExternalSessionCodex.model_validate(
        agent.external_session_ref.model_dump(mode="python")
    )
    assert codex_ref.turn_id == "tu_1"


@pytest.mark.unit
def test_codex_agent_uses_prompt_heuristics_for_turn_completion() -> None:
    now_s = 0.0

    def clock() -> float:
        return now_s

    agent = CodexAgent(clock_s=clock, max_busy_s=5.0)

    agent.consume_output("OpenAI Codex\n> ")
    assert agent.semantic_status.turn_state == AgentTurnState.Ready

    agent.consume_output("> hello\n")
    assert agent.semantic_status.turn_state == AgentTurnState.Busy

    agent.consume_output("response line 1\nresponse line 2\n> ")
    assert agent.semantic_status.turn_state == AgentTurnState.Completed


@pytest.mark.unit
def test_codex_agent_degrades_to_unknown_on_timeout() -> None:
    now_s = 0.0

    def clock() -> float:
        return now_s

    agent = CodexAgent(clock_s=clock, max_busy_s=1.0)

    agent.consume_output("OpenAI Codex\n> ")
    agent.consume_output("> do something\n")
    assert agent.semantic_status.turn_state == AgentTurnState.Busy

    now_s = 2.0
    agent.consume_output("")
    assert agent.semantic_status.turn_state == AgentTurnState.Unknown
    assert agent.semantic_status.detail is not None


@pytest.mark.unit
def test_codex_agent_emits_assistant_message_events_from_structured_stream() -> None:
    agent = CodexAgent()
    agent.consume_output('{"type":"thread.started","thread_id":"th_123"}\n')
    events = agent.consume_output(
        '{"type":"assistant.message","role":"assistant","text":"Hello there"}\n'
    )
    assert any(isinstance(e, AgentAssistantMessageEvent) for e in events)


@pytest.mark.unit
def test_codex_agent_emits_assistant_message_events_from_item_completed_reasoning() -> (
    None
):
    agent = CodexAgent()
    events = agent.consume_output(
        '{"type":"item.completed","item":{"id":"item_88","type":"reasoning","text":"Hello world"}}\n'
    )
    assert any(isinstance(e, AgentAssistantMessageEvent) for e in events)
    assert agent.capabilities.can_stream_semantic_events is True
    assert agent.capabilities.can_detect_turn_complete is False


@pytest.mark.unit
def test_codex_agent_treats_item_started_as_activity_signal() -> None:
    agent = CodexAgent()
    events = agent.consume_output(
        '{"type":"item.started","item":{"id":"item_90","type":"command_execution","command":"/bin/zsh -lc \\"echo hi\\""}}\n'
    )
    assert any(isinstance(e, AgentTurnStartedEvent) for e in events)
    assert agent.semantic_status.turn_state == AgentTurnState.Busy
    assert agent.semantic_status.detail is not None
    assert agent.capabilities.can_detect_turn_complete is False
