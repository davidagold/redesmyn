from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from redesmyn.agent_interface import register_agent_backend_builder
from redesmyn.agent_interface.v0 import (
    AgentCapabilities,
    AgentBackend,
    AgentEvent,
    AgentSemanticStatus,
    ExternalSessionClaude,
    ExternalSessionNone,
    ExternalSessionRef,
)
from redesmyn.domain.enums import AgentKind, AgentTurnState

if TYPE_CHECKING:
    from redesmyn.db.models import AgentSession


class _ClaudeEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    session_id: str | None = None
    subtype: str | None = None


_claude_event_adapter = TypeAdapter(_ClaudeEvent)
_external_session_ref_adapter = TypeAdapter(ExternalSessionRef)
_external_claude_ref_adapter = TypeAdapter(ExternalSessionClaude)


def _try_parse_claude_event_json(line: str) -> _ClaudeEvent | None:
    candidate = line.strip()
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None
    try:
        return _claude_event_adapter.validate_json(candidate)
    except ValidationError:
        return None


def _as_claude_session_ref(ref: ExternalSessionRef) -> ExternalSessionClaude | None:
    if ref.type != "claude_session":
        return None
    try:
        return _external_claude_ref_adapter.validate_python(ref)
    except ValidationError:
        return None


@dataclass(slots=True)
class ClaudeCodeAgent:
    """
    Claude Code agent interpreter.

    Preferred signal path: Claude "print mode" `--output-format stream-json`, which emits
    one JSON object per line and includes a stable `session_id`.

    When structured output is not present (interactive TUI), we conservatively remain in
    `Unknown` turn state.
    """

    _max_buffer_chars: int = 256_000
    _buffer: str = ""
    _saw_structured_events: bool = False
    _capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(
            can_send_text=True,
            # Resume-by-id requires an observed/seeded Claude session_id.
            can_resume_by_id=False,
            can_continue_in_cwd=True,
        )
    )
    _status: AgentSemanticStatus = field(default_factory=AgentSemanticStatus)
    _external_session_ref: ExternalSessionRef = field(
        default_factory=ExternalSessionNone
    )

    def __post_init__(self) -> None:
        self._set_external_session_ref(self._external_session_ref)

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    @property
    def semantic_status(self) -> AgentSemanticStatus:
        return self._status

    @property
    def external_session_ref(self) -> ExternalSessionRef:
        return self._external_session_ref

    def seed_external_session_ref(self, ref: ExternalSessionRef) -> None:
        self._set_external_session_ref(ref)

    def _set_external_session_ref(self, ref: ExternalSessionRef) -> None:
        self._external_session_ref = ref
        can_resume = ref.type == "claude_session"
        if self._capabilities.can_resume_by_id != can_resume:
            self._capabilities = self._capabilities.model_copy(
                update={"can_resume_by_id": can_resume}
            )

    def _enable_stream_semantic_capabilities(self) -> None:
        cap_updates: dict[str, bool] = {}
        if self._capabilities.can_stream_semantic_events is not True:
            cap_updates["can_stream_semantic_events"] = True
        if self._capabilities.can_detect_ready_for_input is not True:
            cap_updates["can_detect_ready_for_input"] = True
        if self._capabilities.can_detect_turn_complete is not True:
            cap_updates["can_detect_turn_complete"] = True
        if cap_updates:
            self._capabilities = self._capabilities.model_copy(update=cap_updates)

    def _maybe_update_external_session_ref(self, event: _ClaudeEvent) -> None:
        session_id = event.session_id
        if session_id is None or not session_id.strip():
            return
        if _as_claude_session_ref(self._external_session_ref) is not None:
            return
        self._set_external_session_ref(ExternalSessionClaude(session_id=session_id))

    def _consume_event(self, event: _ClaudeEvent) -> None:
        self._maybe_update_external_session_ref(event)
        self._saw_structured_events = True

        if event.type == "system" and event.subtype == "init":
            self._enable_stream_semantic_capabilities()
            self._status = AgentSemanticStatus(
                turn_state=AgentTurnState.Ready,
                detail="claude:init",
            )
            return

        if event.type in {"assistant", "user"}:
            self._enable_stream_semantic_capabilities()
            self._status = AgentSemanticStatus(
                turn_state=AgentTurnState.Busy,
                detail=f"claude:{event.type}",
            )
            return

        if event.type == "result":
            self._enable_stream_semantic_capabilities()
            subtype_str = event.subtype or "unknown"
            self._status = AgentSemanticStatus(
                turn_state=AgentTurnState.Ready,
                detail=f"claude:result:{subtype_str}",
            )
            return

    def _consume_json_block_from_buffer(self) -> None:
        if self._saw_structured_events:
            return
        stripped = self._buffer.lstrip()
        if not stripped.startswith("{"):
            return

        decoder = json.JSONDecoder()
        leading = len(self._buffer) - len(stripped)
        try:
            _, end_index = decoder.raw_decode(stripped)
        except json.JSONDecodeError:
            return

        json_text = stripped[:end_index]
        try:
            event = _claude_event_adapter.validate_json(json_text)
        except ValidationError:
            event = None
        if event is not None:
            self._consume_event(event)
        self._buffer = self._buffer[leading + end_index :]

    def consume_output(self, text: str) -> list[AgentEvent]:
        self._buffer += text
        if len(self._buffer) > self._max_buffer_chars:
            self._buffer = self._buffer[-self._max_buffer_chars :]

        while "\n" in self._buffer:
            line, rest = self._buffer.split("\n", 1)
            event = _try_parse_claude_event_json(line)
            if event is not None:
                self._consume_event(event)
                self._buffer = rest
                continue
            if line.lstrip().startswith("{") and not self._saw_structured_events:
                break
            self._buffer = rest
            continue

        self._consume_json_block_from_buffer()
        return []


def _build_claude_code_backend(agent_session: "AgentSession") -> AgentBackend:
    try:
        ref = _external_session_ref_adapter.validate_python(
            agent_session.external_session_ref
        )
    except ValidationError:
        ref = ExternalSessionNone()
    agent = ClaudeCodeAgent()
    agent.seed_external_session_ref(
        _as_claude_session_ref(ref) or ExternalSessionNone()
    )
    return agent


register_agent_backend_builder(AgentKind.ClaudeCode, _build_claude_code_backend)
