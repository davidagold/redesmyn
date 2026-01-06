from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from redesmyn.domain.enums import AgentTurnState


class AgentCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    can_detect_ready_for_input: bool = False
    can_detect_turn_complete: bool = False
    can_send_text: bool = False
    can_interrupt: bool = False
    can_receive_notifications: bool = False
    can_resume_by_id: bool = False
    can_continue_in_cwd: bool = False
    can_stream_semantic_events: bool = False


class AgentSemanticStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_state: AgentTurnState = AgentTurnState.Unknown
    detail: str | None = None


class ExternalSessionNone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["none"] = "none"


class ExternalSessionCodex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["codex_thread"] = "codex_thread"
    thread_id: str
    turn_id: str | None = None


class ExternalSessionClaude(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["claude_session"] = "claude_session"
    session_id: str


ExternalSessionRef = Annotated[
    ExternalSessionNone | ExternalSessionCodex | ExternalSessionClaude,
    Field(discriminator="type"),
]


class AgentSemanticStatusUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_semantic_status_update"] = "agent_semantic_status_update"
    status: AgentSemanticStatus


class AgentExternalSessionRefUpdateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["agent_external_session_ref_update"] = (
        "agent_external_session_ref_update"
    )
    external_session_ref: ExternalSessionRef


AgentEvent = Annotated[
    AgentSemanticStatusUpdateEvent | AgentExternalSessionRefUpdateEvent,
    Field(discriminator="type"),
]


class AgentTransport(Protocol):
    def send_text(self, text: str, *, submit: bool = True) -> None: ...

    def interrupt(self) -> None: ...


class AgentBackend(Protocol):
    @property
    def capabilities(self) -> AgentCapabilities: ...

    @property
    def semantic_status(self) -> AgentSemanticStatus: ...

    @property
    def external_session_ref(self) -> ExternalSessionRef: ...

    def consume_output(self, text: str) -> list[AgentEvent]: ...


@dataclass(frozen=True, slots=True)
class ShellAgent:
    _capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(can_send_text=True)
    )
    _status: AgentSemanticStatus = field(default_factory=AgentSemanticStatus)
    _external_session_ref: ExternalSessionRef = field(
        default_factory=ExternalSessionNone
    )

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    @property
    def semantic_status(self) -> AgentSemanticStatus:
        return self._status

    @property
    def external_session_ref(self) -> ExternalSessionRef:
        return self._external_session_ref

    def consume_output(self, text: str) -> list[AgentEvent]:
        # ShellAgent: no stable semantics without agent-specific signals.
        return []
