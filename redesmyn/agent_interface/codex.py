from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)

from redesmyn.agent_interface import register_agent_backend_builder
from redesmyn.agent_interface.v0 import (
    AgentCapabilities,
    AgentEvent,
    AgentSemanticStatus,
    ExternalSessionCodex,
    ExternalSessionNone,
    ExternalSessionRef,
)
from redesmyn.domain.enums import AgentKind
from redesmyn.domain.enums import AgentTurnState

if TYPE_CHECKING:
    from redesmyn.db.models import AgentSession

_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)


def _strip_ansi(text: str) -> str:
    # Codex output in tmux logs can include ANSI control sequences and OSC
    # payloads. We strip them so prompt / marker heuristics operate on stable
    # text.
    text = _ANSI_OSC_RE.sub("", text)
    return _ANSI_CSI_RE.sub("", text)


_PROMPT_TAIL_RE = re.compile(r"(?:^|[\r\n])>\s*$")
_PROMPT_TAIL_ALT_RE = re.compile(r"(?:^|[\r\n])>[_▌]\s*$")


def _tail_looks_like_codex_prompt(tail: str) -> bool:
    trimmed = tail[-400:].replace("\x00", "")
    trimmed = trimmed.rstrip(" \t")
    return bool(_PROMPT_TAIL_RE.search(trimmed) or _PROMPT_TAIL_ALT_RE.search(trimmed))


_EXTERNAL_CODEX_REF_ADAPTER = TypeAdapter(ExternalSessionCodex)


def _as_codex_thread_ref(ref: ExternalSessionRef) -> ExternalSessionCodex | None:
    if ref.type != "codex_thread":
        return None
    try:
        return _EXTERNAL_CODEX_REF_ADAPTER.validate_python(ref)
    except ValidationError:
        return None


class _CodexJsonlEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    thread_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("thread_id", "threadId"),
    )
    turn_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("turn_id", "turnId"),
    )


_CODEX_JSONL_EVENT_ADAPTER = TypeAdapter(_CodexJsonlEvent)


@dataclass(slots=True)
class CodexAgent:
    """Best-effort Codex agent semantics.

    Layered strategy:
    - Prefer structured JSONL events when present (codex exec --json).
    - Fall back to tmux-log heuristics (prompt detection) for interactive TUIs.
    - Avoid wedging forever: if we think we're busy but can't observe completion,
      we degrade to Unknown after a timeout.
    """

    clock_s: Callable[[], float] = time.monotonic
    max_busy_s: float = 15.0 * 60.0

    _capabilities: AgentCapabilities = field(
        default_factory=lambda: AgentCapabilities(
            can_detect_ready_for_input=True,
            can_send_text=True,
            # Turn-complete detection is only reliable when we see Codex's
            # structured JSONL stream (codex exec --json).
            can_detect_turn_complete=False,
            can_resume_by_id=False,
        )
    )
    _semantic_status: AgentSemanticStatus = field(default_factory=AgentSemanticStatus)
    _external_session_ref: ExternalSessionRef = field(
        default_factory=ExternalSessionNone
    )

    _jsonl_buffer: str = ""
    _text_tail: str = ""
    _saw_structured_events: bool = False

    _output_since_prompt: bool = False
    _saw_prompt_once: bool = False
    _busy_since_s: float | None = None

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    @property
    def semantic_status(self) -> AgentSemanticStatus:
        return self._semantic_status

    @property
    def external_session_ref(self) -> ExternalSessionRef:
        return self._external_session_ref

    def seed_external_session_ref(self, ref: ExternalSessionRef) -> None:
        self._set_external_session_ref(ref)

    def _set_external_session_ref(self, ref: ExternalSessionRef) -> None:
        self._external_session_ref = ref
        can_resume = ref.type == "codex_thread"
        if self._capabilities.can_resume_by_id != can_resume:
            self._capabilities = self._capabilities.model_copy(
                update={"can_resume_by_id": can_resume}
            )

    def _set_turn_state(
        self, state: AgentTurnState, *, detail: str | None = None
    ) -> None:
        if (
            self._semantic_status.turn_state == state
            and self._semantic_status.detail == detail
        ):
            return
        self._semantic_status = AgentSemanticStatus(turn_state=state, detail=detail)

    def _note_busy(self) -> None:
        if self._busy_since_s is None:
            self._busy_since_s = self.clock_s()
        self._set_turn_state(AgentTurnState.Busy)

    def _note_prompt(self) -> None:
        if (
            self._semantic_status.turn_state == AgentTurnState.Busy
            or self._busy_since_s is not None
            or (self._saw_prompt_once and self._output_since_prompt)
        ):
            self._set_turn_state(AgentTurnState.Completed)
        else:
            self._set_turn_state(AgentTurnState.Ready)
        self._saw_prompt_once = True
        self._output_since_prompt = False
        self._busy_since_s = None

    def _maybe_timeout(self) -> None:
        if self._busy_since_s is None:
            return
        if (self.clock_s() - self._busy_since_s) < self.max_busy_s:
            return
        self._busy_since_s = None
        self._output_since_prompt = False
        self._set_turn_state(
            AgentTurnState.Unknown,
            detail="Timeout waiting for Codex turn completion; signals unavailable.",
        )

    def _consume_structured_line(self, line: str) -> None:
        try:
            event = _CODEX_JSONL_EVENT_ADAPTER.validate_json(line)
        except ValidationError:
            return

        event_type = event.type.strip()
        if not event_type:
            return

        self._saw_structured_events = True
        cap_updates: dict[str, bool] = {}
        if self._capabilities.can_stream_semantic_events is not True:
            cap_updates["can_stream_semantic_events"] = True
        if self._capabilities.can_detect_turn_complete is not True:
            cap_updates["can_detect_turn_complete"] = True
        if cap_updates:
            self._capabilities = self._capabilities.model_copy(update=cap_updates)

        if event_type == "thread.started":
            if event.thread_id:
                existing_codex = _as_codex_thread_ref(self._external_session_ref)
                turn_id = existing_codex.turn_id if existing_codex is not None else None
                self._set_external_session_ref(
                    ExternalSessionCodex(thread_id=event.thread_id, turn_id=turn_id)
                )
            return

        if event_type == "turn.started":
            existing_codex = _as_codex_thread_ref(self._external_session_ref)
            if event.turn_id and existing_codex is not None:
                self._set_external_session_ref(
                    ExternalSessionCodex(
                        thread_id=existing_codex.thread_id,
                        turn_id=event.turn_id,
                    )
                )
            self._note_busy()
            return

        if event_type == "turn.completed":
            existing_codex = _as_codex_thread_ref(self._external_session_ref)
            if event.turn_id and existing_codex is not None:
                self._set_external_session_ref(
                    ExternalSessionCodex(
                        thread_id=existing_codex.thread_id,
                        turn_id=event.turn_id,
                    )
                )
            self._set_turn_state(AgentTurnState.Completed)
            self._busy_since_s = None
            self._output_since_prompt = False
            return

    def _consume_text_for_external_ids(self, text: str) -> None:
        # Best-effort parse of Codex notifications that include thread/turn ids.
        # When present, this lets us persist resume handles even for interactive
        # sessions where JSONL isn't available.
        match = re.search(r"\bthread[-_ ]?id\b\s*[:=]\s*([A-Za-z0-9_-]+)", text)
        thread_id = match.group(1) if match else None
        match = re.search(r"\bturn[-_ ]?id\b\s*[:=]\s*([A-Za-z0-9_-]+)", text)
        turn_id = match.group(1) if match else None

        existing_codex = _as_codex_thread_ref(self._external_session_ref)
        if not thread_id and existing_codex is not None:
            thread_id = existing_codex.thread_id
        if not thread_id:
            return

        self._set_external_session_ref(
            ExternalSessionCodex(
                thread_id=thread_id,
                turn_id=turn_id
                if turn_id
                else (existing_codex.turn_id if existing_codex is not None else None),
            )
        )

    def consume_output(self, text: str) -> list[AgentEvent]:
        # Empty text is treated as a "tick" so timeouts can fire even when logs
        # are quiet.
        if not text:
            self._maybe_timeout()
            return []

        # JSONL parsing: only activate when the stream is clean JSON objects per
        # line. If we don't successfully parse anything, we fall back to plain
        # text heuristics.
        saw_structured_in_chunk = False
        self._jsonl_buffer += text
        while True:
            if "\n" not in self._jsonl_buffer:
                break
            raw_line, self._jsonl_buffer = self._jsonl_buffer.split("\n", 1)
            line = raw_line.strip()
            if not line or not line.startswith("{") or not line.endswith("}"):
                continue
            before = self._saw_structured_events
            self._consume_structured_line(line)
            if self._saw_structured_events and not before:
                saw_structured_in_chunk = True
            elif self._saw_structured_events:
                saw_structured_in_chunk = True

        cleaned = _strip_ansi(text)
        self._consume_text_for_external_ids(cleaned)

        # In codex exec --json mode, the output stream itself is the semantic
        # signal; prompt-based heuristics are both unnecessary and actively
        # harmful (they would treat the JSON as "output since prompt" and mark
        # the agent busy forever).
        if saw_structured_in_chunk and cleaned.lstrip().startswith("{"):
            return []

        self._text_tail = (self._text_tail + cleaned)[-4000:]

        if cleaned.strip():
            self._output_since_prompt = True

        if _tail_looks_like_codex_prompt(self._text_tail):
            self._note_prompt()
        else:
            if self._output_since_prompt and (
                self._semantic_status.turn_state
                in {AgentTurnState.Ready, AgentTurnState.Completed}
                or self._saw_prompt_once
            ):
                self._note_busy()
            self._maybe_timeout()

        return []


def codex_agent_from_db_external_ref(ref_raw: object) -> CodexAgent:
    adapter = TypeAdapter(ExternalSessionRef)
    agent = CodexAgent()
    try:
        ref = adapter.validate_python(ref_raw)
    except ValidationError:
        ref = ExternalSessionNone()
    agent.seed_external_session_ref(ref)
    return agent


_EXTERNAL_SESSION_REF_ADAPTER = TypeAdapter(ExternalSessionRef)


def _build_codex_backend(agent_session: "AgentSession") -> CodexAgent:
    agent = CodexAgent()
    try:
        ref = _EXTERNAL_SESSION_REF_ADAPTER.validate_python(
            agent_session.external_session_ref
        )
    except ValidationError:
        ref = ExternalSessionNone()
    agent.seed_external_session_ref(ref)
    return agent


register_agent_backend_builder(AgentKind.Codex, _build_codex_backend)
