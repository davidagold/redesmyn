from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, cast

from pydantic import TypeAdapter

from redesmyn.agent_interface.v0 import (
    AgentCapabilities,
    AgentEvent,
    AgentSemanticStatus,
    ExternalSessionCodex,
    ExternalSessionNone,
    ExternalSessionRef,
)
from redesmyn.domain.enums import AgentTurnState

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


def _best_effort_str(obj: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
            can_detect_turn_complete=True,
            can_send_text=True,
            can_resume_by_id=True,
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
        if ref.type == "codex_thread":
            self._external_session_ref = ref

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
            obj = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(obj, dict):
            return
        type_value = obj.get("type")
        if not isinstance(type_value, str):
            return
        event_type = type_value.strip()
        if not event_type:
            return

        self._saw_structured_events = True
        if self._capabilities.can_stream_semantic_events is not True:
            self._capabilities = self._capabilities.model_copy(
                update={"can_stream_semantic_events": True}
            )

        if event_type == "thread.started":
            thread_id = _best_effort_str(
                cast(dict[str, object], obj), "thread_id", "threadId"
            )
            if thread_id:
                turn_id = None
                if self._external_session_ref.type == "codex_thread":
                    turn_id = self._external_session_ref.turn_id
                self._external_session_ref = ExternalSessionCodex(
                    thread_id=thread_id, turn_id=turn_id
                )
            return

        if event_type == "turn.started":
            turn_id = _best_effort_str(
                cast(dict[str, object], obj), "turn_id", "turnId"
            )
            if turn_id and self._external_session_ref.type == "codex_thread":
                self._external_session_ref = ExternalSessionCodex(
                    thread_id=self._external_session_ref.thread_id,
                    turn_id=turn_id,
                )
            self._note_busy()
            return

        if event_type == "turn.completed":
            turn_id = _best_effort_str(
                cast(dict[str, object], obj), "turn_id", "turnId"
            )
            if turn_id and self._external_session_ref.type == "codex_thread":
                self._external_session_ref = ExternalSessionCodex(
                    thread_id=self._external_session_ref.thread_id,
                    turn_id=turn_id,
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

        if not thread_id and self._external_session_ref.type == "codex_thread":
            thread_id = self._external_session_ref.thread_id
        if not thread_id:
            return

        self._external_session_ref = ExternalSessionCodex(
            thread_id=thread_id,
            turn_id=turn_id
            if turn_id
            else (
                self._external_session_ref.turn_id
                if self._external_session_ref.type == "codex_thread"
                else None
            ),
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
    except Exception:
        ref = ExternalSessionNone()
    agent.seed_external_session_ref(ref)
    return agent
