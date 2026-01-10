import pytest

from redesmyn.agent_turn_transport import (
    StructuredTurnTransportError,
    build_resume_by_id_turn,
)
from redesmyn.domain.enums import AgentKind


@pytest.mark.unit
def test_build_resume_by_id_turn_codex_exec_resume_stdin() -> None:
    turn = build_resume_by_id_turn(
        base_argv=["codex", "exec", "--json"],
        agent_kind=AgentKind.Codex,
        external_session_ref_raw={
            "type": "codex_thread",
            "thread_id": "th_123",
            "turn_id": None,
        },
        prompt="hello",
    )
    assert turn.argv == ["codex", "exec", "--json", "resume", "th_123", "-"]
    assert turn.stdin_prompt == "hello\n"


@pytest.mark.unit
def test_build_resume_by_id_turn_codex_preserves_exec_options() -> None:
    turn = build_resume_by_id_turn(
        base_argv=[
            "uv",
            "run",
            "codex",
            "exec",
            "--json",
            "-C",
            "/tmp/repo",
            "--color",
            "never",
        ],
        agent_kind=AgentKind.Codex,
        external_session_ref_raw={
            "type": "codex_thread",
            "thread_id": "th_abc",
            "turn_id": "turn_1",
        },
        prompt="go",
    )
    assert turn.argv == [
        "uv",
        "run",
        "codex",
        "exec",
        "--json",
        "-C",
        "/tmp/repo",
        "--color",
        "never",
        "resume",
        "th_abc",
        "-",
    ]


@pytest.mark.unit
def test_build_resume_by_id_turn_claude_resume_inserts_after_print() -> None:
    turn = build_resume_by_id_turn(
        base_argv=["claude", "--print", "--output-format", "stream-json"],
        agent_kind=AgentKind.ClaudeCode,
        external_session_ref_raw={"type": "claude_session", "session_id": "sess_123"},
        prompt="fix conflicts",
    )
    assert turn.argv == [
        "claude",
        "--print",
        "--resume",
        "sess_123",
        "--output-format",
        "stream-json",
    ]


@pytest.mark.unit
def test_build_resume_by_id_turn_claude_drops_continue_and_replaces_resume() -> None:
    turn = build_resume_by_id_turn(
        base_argv=[
            "claude",
            "--print",
            "--output-format",
            "stream-json",
            "--continue",
            "--resume",
            "old",
        ],
        agent_kind=AgentKind.ClaudeCode,
        external_session_ref_raw={"type": "claude_session", "session_id": "sess_new"},
        prompt="go",
    )
    assert "--continue" not in turn.argv
    assert turn.argv.count("--resume") == 1
    assert turn.argv[turn.argv.index("--resume") + 1] == "sess_new"


@pytest.mark.unit
def test_build_resume_by_id_turn_requires_external_session_ref() -> None:
    with pytest.raises(
        StructuredTurnTransportError, match="Missing external resume handle"
    ):
        build_resume_by_id_turn(
            base_argv=["codex", "exec", "--json"],
            agent_kind=AgentKind.Codex,
            external_session_ref_raw={"type": "none"},
            prompt="hello",
        )


@pytest.mark.unit
def test_build_resume_by_id_turn_claude_requires_print_mode() -> None:
    with pytest.raises(StructuredTurnTransportError, match="must include `--print`"):
        build_resume_by_id_turn(
            base_argv=["claude", "--output-format", "stream-json"],
            agent_kind=AgentKind.ClaudeCode,
            external_session_ref_raw={
                "type": "claude_session",
                "session_id": "sess_123",
            },
            prompt="hi",
        )


@pytest.mark.unit
def test_build_resume_by_id_turn_codex_requires_json_flag() -> None:
    with pytest.raises(StructuredTurnTransportError, match="must include `--json`"):
        build_resume_by_id_turn(
            base_argv=["codex", "exec"],
            agent_kind=AgentKind.Codex,
            external_session_ref_raw={
                "type": "codex_thread",
                "thread_id": "th_123",
                "turn_id": None,
            },
            prompt="hello",
        )
