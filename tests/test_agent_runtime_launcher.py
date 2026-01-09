from __future__ import annotations

from pathlib import Path

import pytest

from redesmyn.agent_runtime import agent_session_dir, write_agent_launcher
from redesmyn.context import build_repo_context


@pytest.mark.unit
def test_write_agent_launcher_redirects_stdin_when_provided(tmp_path: Path) -> None:
    ctx = build_repo_context(repo_root=tmp_path)
    script_path = write_agent_launcher(
        ctx=ctx,
        task_id=1,
        session_id=2,
        argv=["echo", "hello"],
        env={},
        stdin_text="prelude",
    )

    run_dir = agent_session_dir(ctx, task_id=1, session_id=2)
    stdin_path = run_dir / "stdin.txt"
    assert stdin_path.read_text(encoding="utf-8") == "prelude\n"

    script = script_path.read_text(encoding="utf-8")
    assert "EXIT_CODE_PATH=" in script
    assert "STDIN_PATH=" in script
    assert '<"$STDIN_PATH"' in script
    assert "EXIT_CODE=$?" in script


@pytest.mark.unit
def test_write_agent_launcher_does_not_redirect_stdin_by_default(
    tmp_path: Path,
) -> None:
    ctx = build_repo_context(repo_root=tmp_path)
    script_path = write_agent_launcher(
        ctx=ctx,
        task_id=1,
        session_id=2,
        argv=["echo", "hello"],
        env={},
    )

    run_dir = agent_session_dir(ctx, task_id=1, session_id=2)
    assert not (run_dir / "stdin.txt").exists()

    script = script_path.read_text(encoding="utf-8")
    assert "EXIT_CODE_PATH=" in script
    assert "STDIN_PATH=" not in script
    assert '<"$STDIN_PATH"' not in script
    assert "EXIT_CODE=$?" in script
