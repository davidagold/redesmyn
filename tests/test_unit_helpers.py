from __future__ import annotations

from pathlib import Path

import pytest

from redesmyn.agent_runtime import agent_log_path_for_session_row
from redesmyn.context import build_repo_context
from redesmyn.db import AgentSession
from redesmyn.domain.enums import AgentStatus
from redesmyn.git_mechanics_v0 import (
    RunningAgentInfo,
    format_running_agents_confirmation,
)


@pytest.mark.unit
def test_format_running_agents_confirmation_includes_prompt() -> None:
    text = format_running_agents_confirmation(
        [
            RunningAgentInfo(
                task_id=12,
                branch_name="feat/example",
                agent_session_id=34,
                agent_label="a-12",
                agent_status=AgentStatus.Running,
            )
        ]
    )
    assert "Proceed anyway?" in text
    assert "T-12" in text


@pytest.mark.unit
def test_agent_log_path_for_session_row_defaults_to_task_scoped_path(
    tmp_path: Path,
) -> None:
    ctx = build_repo_context(repo_root=tmp_path, worktree_root=tmp_path)
    row = AgentSession(task_id=12, status=AgentStatus.Running, attach={"type": "none"})
    row.id = 34
    assert agent_log_path_for_session_row(ctx, agent_session_row=row) == (
        tmp_path / ".redesmyn" / "tasks" / "12" / "agent-sessions" / "34" / "output.log"
    )
