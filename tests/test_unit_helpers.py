from __future__ import annotations

import pytest

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
                agent_id=34,
                agent_name="local",
                agent_status=AgentStatus.Running,
            )
        ]
    )
    assert "Proceed anyway?" in text
    assert "T-12" in text
