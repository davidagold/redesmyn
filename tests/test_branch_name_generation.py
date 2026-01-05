from __future__ import annotations

import pytest

from redesmyn.agent_runtime import _default_branch_name_for_task
from redesmyn.db import Task


@pytest.mark.unit
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "T-2 Harness identification + user override (Codex/Claude/Generic)",
            "rn/harness-interface-v0/T-2-harness-identification",
        ),
        (
            "T-3 Codex harness interface implementation (turn detection + capabilities)",
            "rn/harness-interface-v0/T-3-codex-harness-interface",
        ),
        (
            "T-1 Harness interface + capabilities + GenericHarness",
            "rn/harness-interface-v0/T-1-harness-interface",
        ),
        (
            "Some non-task title",
            "rn/harness-interface-v0/task-74-some-non-task-title",
        ),
    ],
)
def test_default_branch_name_for_task_generates_expected_format(
    title: str, expected: str
) -> None:
    task = Task(id=74, epic_id=1, title=title, linear_identifier=None)
    assert (
        _default_branch_name_for_task(epic_slug="harness-interface-v0", task=task)
        == expected
    )


@pytest.mark.unit
def test_default_branch_name_for_task_prefers_linear_identifier() -> None:
    task = Task(
        id=12,
        epic_id=1,
        title="RED-12 Linear client: write support (labels, state, dependencies, create/update)",
        linear_identifier="RED-12",
    )
    assert (
        _default_branch_name_for_task(epic_slug="linear-integration", task=task)
        == "rn/linear-integration/RED-12-linear-client-write-support"
    )
