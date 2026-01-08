from __future__ import annotations

import pytest

from redesmyn.schemas.core import AgentPreviewResponse


@pytest.mark.unit
def test_agent_preview_response_decodes_html_entities() -> None:
    preview = AgentPreviewResponse.model_validate(
        {
            "last_assistant_message_preview": "I&amp;#39;ve added settings",
            "last_assistant_message_at": None,
            "last_message_turn_id": None,
        }
    )
    assert preview.last_assistant_message_preview == "I've added settings"
