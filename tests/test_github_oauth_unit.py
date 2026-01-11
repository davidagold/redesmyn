from __future__ import annotations

import pytest

from redesmyn.integrations.github_oauth import _normalize_scopes


@pytest.mark.unit
def test_normalize_scopes_dedupes_and_normalizes() -> None:
    assert _normalize_scopes("repo, read:user repo  ") == "repo read:user"
