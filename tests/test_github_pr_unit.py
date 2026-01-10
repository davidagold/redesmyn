from __future__ import annotations

import pytest

from redesmyn.integrations.github_pr import GitHubPullRequestRef


@pytest.mark.unit
def test_github_pr_ref_round_trips() -> None:
    ref = GitHubPullRequestRef(owner="octo-org", repo="octo-repo", number=123)
    assert ref.to_id() == "octo-org/octo-repo#123"
    assert GitHubPullRequestRef.parse(ref.to_id()) == ref
    assert ref.url == "https://github.com/octo-org/octo-repo/pull/123"


@pytest.mark.unit
def test_github_pr_ref_parse_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        GitHubPullRequestRef.parse("octo-org/octo-repo")
    with pytest.raises(ValueError):
        GitHubPullRequestRef.parse("octo-org/octo-repo#0")
