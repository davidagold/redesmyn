from __future__ import annotations

import pytest

from redesmyn.integrations import linear


@pytest.mark.unit
def test_linear_project_queries_use_id_variables() -> None:
    # Linear's GraphQL schema expects ID for project.id arguments/filters.
    for query in (
        linear.PROJECT_URL_QUERY,
        linear.PROJECT_ISSUES_QUERY,
        linear.PROJECT_ISSUES_BY_LABEL_QUERY,
        linear.PROJECT_RELATIONS_QUERY,
        linear.PROJECT_TEAMS_QUERY,
        linear.PROJECT_MILESTONES_QUERY,
        linear.PROJECT_MILESTONES_QUERY_FALLBACK,
        linear.PROJECT_QUERY_MIN,
        linear.PROJECT_QUERY_BARE,
    ):
        assert "$projectId: ID!" in query
