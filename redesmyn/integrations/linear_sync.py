from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Literal, cast

from redesmyn.integrations.linear_client import LinearApiError, LinearClient
from redesmyn.integrations.linear_models import (
    LinearIssue,
    LinearIssueRelation,
    LinearLabel,
    LinearMilestone,
    LinearProject,
    LinearTeam,
    LinearWorkflowState,
)

logger = logging.getLogger("redesmyn.integrations.linear")

_team_state_id_cache: dict[tuple[str, str, str], tuple[str, float]] = {}
_TEAM_STATE_ID_CACHE_TTL_S = 10 * 60


PROJECT_URL_QUERY = """
query ProjectUrl($projectId: ID!) {
  project(id: $projectId) {
    url
  }
}
"""

ISSUE_URL_QUERY = """
query IssueUrl($issueId: String!) {
  issue(id: $issueId) {
    url
  }
}
"""

PROJECT_ISSUES_QUERY = """
query ProjectIssues($projectId: ID!, $after: String) {
  project(id: $projectId) {
    id
    name
    issues(first: 50, after: $after) {
      nodes {
        id
        identifier
        title
        description
        state { type name }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


async def fetch_project_url(client: LinearClient, *, project_id: str) -> str:
    data = await client.graphql(PROJECT_URL_QUERY, variables={"projectId": project_id})
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("Linear project not found")
    url = project.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("Linear project URL missing")
    return url


async def fetch_issue_url(client: LinearClient, *, issue_id: str) -> str:
    data = await client.graphql(ISSUE_URL_QUERY, variables={"issueId": issue_id})
    issue = data.get("issue")
    if not isinstance(issue, dict):
        raise ValueError("Linear issue not found")
    url = issue.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("Linear issue URL missing")
    return url


PROJECT_ISSUES_BY_LABEL_QUERY = """
query ProjectIssuesByLabel($projectId: ID!, $labelName: String!, $after: String) {
  issues(
    first: 50,
    after: $after,
    filter: { project: { id: { eq: $projectId } }, labels: { name: { eq: $labelName } } }
  ) {
    nodes {
      id
      identifier
      title
      description
      state { type name }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

PROJECT_RELATIONS_QUERY = """
query ProjectIssueRelations($projectId: ID!, $after: String) {
  issueRelations(first: 100, after: $after, filter: { issue: { project: { id: { eq: $projectId } } } }) {
    nodes {
      id
      type
      issue { id }
      relatedIssue { id }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

PROJECT_TEAMS_QUERY = """
query ProjectTeams($projectId: ID!) {
  project(id: $projectId) {
    id
    teams(first: 50) {
      nodes { id key name }
    }
  }
}
"""

PROJECT_MILESTONES_QUERY = """
query ProjectMilestones($projectId: ID!, $after: String) {
  project(id: $projectId) {
    id
    projectMilestones(first: 50, after: $after) {
      nodes { id name }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

PROJECT_MILESTONES_QUERY_FALLBACK = """
query ProjectMilestonesFallback($projectId: ID!, $after: String) {
  project(id: $projectId) {
    id
    milestones(first: 50, after: $after) {
      nodes { id name }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

ISSUE_MILESTONE_QUERY = """
query IssueMilestone($id: String!) {
  issue(id: $id) {
    id
    projectMilestone { id name }
  }
}
"""

ISSUE_MILESTONE_QUERY_FALLBACK = """
query IssueMilestoneFallback($id: String!) {
  issue(id: $id) {
    id
    milestone { id name }
  }
}
"""

PROJECT_QUERY_MIN = """
query Project($projectId: ID!) {
  project(id: $projectId) {
    id
    name
    slugId
  }
}
"""

PROJECT_QUERY_BARE = """
query Project($projectId: ID!) {
  project(id: $projectId) {
    id
    name
  }
}
"""

PROJECTS_QUERY_MIN = """
query Projects($after: String) {
  projects(first: 50, after: $after) {
    nodes {
      id
      name
      slugId
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

PROJECTS_QUERY_FULL = """
query Projects($after: String) {
  projects(first: 50, after: $after) {
    nodes {
      id
      name
      slugId
      teams(first: 25) {
        nodes { id key name }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

TEAMS_QUERY = """
query Teams($after: String) {
  teams(first: 50, after: $after) {
    nodes { id key name }
    pageInfo { hasNextPage endCursor }
  }
}
"""

ISSUE_LABELS_QUERY = """
query IssueLabels($labelName: String!, $after: String) {
  issueLabels(
    first: 50,
    after: $after,
    filter: { name: { eq: $labelName } }
  ) {
    nodes { id name }
    pageInfo { hasNextPage endCursor }
  }
}
"""

ISSUE_LABELS_BY_TEAM_QUERY = """
query IssueLabelsByTeam($teamId: ID!, $labelName: String!, $after: String) {
  issueLabels(
    first: 50,
    after: $after,
    filter: { team: { id: { eq: $teamId } }, name: { eq: $labelName } }
  ) {
    nodes { id name }
    pageInfo { hasNextPage endCursor }
  }
}
"""

ISSUE_LABEL_CREATE_MUTATION = """
mutation IssueLabelCreate($name: String!) {
  issueLabelCreate(input: { name: $name }) {
    success
    issueLabel { id name }
  }
}
"""

ISSUE_CREATE_MUTATION = """
mutation IssueCreate(
  $teamId: String!,
  $projectId: String,
  $projectMilestoneId: String,
  $title: String!,
  $description: String,
  $labelIds: [String!],
  $stateId: String
) {
  issueCreate(input: {
    teamId: $teamId,
    projectId: $projectId,
    projectMilestoneId: $projectMilestoneId,
    title: $title,
    description: $description,
    labelIds: $labelIds,
    stateId: $stateId
  }) {
    success
    issue { id identifier title description state { type name } }
  }
}
"""

ISSUE_UPDATE_MUTATION = """
mutation IssueUpdate(
  $id: String!,
  $title: String!,
  $description: String,
  $projectMilestoneId: String,
  $stateId: String!
) {
  issueUpdate(id: $id, input: {
    title: $title,
    description: $description,
    projectMilestoneId: $projectMilestoneId,
    stateId: $stateId
  }) {
    success
    issue { id identifier title description state { type name } }
  }
}
"""

ISSUE_UPDATE_STATE_MUTATION = """
mutation IssueUpdateState($id: String!, $stateId: String!) {
  issueUpdate(id: $id, input: { stateId: $stateId }) {
    success
    issue { id identifier title description state { type name } }
  }
}
"""

ISSUE_UPDATE_LABELS_MUTATION = """
mutation IssueUpdateLabels($id: String!, $labelIds: [String!]) {
  issueUpdate(id: $id, input: { labelIds: $labelIds }) {
    success
    issue { id labelIds }
  }
}
"""

ISSUE_QUERY = """
query Issue($id: String!) {
  issue(id: $id) {
    id
    identifier
    title
    description
    labelIds
    labels(first: 50) { nodes { name } }
    state { type name }
    team {
      id
      states(first: 100) {
        nodes { id name type position }
      }
    }
  }
}
"""

TEAM_STATES_QUERY = """
query TeamStates($teamId: String!, $after: String) {
  team(id: $teamId) {
    id
    states(first: 50, after: $after) {
      nodes { id name type position }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

ISSUE_BLOCKER_RELATIONS_QUERY = """
query IssueBlockerRelations($issueId: String!, $after: String) {
  issue(id: $issueId) {
    id
    inverseRelations(first: 100, after: $after) {
      nodes {
        id
        type
        issue { id }
        relatedIssue { id }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

ISSUE_BLOCKER_RELATIONS_FALLBACK_QUERY = """
query IssueBlockerRelationsFallback($issueId: String!, $after: String) {
  issueRelations(first: 100, after: $after, relatedIssueId: $issueId) {
    nodes {
      id
      type
      issue { id }
      relatedIssue { id }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

ISSUE_RELATION_CREATE_MUTATION = """
mutation IssueRelationCreate($issueId: String!, $relatedIssueId: String!) {
  issueRelationCreate(input: { type: blocks, issueId: $issueId, relatedIssueId: $relatedIssueId }) {
    success
    issueRelation { id type issue { id } relatedIssue { id } }
  }
}
"""

ISSUE_RELATION_DELETE_MUTATION = """
mutation IssueRelationDelete($id: String!) {
  issueRelationDelete(id: $id) {
    success
  }
}
"""


def _maybe_page_info(container: object) -> tuple[bool, str | None]:
    if not isinstance(container, dict):
        return False, None
    container_dict = cast(dict[str, Any], container)
    page_info = container_dict.get("pageInfo")
    if not isinstance(page_info, dict):
        return False, None
    page_info_dict = cast(dict[str, Any], page_info)
    has_next = bool(page_info_dict.get("hasNextPage"))
    end_cursor = page_info_dict.get("endCursor")
    return has_next, end_cursor if isinstance(end_cursor, str) else None


def _parse_issue(node: object) -> LinearIssue | None:
    if not isinstance(node, dict):
        return None
    node_dict = cast(dict[str, Any], node)
    issue_id = node_dict.get("id")
    identifier = node_dict.get("identifier")
    title = node_dict.get("title")
    description = node_dict.get("description")
    state_type = None
    state_name = None
    state = node_dict.get("state")
    if isinstance(state, dict):
        state_dict = cast(dict[str, Any], state)
        st = state_dict.get("type")
        if isinstance(st, str):
            state_type = st
        sn = state_dict.get("name")
        if isinstance(sn, str):
            state_name = sn

    team_id: str | None = None
    team = node_dict.get("team")
    if isinstance(team, dict):
        team_dict = cast(dict[str, Any], team)
        tid = team_dict.get("id")
        if isinstance(tid, str) and tid:
            team_id = tid

    label_ids: tuple[str, ...] = ()
    raw_label_ids = node_dict.get("labelIds")
    if isinstance(raw_label_ids, list):
        label_ids = tuple(v for v in raw_label_ids if isinstance(v, str) and v)

    label_names: tuple[str, ...] = ()
    raw_labels = node_dict.get("labels")
    if isinstance(raw_labels, dict):
        labels_dict = cast(dict[str, Any], raw_labels)
        nodes = labels_dict.get("nodes")
        if isinstance(nodes, list):
            names: list[str] = []
            for item in nodes:
                if not isinstance(item, dict):
                    continue
                name = cast(dict[str, Any], item).get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name)
            label_names = tuple(names)

    team_states: tuple[LinearWorkflowState, ...] = ()
    raw_team_states = None
    if isinstance(team, dict):
        team_dict = cast(dict[str, Any], team)
        raw_team_states = team_dict.get("states")
    if isinstance(raw_team_states, dict):
        states_dict = cast(dict[str, Any], raw_team_states)
        nodes = states_dict.get("nodes")
        if isinstance(nodes, list):
            states: list[LinearWorkflowState] = []
            for item in nodes:
                state = _parse_workflow_state(item)
                if state is not None:
                    states.append(state)
            team_states = tuple(states)

    if (
        not isinstance(issue_id, str)
        or not isinstance(identifier, str)
        or not isinstance(title, str)
    ):
        return None

    return LinearIssue(
        id=issue_id,
        identifier=identifier,
        title=title,
        description=description if isinstance(description, str) else None,
        state_type=state_type,
        state_name=state_name,
        team_id=team_id,
        label_ids=label_ids,
        label_names=label_names,
        team_states=team_states,
    )


def _parse_team(node: object) -> LinearTeam | None:
    if not isinstance(node, dict):
        return None
    node_dict = cast(dict[str, Any], node)
    team_id = node_dict.get("id")
    key = node_dict.get("key")
    name = node_dict.get("name")
    if (
        not isinstance(team_id, str)
        or not isinstance(key, str)
        or not isinstance(name, str)
    ):
        return None
    return LinearTeam(id=team_id, key=key, name=name)


def _parse_label(node: object) -> LinearLabel | None:
    if not isinstance(node, dict):
        return None
    node_dict = cast(dict[str, Any], node)
    label_id = node_dict.get("id")
    name = node_dict.get("name")
    if not isinstance(label_id, str) or not isinstance(name, str):
        return None
    return LinearLabel(id=label_id, name=name)


def _parse_milestone(node: object) -> LinearMilestone | None:
    if not isinstance(node, dict):
        return None
    node_dict = cast(dict[str, Any], node)
    milestone_id = node_dict.get("id")
    name = node_dict.get("name")
    if not isinstance(milestone_id, str) or not isinstance(name, str):
        return None
    return LinearMilestone(id=milestone_id, name=name)


def _parse_project(node: object) -> LinearProject | None:
    if not isinstance(node, dict):
        return None
    node_dict = cast(dict[str, Any], node)
    project_id = node_dict.get("id")
    name = node_dict.get("name")
    slug_id = node_dict.get("slugId")

    if not isinstance(project_id, str) or not isinstance(name, str):
        return None

    teams: list[LinearTeam] = []
    teams_conn = node_dict.get("teams")
    if isinstance(teams_conn, dict):
        teams_nodes = cast(dict[str, Any], teams_conn).get("nodes")
        if isinstance(teams_nodes, list):
            for tnode in teams_nodes:
                team = _parse_team(tnode)
                if team is not None:
                    teams.append(team)

    return LinearProject(
        id=project_id,
        name=name,
        slug=slug_id if isinstance(slug_id, str) and slug_id else None,
        teams=tuple(teams),
    )


def _parse_workflow_state(node: object) -> LinearWorkflowState | None:
    if not isinstance(node, dict):
        return None
    node_dict = cast(dict[str, Any], node)
    state_id = node_dict.get("id")
    name = node_dict.get("name")
    state_type = node_dict.get("type")
    position = node_dict.get("position")
    if (
        not isinstance(state_id, str)
        or not isinstance(name, str)
        or not isinstance(state_type, str)
    ):
        return None
    if position is not None and not isinstance(position, (int, float)):
        position = None
    return LinearWorkflowState(
        id=state_id,
        name=name,
        type=state_type,
        position=float(position) if isinstance(position, (int, float)) else None,
    )


async def fetch_project_issues(
    client: LinearClient, *, project_id: str
) -> list[LinearIssue]:
    issues: list[LinearIssue] = []
    after: str | None = None
    while True:
        data = await client.graphql(
            PROJECT_ISSUES_QUERY, variables={"projectId": project_id, "after": after}
        )
        project = data.get("project")
        if not isinstance(project, dict):
            raise ValueError("Linear: project not found or invalid response")
        project_dict = cast(dict[str, Any], project)

        conn = project_dict.get("issues")
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing project.issues")
        conn_dict = cast(dict[str, Any], conn)

        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing project.issues.nodes")

        for node in nodes:
            issue = _parse_issue(node)
            if issue is not None:
                issues.append(issue)

        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break

    return issues


async def fetch_project_issues_by_label(
    client: LinearClient, *, project_id: str, label_name: str
) -> list[LinearIssue]:
    issues: list[LinearIssue] = []
    after: str | None = None
    while True:
        data = await client.graphql(
            PROJECT_ISSUES_BY_LABEL_QUERY,
            variables={
                "projectId": project_id,
                "labelName": label_name,
                "after": after,
            },
        )

        conn = data.get("issues")
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing issues")
        conn_dict = cast(dict[str, Any], conn)

        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing issues.nodes")

        for node in nodes:
            issue = _parse_issue(node)
            if issue is not None:
                issues.append(issue)

        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break

    return issues


async def fetch_project_issues_by_milestone(
    client: LinearClient, *, project_id: str, milestone_id: str
) -> list[LinearIssue]:
    issues = await fetch_project_issues(client, project_id=project_id)
    scoped: list[LinearIssue] = []
    for issue in issues:
        mid = await fetch_issue_project_milestone_id(client, issue_id=issue.id)
        if mid == milestone_id:
            scoped.append(issue)
    return scoped


async def fetch_project_issue_relations(
    client: LinearClient, *, project_id: str
) -> list[LinearIssueRelation]:
    relations: list[LinearIssueRelation] = []
    after: str | None = None
    while True:
        data = await client.graphql(
            PROJECT_RELATIONS_QUERY, variables={"projectId": project_id, "after": after}
        )
        conn = data.get("issueRelations")
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing issueRelations")
        conn_dict = cast(dict[str, Any], conn)

        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing issueRelations.nodes")

        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_dict = cast(dict[str, Any], node)
            rel_id = node_dict.get("id")
            rel_type = node_dict.get("type")
            issue = node_dict.get("issue")
            related = node_dict.get("relatedIssue")
            if (
                not isinstance(rel_id, str)
                or not isinstance(rel_type, str)
                or not isinstance(issue, dict)
                or not isinstance(related, dict)
            ):
                continue
            issue_dict = cast(dict[str, Any], issue)
            related_dict = cast(dict[str, Any], related)
            issue_id = issue_dict.get("id")
            related_id = related_dict.get("id")
            if not isinstance(issue_id, str) or not isinstance(related_id, str):
                continue
            relations.append(
                LinearIssueRelation(
                    id=rel_id,
                    type=rel_type,
                    issue_id=issue_id,
                    related_issue_id=related_id,
                )
            )

        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break

    return relations


async def fetch_project(
    client: LinearClient, *, project_id: str
) -> LinearProject | None:
    try:
        data = await client.graphql(
            PROJECT_QUERY_MIN, variables={"projectId": project_id}
        )
    except LinearApiError:
        data = await client.graphql(
            PROJECT_QUERY_BARE, variables={"projectId": project_id}
        )
    project = data.get("project")
    return _parse_project(project)


async def fetch_project_teams(
    client: LinearClient, *, project_id: str
) -> list[LinearTeam]:
    data = await client.graphql(
        PROJECT_TEAMS_QUERY, variables={"projectId": project_id}
    )
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("Linear: project not found or invalid response")
    project_dict = cast(dict[str, Any], project)
    teams_conn = project_dict.get("teams")
    if not isinstance(teams_conn, dict):
        raise ValueError("Linear: missing project.teams")
    teams_conn_dict = cast(dict[str, Any], teams_conn)
    nodes = teams_conn_dict.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Linear: missing project.teams.nodes")
    teams: list[LinearTeam] = []
    for node in nodes:
        team = _parse_team(node)
        if team is not None:
            teams.append(team)
    return teams


async def fetch_project_milestones(
    client: LinearClient, *, project_id: str
) -> list[LinearMilestone]:
    milestones: list[LinearMilestone] = []
    after: str | None = None
    query = PROJECT_MILESTONES_QUERY
    while True:
        try:
            data = await client.graphql(
                query, variables={"projectId": project_id, "after": after}
            )
        except LinearApiError:
            if query == PROJECT_MILESTONES_QUERY:
                query = PROJECT_MILESTONES_QUERY_FALLBACK
                after = None
                milestones = []
                continue
            raise

        project = data.get("project")
        if not isinstance(project, dict):
            raise ValueError("Linear: project not found or invalid response")
        project_dict = cast(dict[str, Any], project)
        conn = project_dict.get(
            "projectMilestones" if query == PROJECT_MILESTONES_QUERY else "milestones"
        )
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing project milestones")
        conn_dict = cast(dict[str, Any], conn)
        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing project milestones nodes")
        for node in nodes:
            milestone = _parse_milestone(node)
            if milestone is not None:
                milestones.append(milestone)

        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break

    return milestones


async def fetch_teams(client: LinearClient) -> list[LinearTeam]:
    teams: list[LinearTeam] = []
    after: str | None = None
    while True:
        data = await client.graphql(TEAMS_QUERY, variables={"after": after})
        conn = data.get("teams")
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing teams")
        conn_dict = cast(dict[str, Any], conn)
        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing teams.nodes")
        for node in nodes:
            team = _parse_team(node)
            if team is not None:
                teams.append(team)
        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break
    return teams


async def fetch_projects(client: LinearClient) -> list[LinearProject]:
    after: str | None = None
    projects: list[LinearProject] = []

    query = PROJECTS_QUERY_FULL
    while True:
        try:
            data = await client.graphql(query, variables={"after": after})
        except LinearApiError:
            if query == PROJECTS_QUERY_MIN:
                raise
            # Be conservative: if the workspace GraphQL schema doesn't expose some
            # fields we request (e.g. teams), fall back to a minimal query.
            query = PROJECTS_QUERY_MIN
            after = None
            projects = []
            continue

        conn = data.get("projects")
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing projects")
        conn_dict = cast(dict[str, Any], conn)

        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing projects.nodes")
        for node in nodes:
            project = _parse_project(node)
            if project is not None:
                projects.append(project)

        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break

    return projects


def select_default_team(teams: list[LinearTeam]) -> LinearTeam:
    if not teams:
        raise ValueError("Linear: no teams available for default team selection")
    return sorted(teams, key=lambda t: (t.key.lower(), t.name.lower(), t.id))[0]


async def resolve_default_team(
    client: LinearClient, *, project_id: str, preferred_team_id: str | None = None
) -> LinearTeam:
    if preferred_team_id:
        teams = await fetch_teams(client)
        for team in teams:
            if team.id == preferred_team_id:
                return team
        raise ValueError(f"Linear: preferred team id not found: {preferred_team_id}")

    project_teams = await fetch_project_teams(client, project_id=project_id)
    if project_teams:
        return select_default_team(project_teams)

    teams = await fetch_teams(client)
    return select_default_team(teams)


async def resolve_or_create_label(
    client: LinearClient, *, label_name: str, team_id: str
) -> LinearLabel:
    after: str | None = None
    while True:
        data = await client.graphql(
            ISSUE_LABELS_BY_TEAM_QUERY,
            variables={"teamId": team_id, "labelName": label_name, "after": after},
        )
        conn = data.get("issueLabels")
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing issueLabels")
        conn_dict = cast(dict[str, Any], conn)
        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing issueLabels.nodes")
        for node in nodes:
            label = _parse_label(node)
            if label is not None:
                return label
        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break

    data = await client.graphql(
        ISSUE_LABEL_CREATE_MUTATION, variables={"name": label_name}
    )
    payload = data.get("issueLabelCreate")
    if not isinstance(payload, dict):
        raise ValueError("Linear: missing issueLabelCreate")
    payload_dict = cast(dict[str, Any], payload)
    issue_label = payload_dict.get("issueLabel")
    label = _parse_label(issue_label)
    if label is None:
        raise ValueError("Linear: invalid issueLabelCreate response")
    return label


async def fetch_label_by_name(
    client: LinearClient, *, label_name: str
) -> LinearLabel | None:
    after: str | None = None
    while True:
        data = await client.graphql(
            ISSUE_LABELS_QUERY,
            variables={"labelName": label_name, "after": after},
        )
        conn = data.get("issueLabels")
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing issueLabels")
        conn_dict = cast(dict[str, Any], conn)
        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing issueLabels.nodes")
        for node in nodes:
            label = _parse_label(node)
            if label is not None:
                return label
        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break
    return None


async def fetch_issue(client: LinearClient, *, issue_id: str) -> LinearIssue:
    data = await client.graphql(ISSUE_QUERY, variables={"id": issue_id})
    issue = data.get("issue")
    parsed = _parse_issue(issue)
    if parsed is None:
        raise ValueError("Linear: issue not found or invalid response")
    return parsed


async def fetch_issue_project_milestone_id(
    client: LinearClient, *, issue_id: str
) -> str | None:
    query = ISSUE_MILESTONE_QUERY
    try:
        data = await client.graphql(query, variables={"id": issue_id})
    except LinearApiError:
        query = ISSUE_MILESTONE_QUERY_FALLBACK
        data = await client.graphql(query, variables={"id": issue_id})
    issue = data.get("issue")
    if not isinstance(issue, dict):
        raise ValueError("Linear: issue not found or invalid response")
    issue_dict = cast(dict[str, Any], issue)
    milestone = issue_dict.get(
        "projectMilestone" if query == ISSUE_MILESTONE_QUERY else "milestone"
    )
    if milestone is None:
        return None
    if not isinstance(milestone, dict):
        raise ValueError("Linear: invalid issue milestone payload")
    milestone_dict = cast(dict[str, Any], milestone)
    milestone_id = milestone_dict.get("id")
    return milestone_id if isinstance(milestone_id, str) and milestone_id else None


async def fetch_issue_label_ids(client: LinearClient, *, issue_id: str) -> list[str]:
    data = await client.graphql(ISSUE_QUERY, variables={"id": issue_id})
    issue = data.get("issue")
    if not isinstance(issue, dict):
        raise ValueError("Linear: issue not found or invalid response")
    issue_dict = cast(dict[str, Any], issue)
    label_ids = issue_dict.get("labelIds")
    if not isinstance(label_ids, list):
        raise ValueError("Linear: missing issue.labelIds")
    out: list[str] = []
    for label_id in label_ids:
        if isinstance(label_id, str):
            out.append(label_id)
    return out


async def fetch_issue_team_id(client: LinearClient, *, issue_id: str) -> str:
    data = await client.graphql(ISSUE_QUERY, variables={"id": issue_id})
    issue = data.get("issue")
    if not isinstance(issue, dict):
        raise ValueError("Linear: issue not found or invalid response")
    issue_dict = cast(dict[str, Any], issue)
    team = issue_dict.get("team")
    if not isinstance(team, dict):
        raise ValueError("Linear: missing issue.team")
    team_dict = cast(dict[str, Any], team)
    team_id = team_dict.get("id")
    if not isinstance(team_id, str) or not team_id:
        raise ValueError("Linear: missing issue.team.id")
    return team_id


async def ensure_issue_has_label(
    client: LinearClient, *, issue_id: str, label_id: str
) -> list[str]:
    label_ids = await fetch_issue_label_ids(client, issue_id=issue_id)
    if label_id in label_ids:
        return label_ids

    next_label_ids = sorted(set([*label_ids, label_id]))
    data = await client.graphql(
        ISSUE_UPDATE_LABELS_MUTATION,
        variables={"id": issue_id, "labelIds": next_label_ids},
    )
    payload = data.get("issueUpdate")
    if not isinstance(payload, dict):
        raise ValueError("Linear: missing issueUpdate")
    payload_dict = cast(dict[str, Any], payload)
    issue = payload_dict.get("issue")
    if not isinstance(issue, dict):
        raise ValueError("Linear: missing issueUpdate.issue")
    issue_dict = cast(dict[str, Any], issue)
    label_ids_raw = issue_dict.get("labelIds")
    if not isinstance(label_ids_raw, list):
        raise ValueError("Linear: missing issueUpdate.issue.labelIds")
    return [v for v in label_ids_raw if isinstance(v, str)]


async def create_issue(
    client: LinearClient,
    *,
    team_id: str,
    project_id: str | None,
    project_milestone_id: str | None = None,
    title: str,
    description: str | None,
    label_ids: list[str] | None = None,
    state_id: str | None = None,
) -> LinearIssue:
    data = await client.graphql(
        ISSUE_CREATE_MUTATION,
        variables={
            "teamId": team_id,
            "projectId": project_id,
            "projectMilestoneId": project_milestone_id,
            "title": title,
            "description": description,
            "labelIds": label_ids,
            "stateId": state_id,
        },
    )
    payload = data.get("issueCreate")
    if not isinstance(payload, dict):
        raise ValueError("Linear: missing issueCreate")
    payload_dict = cast(dict[str, Any], payload)
    issue = payload_dict.get("issue")
    parsed = _parse_issue(issue)
    if parsed is None:
        raise ValueError("Linear: invalid issueCreate response")
    return parsed


async def update_issue(
    client: LinearClient,
    *,
    issue_id: str,
    title: str,
    description: str | None = None,
    project_milestone_id: str | None = None,
    state_id: str,
) -> LinearIssue:
    data = await client.graphql(
        ISSUE_UPDATE_MUTATION,
        variables={
            "id": issue_id,
            "title": title,
            "description": description,
            "projectMilestoneId": project_milestone_id,
            "stateId": state_id,
        },
    )
    payload = data.get("issueUpdate")
    if not isinstance(payload, dict):
        raise ValueError("Linear: missing issueUpdate")
    payload_dict = cast(dict[str, Any], payload)
    issue = payload_dict.get("issue")
    parsed = _parse_issue(issue)
    if parsed is None:
        raise ValueError("Linear: invalid issueUpdate response")
    return parsed


async def update_issue_state(
    client: LinearClient,
    *,
    issue_id: str,
    state_id: str,
) -> LinearIssue:
    data = await client.graphql(
        ISSUE_UPDATE_STATE_MUTATION,
        variables={
            "id": issue_id,
            "stateId": state_id,
        },
    )
    payload = data.get("issueUpdate")
    if not isinstance(payload, dict):
        raise ValueError("Linear: missing issueUpdate")
    payload_dict = cast(dict[str, Any], payload)
    issue = payload_dict.get("issue")
    parsed = _parse_issue(issue)
    if parsed is None:
        raise ValueError("Linear: invalid issueUpdate response")
    return parsed


async def fetch_team_states(
    client: LinearClient, *, team_id: str
) -> list[LinearWorkflowState]:
    states: list[LinearWorkflowState] = []
    after: str | None = None
    while True:
        data = await client.graphql(
            TEAM_STATES_QUERY, variables={"teamId": team_id, "after": after}
        )
        team = data.get("team")
        if not isinstance(team, dict):
            raise ValueError("Linear: team not found or invalid response")
        team_dict = cast(dict[str, Any], team)
        conn = team_dict.get("states")
        if not isinstance(conn, dict):
            raise ValueError("Linear: missing team.states")
        conn_dict = cast(dict[str, Any], conn)
        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing team.states.nodes")
        for node in nodes:
            state = _parse_workflow_state(node)
            if state is not None:
                states.append(state)
        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break
    return states


def resolve_team_state_id_from_states(
    states: Sequence[LinearWorkflowState],
    *,
    state_type: str,
    pick: Literal["first", "last"] = "first",
) -> str:
    normalized = state_type.lower().strip()
    candidates = [s for s in states if s.type.lower() == normalized]
    if not candidates:
        raise ValueError(f"Linear: no workflow state found for type={state_type!r}")

    def _rank(state: LinearWorkflowState) -> tuple[float, str]:
        if state.position is None:
            pos = float("inf") if pick == "first" else float("-inf")
        else:
            pos = state.position
        return (pos, state.id)

    ordered = sorted(candidates, key=_rank)
    return ordered[0].id if pick == "first" else ordered[-1].id


async def resolve_team_state_id(
    client: LinearClient,
    *,
    team_id: str,
    state_type: str,
    pick: Literal["first", "last"] = "first",
) -> str:
    normalized = state_type.lower().strip()
    cache_key = (team_id, normalized, pick)
    now = time.monotonic()
    cached = _team_state_id_cache.get(cache_key)
    if cached is not None:
        state_id, expires_at = cached
        if expires_at > now:
            return state_id
        _team_state_id_cache.pop(cache_key, None)

    states = await fetch_team_states(client, team_id=team_id)
    state_id = resolve_team_state_id_from_states(
        states, state_type=state_type, pick=pick
    )
    _team_state_id_cache[cache_key] = (state_id, now + _TEAM_STATE_ID_CACHE_TTL_S)
    return state_id


async def fetch_issue_blocker_relations(
    client: LinearClient, *, issue_id: str
) -> list[LinearIssueRelation]:
    relations: list[LinearIssueRelation] = []
    after: str | None = None
    while True:
        try:
            data = await client.graphql(
                ISSUE_BLOCKER_RELATIONS_QUERY,
                variables={"issueId": issue_id, "after": after},
            )
            issue = data.get("issue")
            if not isinstance(issue, dict):
                raise ValueError("Linear: issue not found or invalid response")
            issue_dict = cast(dict[str, Any], issue)
            conn = issue_dict.get("inverseRelations")
            if not isinstance(conn, dict):
                raise ValueError("Linear: missing issue.inverseRelations")
            conn_dict = cast(dict[str, Any], conn)
        except LinearApiError as e:
            if e.code != "GRAPHQL_VALIDATION_FAILED":
                raise
            data = await client.graphql(
                ISSUE_BLOCKER_RELATIONS_FALLBACK_QUERY,
                variables={"issueId": issue_id, "after": after},
            )
            conn = data.get("issueRelations")
            if not isinstance(conn, dict):
                raise ValueError("Linear: missing issueRelations")
            conn_dict = cast(dict[str, Any], conn)
        nodes = conn_dict.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("Linear: missing issue relations nodes")
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_dict = cast(dict[str, Any], node)
            rel_id = node_dict.get("id")
            rel_type = node_dict.get("type")
            issue = node_dict.get("issue")
            related = node_dict.get("relatedIssue")
            if (
                not isinstance(rel_id, str)
                or not isinstance(rel_type, str)
                or not isinstance(issue, dict)
                or not isinstance(related, dict)
            ):
                continue
            issue_dict = cast(dict[str, Any], issue)
            related_dict = cast(dict[str, Any], related)
            issue_inner_id = issue_dict.get("id")
            related_id = related_dict.get("id")
            if not isinstance(issue_inner_id, str) or not isinstance(related_id, str):
                continue
            relations.append(
                LinearIssueRelation(
                    id=rel_id,
                    type=rel_type,
                    issue_id=issue_inner_id,
                    related_issue_id=related_id,
                )
            )

        has_next, end_cursor = _maybe_page_info(conn_dict)
        if not has_next:
            break
        after = end_cursor
        if after is None:
            break
    return relations


async def fetch_issue_blocker_ids(client: LinearClient, *, issue_id: str) -> list[str]:
    relations = await fetch_issue_blocker_relations(client, issue_id=issue_id)
    blockers: list[str] = []
    for rel in relations:
        if rel.type.lower() != "blocks":
            continue
        if rel.related_issue_id != issue_id:
            continue
        blockers.append(rel.issue_id)
    return blockers


async def set_issue_blockers(
    client: LinearClient, *, issue_id: str, blocker_issue_ids: list[str]
) -> None:
    desired = {bid for bid in blocker_issue_ids if bid and bid != issue_id}
    existing = await fetch_issue_blocker_relations(client, issue_id=issue_id)

    existing_by_blocker: dict[str, list[LinearIssueRelation]] = {}
    for rel in existing:
        if rel.type.lower() != "blocks":
            continue
        if rel.related_issue_id != issue_id:
            continue
        existing_by_blocker.setdefault(rel.issue_id, []).append(rel)

    for blocker_id in sorted(desired - set(existing_by_blocker.keys())):
        await client.graphql(
            ISSUE_RELATION_CREATE_MUTATION,
            variables={"issueId": blocker_id, "relatedIssueId": issue_id},
        )

    for blocker_id in sorted(set(existing_by_blocker.keys()) - desired):
        for rel in existing_by_blocker[blocker_id]:
            await client.graphql(
                ISSUE_RELATION_DELETE_MUTATION, variables={"id": rel.id}
            )
