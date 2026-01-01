from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import timedelta
from typing import Any, cast
from urllib.parse import urlencode

import httpx

from redesmyn.settings import RedesmynSettings

LINEAR_OAUTH_AUTHORIZE_URL = "https://linear.app/oauth/authorize"
LINEAR_OAUTH_TOKEN_URL = "https://api.linear.app/oauth/token"
LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"


@dataclass(frozen=True, slots=True)
class LinearToken:
    access_token: str
    refresh_token: str | None
    token_type: str
    scope: str | None
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class LinearIssue:
    id: str
    identifier: str
    title: str
    description: str | None
    state_type: str | None


@dataclass(frozen=True, slots=True)
class LinearIssueRelation:
    id: str
    type: str
    issue_id: str
    related_issue_id: str


@dataclass(frozen=True, slots=True)
class LinearTeam:
    id: str
    key: str
    name: str


@dataclass(frozen=True, slots=True)
class LinearLabel:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class LinearWorkflowState:
    id: str
    name: str
    type: str
    position: float | None


def linear_redirect_uri(settings: RedesmynSettings) -> str:
    return f"http://{settings.api_host}:{settings.api_port}/v1/linear/oauth/callback"


def new_oauth_state() -> str:
    return secrets.token_urlsafe(24)


def new_pkce_verifier() -> str:
    # RFC 7636: 43-128 chars. token_urlsafe() yields URL-safe base64 characters.
    verifier = secrets.token_urlsafe(64)
    return verifier[:128]


def pkce_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def linear_authorize_url(
    settings: RedesmynSettings,
    *,
    state: str,
    redirect_uri: str | None = None,
    code_challenge: str | None = None,
) -> str:
    client_id = settings.linear_client_id
    if not client_id:
        raise ValueError("Missing REDESMYN_LINEAR_CLIENT_ID")

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri or linear_redirect_uri(settings),
        "scope": settings.linear_scopes,
        "state": state,
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    return f"{LINEAR_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_token(
    settings: RedesmynSettings,
    *,
    code: str,
    redirect_uri: str,
    code_verifier: str | None = None,
) -> LinearToken:
    client_id = settings.linear_client_id
    client_secret = settings.linear_client_secret
    if not client_id:
        raise ValueError("Missing REDESMYN_LINEAR_CLIENT_ID")
    if not client_secret and not code_verifier:
        raise ValueError(
            "Missing REDESMYN_LINEAR_CLIENT_SECRET (or PKCE code_verifier)"
        )

    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    if client_secret:
        data["client_secret"] = client_secret
    if code_verifier:
        data["code_verifier"] = code_verifier

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(LINEAR_OAUTH_TOKEN_URL, data=data)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()

    access_token = payload.get("access_token") or payload.get("accessToken")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("Linear OAuth token response missing access_token")

    refresh_token = payload.get("refresh_token") or payload.get("refreshToken")
    token_type = payload.get("token_type") or payload.get("tokenType") or "Bearer"
    scope = payload.get("scope")

    expires_in = payload.get("expires_in") or payload.get("expiresIn")
    expires_at: datetime | None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = datetime.now(UTC) + timedelta(seconds=int(expires_in))
    else:
        expires_at = None

    return LinearToken(
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) else None,
        token_type=str(token_type),
        scope=scope if isinstance(scope, str) else None,
        expires_at=expires_at,
    )


async def refresh_access_token(
    settings: RedesmynSettings, *, refresh_token: str
) -> LinearToken:
    client_id = settings.linear_client_id
    if not client_id:
        raise ValueError("Missing REDESMYN_LINEAR_CLIENT_ID")

    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    if settings.linear_client_secret:
        data["client_secret"] = settings.linear_client_secret

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(LINEAR_OAUTH_TOKEN_URL, data=data)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()

    access_token = payload.get("access_token") or payload.get("accessToken")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("Linear OAuth refresh response missing access_token")

    next_refresh_token = payload.get("refresh_token") or payload.get("refreshToken")
    token_type = payload.get("token_type") or payload.get("tokenType") or "Bearer"
    scope = payload.get("scope")

    expires_in = payload.get("expires_in") or payload.get("expiresIn")
    expires_at: datetime | None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = datetime.now(UTC) + timedelta(seconds=int(expires_in))
    else:
        expires_at = None

    return LinearToken(
        access_token=access_token,
        refresh_token=next_refresh_token
        if isinstance(next_refresh_token, str)
        else None,
        token_type=str(token_type),
        scope=scope if isinstance(scope, str) else None,
        expires_at=expires_at,
    )


class LinearClient:
    def __init__(self, *, access_token: str):
        self._access_token = access_token

    async def graphql(
        self, query: str, variables: dict[str, object] | None = None
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                LINEAR_GRAPHQL_URL,
                json={"query": query, "variables": variables or {}},
                headers=headers,
            )
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
        if "errors" in payload:
            raise ValueError(f"Linear GraphQL error: {payload['errors']}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("Linear GraphQL response missing data")
        return data


PROJECT_ISSUES_QUERY = """
query ProjectIssues($projectId: String!, $after: String) {
  project(id: $projectId) {
    id
    name
    issues(first: 50, after: $after) {
      nodes {
        id
        identifier
        title
        description
        state { type }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

PROJECT_ISSUES_BY_LABEL_QUERY = """
query ProjectIssuesByLabel($projectId: String!, $labelName: String!, $after: String) {
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
      state { type }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""

PROJECT_RELATIONS_QUERY = """
query ProjectIssueRelations($projectId: String!, $after: String) {
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
query ProjectTeams($projectId: String!) {
  project(id: $projectId) {
    id
    teams(first: 50) {
      nodes { id key name }
    }
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
query IssueLabels($teamId: String, $labelName: String!, $after: String) {
  issueLabels(
    first: 50,
    after: $after,
    filter: { name: { eq: $labelName }, team: { id: { eq: $teamId } } }
  ) {
    nodes { id name }
    pageInfo { hasNextPage endCursor }
  }
}
"""

ISSUE_LABEL_CREATE_MUTATION = """
mutation IssueLabelCreate($name: String!, $teamId: String) {
  issueLabelCreate(input: { name: $name, teamId: $teamId }) {
    success
    issueLabel { id name }
  }
}
"""

ISSUE_CREATE_MUTATION = """
mutation IssueCreate(
  $teamId: String!,
  $projectId: String,
  $title: String!,
  $description: String,
  $labelIds: [String!],
  $stateId: String
) {
  issueCreate(input: {
    teamId: $teamId,
    projectId: $projectId,
    title: $title,
    description: $description,
    labelIds: $labelIds,
    stateId: $stateId
  }) {
    success
    issue { id identifier title description state { type } }
  }
}
"""

ISSUE_UPDATE_MUTATION = """
mutation IssueUpdate(
  $id: String!,
  $title: String!,
  $description: String,
  $stateId: String!
) {
  issueUpdate(id: $id, input: {
    title: $title,
    description: $description,
    stateId: $stateId
  }) {
    success
    issue { id identifier title description state { type } }
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
    state { type }
    team { id }
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
  issueRelations(
    first: 100,
    after: $after,
    filter: {
      type: { eq: blocks },
      relatedIssue: { id: { eq: $issueId } }
    }
  ) {
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
    state = node_dict.get("state")
    if isinstance(state, dict):
        state_dict = cast(dict[str, Any], state)
        st = state_dict.get("type")
        if isinstance(st, str):
            state_type = st

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
            ISSUE_LABELS_QUERY,
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
        ISSUE_LABEL_CREATE_MUTATION, variables={"name": label_name, "teamId": team_id}
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


async def fetch_issue(client: LinearClient, *, issue_id: str) -> LinearIssue:
    data = await client.graphql(ISSUE_QUERY, variables={"id": issue_id})
    issue = data.get("issue")
    parsed = _parse_issue(issue)
    if parsed is None:
        raise ValueError("Linear: issue not found or invalid response")
    return parsed


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
    state_id: str,
) -> LinearIssue:
    data = await client.graphql(
        ISSUE_UPDATE_MUTATION,
        variables={
            "id": issue_id,
            "title": title,
            "description": description,
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


async def resolve_team_state_id(
    client: LinearClient, *, team_id: str, state_type: str
) -> str:
    normalized = state_type.lower().strip()
    states = await fetch_team_states(client, team_id=team_id)
    candidates = [s for s in states if s.type.lower() == normalized]
    if not candidates:
        raise ValueError(f"Linear: no workflow state found for type={state_type!r}")

    def _rank(state: LinearWorkflowState) -> tuple[float, str]:
        pos = state.position if state.position is not None else float("inf")
        return (pos, state.id)

    return sorted(candidates, key=_rank)[0].id


async def fetch_issue_blocker_relations(
    client: LinearClient, *, issue_id: str
) -> list[LinearIssueRelation]:
    relations: list[LinearIssueRelation] = []
    after: str | None = None
    while True:
        data = await client.graphql(
            ISSUE_BLOCKER_RELATIONS_QUERY,
            variables={"issueId": issue_id, "after": after},
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
