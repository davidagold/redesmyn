from __future__ import annotations

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
    type: str
    issue_id: str
    related_issue_id: str


def linear_redirect_uri(settings: RedesmynSettings) -> str:
    return f"http://{settings.api_host}:{settings.api_port}/v1/linear/oauth/callback"


def new_oauth_state() -> str:
    return secrets.token_urlsafe(24)


def linear_authorize_url(
    settings: RedesmynSettings, *, state: str, redirect_uri: str | None = None
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
    return f"{LINEAR_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_token(
    settings: RedesmynSettings, *, code: str, redirect_uri: str
) -> LinearToken:
    client_id = settings.linear_client_id
    client_secret = settings.linear_client_secret
    if not client_id:
        raise ValueError("Missing REDESMYN_LINEAR_CLIENT_ID")
    if not client_secret:
        raise ValueError("Missing REDESMYN_LINEAR_CLIENT_SECRET")

    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "code": code,
    }

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
            if not isinstance(node, dict):
                continue
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
                continue

            issues.append(
                LinearIssue(
                    id=issue_id,
                    identifier=identifier,
                    title=title,
                    description=description if isinstance(description, str) else None,
                    state_type=state_type,
                )
            )

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
            rel_type = node_dict.get("type")
            issue = node_dict.get("issue")
            related = node_dict.get("relatedIssue")
            if (
                not isinstance(rel_type, str)
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
