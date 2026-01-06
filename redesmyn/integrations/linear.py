from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from urllib.parse import quote, urlencode

import httpx

from redesmyn.integrations.linear_client import LinearApiError, LinearClient
from redesmyn.integrations.linear_models import (
    LinearIssue,
    LinearIssueRelation,
    LinearLabel,
    LinearMilestone,
    LinearProject,
    LinearTeam,
    LinearToken,
    LinearWorkflowState,
)
from redesmyn.integrations.linear_sync import (
    PROJECT_ISSUES_BY_LABEL_QUERY,
    PROJECT_ISSUES_QUERY,
    PROJECT_MILESTONES_QUERY,
    PROJECT_MILESTONES_QUERY_FALLBACK,
    PROJECT_QUERY_BARE,
    PROJECT_QUERY_MIN,
    PROJECT_RELATIONS_QUERY,
    PROJECT_TEAMS_QUERY,
    PROJECT_URL_QUERY,
    create_issue,
    ensure_issue_has_label,
    fetch_issue,
    fetch_issue_blocker_ids,
    fetch_issue_project_milestone_id,
    fetch_issue_team_id,
    fetch_issue_url,
    fetch_label_by_name,
    fetch_project,
    fetch_project_issue_relations,
    fetch_project_issues,
    fetch_project_issues_by_label,
    fetch_project_issues_by_milestone,
    fetch_project_milestones,
    fetch_project_url,
    fetch_projects,
    resolve_default_team,
    resolve_or_create_label,
    resolve_team_state_id,
    resolve_team_state_id_from_states,
    set_issue_blockers,
    update_issue,
    update_issue_state,
)
from redesmyn.settings import RedesmynSettings

LINEAR_OAUTH_AUTHORIZE_URL = "https://linear.app/oauth/authorize"
LINEAR_OAUTH_TOKEN_URL = "https://api.linear.app/oauth/token"


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

    scopes_raw = settings.linear_scopes or ""
    # Linear expects a comma-separated list of scopes. Accept either commas or
    # spaces from config/env and normalize.
    scopes: list[str] = []
    for part in scopes_raw.replace(",", " ").split():
        p = part.strip()
        if not p:
            continue
        if p not in scopes:
            scopes.append(p)
    scope_value = ",".join(scopes) if scopes else "read"

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri or linear_redirect_uri(settings),
        "scope": scope_value,
        "state": state,
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    # Linear expects `scope` as a comma-separated list, and their authorize
    # endpoint appears sensitive to commas being percent-encoded.
    return (
        f"{LINEAR_OAUTH_AUTHORIZE_URL}?{urlencode(params, quote_via=quote, safe=',')}"
    )


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
        payload: dict[str, object] = resp.json()

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
        payload: dict[str, object] = resp.json()

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


__all__ = [
    "LINEAR_OAUTH_AUTHORIZE_URL",
    "LINEAR_OAUTH_TOKEN_URL",
    "LinearApiError",
    "LinearClient",
    "LinearIssue",
    "LinearIssueRelation",
    "LinearLabel",
    "LinearMilestone",
    "LinearProject",
    "LinearTeam",
    "LinearToken",
    "LinearWorkflowState",
    "PROJECT_ISSUES_BY_LABEL_QUERY",
    "PROJECT_ISSUES_QUERY",
    "PROJECT_MILESTONES_QUERY",
    "PROJECT_MILESTONES_QUERY_FALLBACK",
    "PROJECT_QUERY_BARE",
    "PROJECT_QUERY_MIN",
    "PROJECT_RELATIONS_QUERY",
    "PROJECT_TEAMS_QUERY",
    "PROJECT_URL_QUERY",
    "create_issue",
    "ensure_issue_has_label",
    "exchange_code_for_token",
    "fetch_issue",
    "fetch_issue_blocker_ids",
    "fetch_issue_project_milestone_id",
    "fetch_issue_team_id",
    "fetch_issue_url",
    "fetch_label_by_name",
    "fetch_project",
    "fetch_project_issue_relations",
    "fetch_project_issues",
    "fetch_project_issues_by_label",
    "fetch_project_issues_by_milestone",
    "fetch_project_milestones",
    "fetch_project_url",
    "fetch_projects",
    "linear_authorize_url",
    "linear_redirect_uri",
    "new_oauth_state",
    "new_pkce_verifier",
    "pkce_code_challenge",
    "refresh_access_token",
    "resolve_default_team",
    "resolve_or_create_label",
    "resolve_team_state_id",
    "resolve_team_state_id_from_states",
    "set_issue_blockers",
    "update_issue",
    "update_issue_state",
]
