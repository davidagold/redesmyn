from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import timedelta
from typing import Any
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

    async def graphql(self, query: str, variables: dict[str, object] | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                LINEAR_GRAPHQL_URL, json={"query": query, "variables": variables or {}}, headers=headers
            )
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
        if "errors" in payload:
            raise ValueError(f"Linear GraphQL error: {payload['errors']}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("Linear GraphQL response missing data")
        return data
