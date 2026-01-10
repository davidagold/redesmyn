from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from redesmyn.settings import RedesmynSettings

GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"


class GitHubOAuthError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GitHubDeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    expires_in: int
    interval: int


@dataclass(frozen=True, slots=True)
class GitHubToken:
    access_token: str
    token_type: str
    scope: str | None = None


def _normalize_scopes(scopes_raw: str) -> str:
    parts: list[str] = []
    for part in scopes_raw.replace(",", " ").split():
        p = part.strip()
        if not p:
            continue
        if p not in parts:
            parts.append(p)
    return " ".join(parts)


async def request_device_code(
    settings: RedesmynSettings, *, scopes: str | None = None
) -> GitHubDeviceCode:
    if not settings.github_client_id:
        raise GitHubOAuthError("Missing REDESMYN_GITHUB_CLIENT_ID")

    scope_value = _normalize_scopes(scopes or settings.github_scopes or "")
    data = {"client_id": settings.github_client_id}
    if scope_value:
        data["scope"] = scope_value

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            GITHUB_DEVICE_CODE_URL,
            data=data,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        payload: dict[str, object] = resp.json()

    device_code = payload.get("device_code")
    user_code = payload.get("user_code")
    verification_uri = payload.get("verification_uri")
    verification_uri_complete = payload.get("verification_uri_complete")
    expires_in = payload.get("expires_in")
    interval = payload.get("interval")

    if not isinstance(device_code, str) or not device_code:
        raise GitHubOAuthError("GitHub device code response missing device_code")
    if not isinstance(user_code, str) or not user_code:
        raise GitHubOAuthError("GitHub device code response missing user_code")
    if not isinstance(verification_uri, str) or not verification_uri:
        raise GitHubOAuthError("GitHub device code response missing verification_uri")
    if verification_uri_complete is not None and not isinstance(
        verification_uri_complete, str
    ):
        raise GitHubOAuthError(
            "GitHub device code response has invalid verification_uri_complete"
        )
    if not isinstance(expires_in, int) or expires_in <= 0:
        raise GitHubOAuthError("GitHub device code response missing expires_in")
    if not isinstance(interval, int) or interval <= 0:
        interval = 5

    return GitHubDeviceCode(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        verification_uri_complete=verification_uri_complete,
        expires_in=expires_in,
        interval=interval,
    )


async def poll_device_token(
    settings: RedesmynSettings,
    *,
    device_code: str,
    timeout_seconds: int,
    interval_seconds: int,
) -> GitHubToken:
    if not settings.github_client_id:
        raise GitHubOAuthError("Missing REDESMYN_GITHUB_CLIENT_ID")

    async with httpx.AsyncClient(timeout=20.0) as client:
        waited = 0.0
        interval = max(1.0, float(interval_seconds))
        while waited <= float(timeout_seconds):
            resp = await client.post(
                GITHUB_TOKEN_URL,
                data={
                    "client_id": settings.github_client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            payload: dict[str, object] = resp.json()

            access_token = payload.get("access_token")
            if isinstance(access_token, str) and access_token:
                token_type = payload.get("token_type") or "Bearer"
                scope = payload.get("scope")
                return GitHubToken(
                    access_token=access_token,
                    token_type=str(token_type),
                    scope=scope if isinstance(scope, str) else None,
                )

            error = payload.get("error")
            if error == "authorization_pending":
                pass
            elif error == "slow_down":
                interval += 5.0
            elif error == "expired_token":
                raise GitHubOAuthError("GitHub device code expired; try again")
            elif error == "access_denied":
                raise GitHubOAuthError("GitHub authorization denied")
            elif isinstance(error, str) and error:
                raise GitHubOAuthError(f"GitHub device authorization failed: {error}")
            else:
                raise GitHubOAuthError(
                    "GitHub device authorization failed: unexpected response"
                )

            await asyncio.sleep(interval)
            waited += interval

    raise GitHubOAuthError("Timed out waiting for GitHub authorization")
