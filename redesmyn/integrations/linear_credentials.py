from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import keyring
from keyring.errors import PasswordDeleteError

_SERVICE_NAME = "redesmyn.linear"
_ACCOUNT_NAME = "oauth"


@dataclass(frozen=True, slots=True)
class LinearCredentials:
    access_token: str
    refresh_token: str | None
    token_type: str
    scope: str | None
    expires_at: datetime | None
    connected_at: datetime

    def to_json(self) -> str:
        return json.dumps(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "token_type": self.token_type,
                "scope": self.scope,
                "expires_at": self.expires_at.isoformat() if self.expires_at else None,
                "connected_at": self.connected_at.isoformat(),
            },
            sort_keys=True,
        )

    @staticmethod
    def from_json(value: str) -> "LinearCredentials":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("Invalid Linear credentials payload (not an object)")

        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError(
                "Invalid Linear credentials payload (missing access_token)"
            )

        refresh_token = payload.get("refresh_token")
        if refresh_token is not None and not isinstance(refresh_token, str):
            raise ValueError(
                "Invalid Linear credentials payload (invalid refresh_token)"
            )

        token_type = payload.get("token_type")
        if not isinstance(token_type, str) or not token_type:
            raise ValueError("Invalid Linear credentials payload (missing token_type)")

        scope = payload.get("scope")
        if scope is not None and not isinstance(scope, str):
            raise ValueError("Invalid Linear credentials payload (invalid scope)")

        expires_at_raw = payload.get("expires_at")
        expires_at: datetime | None
        if expires_at_raw is None:
            expires_at = None
        elif isinstance(expires_at_raw, str):
            expires_at = datetime.fromisoformat(expires_at_raw)
        else:
            raise ValueError("Invalid Linear credentials payload (invalid expires_at)")

        connected_at_raw = payload.get("connected_at")
        if not isinstance(connected_at_raw, str):
            raise ValueError(
                "Invalid Linear credentials payload (missing connected_at)"
            )
        connected_at = datetime.fromisoformat(connected_at_raw)

        if expires_at is not None and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if connected_at.tzinfo is None:
            connected_at = connected_at.replace(tzinfo=UTC)

        return LinearCredentials(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type=token_type,
            scope=scope,
            expires_at=expires_at,
            connected_at=connected_at,
        )


class LinearCredentialStore(Protocol):
    def get(self) -> LinearCredentials | None: ...

    def set(self, credentials: LinearCredentials) -> None: ...

    def clear(self) -> None: ...


class KeyringLinearCredentialStore:
    def __init__(
        self,
        *,
        service_name: str = _SERVICE_NAME,
        account_name: str = _ACCOUNT_NAME,
    ) -> None:
        self._service_name = service_name
        self._account_name = account_name

    def get(self) -> LinearCredentials | None:
        value = keyring.get_password(self._service_name, self._account_name)
        if value is None:
            return None
        return LinearCredentials.from_json(value)

    def set(self, credentials: LinearCredentials) -> None:
        keyring.set_password(
            self._service_name, self._account_name, credentials.to_json()
        )

    def clear(self) -> None:
        try:
            keyring.delete_password(self._service_name, self._account_name)
        except PasswordDeleteError:
            return


def default_linear_credential_store() -> LinearCredentialStore:
    return KeyringLinearCredentialStore()


def is_expiring_soon(
    credentials: LinearCredentials, *, now: datetime | None = None, skew: timedelta
) -> bool:
    if credentials.expires_at is None:
        return False
    if now is None:
        now = datetime.now(UTC)
    return credentials.expires_at - now <= skew
