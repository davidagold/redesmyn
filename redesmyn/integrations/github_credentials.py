from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import keyring
from keyring.errors import PasswordDeleteError

_SERVICE_NAME = "redesmyn.github"
_ACCOUNT_NAME = "oauth"


@dataclass(frozen=True, slots=True)
class GitHubCredentials:
    access_token: str
    token_type: str
    connected_at: datetime

    def to_json(self) -> str:
        return json.dumps(
            {
                "access_token": self.access_token,
                "token_type": self.token_type,
                "connected_at": self.connected_at.isoformat(),
            },
            sort_keys=True,
        )

    @staticmethod
    def from_json(value: str) -> "GitHubCredentials":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("Invalid GitHub credentials payload (not an object)")

        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError(
                "Invalid GitHub credentials payload (missing access_token)"
            )

        token_type = payload.get("token_type")
        if not isinstance(token_type, str) or not token_type:
            raise ValueError("Invalid GitHub credentials payload (missing token_type)")

        connected_at_raw = payload.get("connected_at")
        if not isinstance(connected_at_raw, str):
            raise ValueError(
                "Invalid GitHub credentials payload (missing connected_at)"
            )
        connected_at = datetime.fromisoformat(connected_at_raw)
        if connected_at.tzinfo is None:
            connected_at = connected_at.replace(tzinfo=UTC)

        return GitHubCredentials(
            access_token=access_token,
            token_type=token_type,
            connected_at=connected_at,
        )


class GitHubCredentialStore(Protocol):
    def get(self) -> GitHubCredentials | None: ...

    def set(self, credentials: GitHubCredentials) -> None: ...

    def clear(self) -> None: ...


class KeyringGitHubCredentialStore:
    def __init__(
        self,
        *,
        service_name: str = _SERVICE_NAME,
        account_name: str = _ACCOUNT_NAME,
    ) -> None:
        self._service_name = service_name
        self._account_name = account_name

    def get(self) -> GitHubCredentials | None:
        value = keyring.get_password(self._service_name, self._account_name)
        if value is None:
            return None
        return GitHubCredentials.from_json(value)

    def set(self, credentials: GitHubCredentials) -> None:
        keyring.set_password(
            self._service_name, self._account_name, credentials.to_json()
        )

    def clear(self) -> None:
        try:
            keyring.delete_password(self._service_name, self._account_name)
        except PasswordDeleteError:
            return


def default_github_credential_store() -> GitHubCredentialStore:
    return KeyringGitHubCredentialStore()
