from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class RedesmynSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REDESMYN_", env_file=".env", extra="ignore"
    )

    repo_root: Path | None = None
    state_dir_name: str = ".redesmyn"
    db_filename: str = "redesmyn.sqlite3"

    api_host: str = "127.0.0.1"
    api_port: int = 9234

    linear_client_id: str | None = None
    linear_client_secret: str | None = None
    linear_scopes: str = "read"


def load_settings(*, repo_root: Path | None = None) -> RedesmynSettings:
    env_file = repo_root / ".env" if repo_root is not None else None
    kwargs: dict[str, Any] = {}
    if env_file is not None:
        kwargs["_env_file"] = env_file
    return RedesmynSettings(**kwargs)
