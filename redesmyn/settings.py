from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class RedesmynSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDESMYN_", env_file=".env", extra="ignore")

    repo_root: Path | None = None
    state_dir_name: str = ".redesmyn"
    db_filename: str = "redesmyn.sqlite3"

    api_host: str = "127.0.0.1"
    api_port: int = 9234

