from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from redesmyn.context import RepoContext
from redesmyn.orchestration_config import (
    global_config_path,
    read_config_file,
    repo_config_path,
    set_config_value,
    write_config,
)

GitHubConfigScope = Literal["repo", "global"]


class GitHubIntegrationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    auto_force_push: bool = False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
            and merged.get(key) is not None
        ):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _config_path_for_scope(ctx: RepoContext, scope: GitHubConfigScope) -> Path:
    if scope == "repo":
        return repo_config_path(ctx)
    if scope == "global":
        return global_config_path()
    raise ValueError(f"Unknown scope: {scope!r}")


def load_github_integration_config(ctx: RepoContext) -> GitHubIntegrationConfig:
    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, read_config_file(global_config_path()))
    merged = _deep_merge(merged, read_config_file(repo_config_path(ctx)))
    github = merged.get("github")
    if not isinstance(github, dict):
        github = {}
    return GitHubIntegrationConfig.model_validate(github)


@dataclass(frozen=True, slots=True)
class GitHubIntegrationConfigUpdate:
    auto_force_push: bool | None = None

    def to_key_values(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.auto_force_push is not None:
            out["github.auto_force_push"] = self.auto_force_push
        return out


def update_github_integration_config(
    ctx: RepoContext,
    *,
    scope: GitHubConfigScope,
    update: GitHubIntegrationConfigUpdate,
) -> None:
    """
    Update GitHub integration settings in the global or repo config layer.

    Values are stored under the top-level TOML table `github`.
    """
    path = _config_path_for_scope(ctx, scope)
    data = read_config_file(path)
    for key, value in update.to_key_values().items():
        set_config_value(data, key, value)
    write_config(path, data)
