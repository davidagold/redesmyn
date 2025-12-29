from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from redesmyn.context import RepoContext

FleetMode = Literal["fixed", "auto"]


class FleetDefaults(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: FleetMode = "fixed"
    size: int | None = None

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("fleet.size must be > 0")
        return value


class HarnessDefaults(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: str | None = None
    detach: bool = True
    prelude: str | None = None

    @field_validator("command")
    @classmethod
    def _normalize_command(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("prelude")
    @classmethod
    def _normalize_prelude(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class OrchestrationDefaults(BaseModel):
    model_config = ConfigDict(extra="ignore")

    default_epic: str | None = None
    fleet: FleetDefaults = Field(default_factory=FleetDefaults)
    harness: HarnessDefaults = Field(default_factory=HarnessDefaults)

    @field_validator("default_epic")
    @classmethod
    def _normalize_default_epic(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


def repo_config_path(ctx: RepoContext) -> Path:
    return ctx.state_dir / "config.toml"


def global_config_path() -> Path:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config_home) if xdg_config_home else Path.home() / ".config"
    return base / "redesmyn" / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            loaded = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise RuntimeError(f"Failed to read config: {path}") from e
    if not isinstance(loaded, dict):
        return {}
    return loaded


def read_config_file(path: Path) -> dict[str, Any]:
    return _read_toml(path)


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


def load_orchestration_defaults(ctx: RepoContext) -> OrchestrationDefaults:
    data: dict[str, Any] = {}
    data = _deep_merge(data, _read_toml(global_config_path()))
    data = _deep_merge(data, _read_toml(repo_config_path(ctx)))
    return OrchestrationDefaults.model_validate(data)


def _toml_string(value: str) -> str:
    if "\n" in value:
        escaped = value.replace("\\", "\\\\").replace('"""', '\\"""')
        return f'"""\n{escaped}\n"""'
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def render_config_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit_table(prefix: list[str], table: dict[str, Any]) -> None:
        scalar_keys = sorted(k for k, v in table.items() if not isinstance(v, dict))
        for key in scalar_keys:
            value = table.get(key)
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")

        nested_keys = sorted(k for k, v in table.items() if isinstance(v, dict))
        for key in nested_keys:
            value = table.get(key)
            if not isinstance(value, dict) or not value:
                continue
            if lines:
                lines.append("")
            header = ".".join([*prefix, key])
            lines.append(f"[{header}]")
            emit_table([*prefix, key], value)

    emit_table([], data)
    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def _cleanup_empty_tables(data: dict[str, Any], path_parts: list[str]) -> None:
    if not path_parts:
        return
    current: dict[str, Any] = data
    parents: list[tuple[dict[str, Any], str]] = []
    for part in path_parts:
        value = current.get(part)
        if not isinstance(value, dict):
            return
        parents.append((current, part))
        current = value
    for parent, key in reversed(parents):
        value = parent.get(key)
        if isinstance(value, dict) and not value:
            parent.pop(key, None)


def set_config_value(data: dict[str, Any], key: str, value: Any | None) -> None:
    parts = [p for p in key.split(".") if p]
    if not parts:
        raise ValueError("Config key is empty")

    current: dict[str, Any] = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value

    leaf = parts[-1]
    if value is None:
        current.pop(leaf, None)
        _cleanup_empty_tables(data, parts[:-1])
        return

    current[leaf] = value


def write_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_config_toml(data)
    path.write_text(rendered, encoding="utf-8")
