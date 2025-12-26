from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LinearRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    project_id: str | None = None
    issue_id: str | None = None
    identifier: str | None = None


class GithubRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    issue_id: str | None = None
    issue_key: str | None = None


class NodeRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    branch: str | None = None


class EpicMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slug: str
    name: str
    root_branch: str
    linear: LinearRef | None = None

    @property
    def linear_project_id(self) -> str | None:
        return self.linear.project_id if self.linear else None


class TaskMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    group_under: str | None = None
    stacked_on: str | None = None
    must_land_after: list[str] = Field(default_factory=list)
    linear: LinearRef | None = None
    github: GithubRef | None = None
    node: NodeRef | None = None

    @field_validator("id", "group_under", "stacked_on", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("must_land_after", mode="before")
    @classmethod
    def _normalize_must_land_after(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

