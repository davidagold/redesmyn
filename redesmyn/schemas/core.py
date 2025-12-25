from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field
from redesmyn.domain.enums import (
    AgentStatus,
    BlockMode,
    BlockPolicy,
    CommandState,
    TaskAuthority,
    TaskSource,
    TaskState,
)
from redesmyn.schemas.base import ApiBaseModel


class RepositoryResponse(ApiBaseModel):
    id: int
    repo_root: str
    default_branch: str
    created_at: datetime


class EpicResponse(ApiBaseModel):
    id: int
    repository_id: int
    name: str
    slug: str
    root_branch: str
    linear_project_id: str | None
    created_at: datetime


class TaskResponse(ApiBaseModel):
    id: int
    epic_id: int
    title: str
    body: str | None
    source: TaskSource
    authority: TaskAuthority
    state: TaskState
    node_id: int | None
    linear_issue_id: str | None
    github_issue_id: str | None
    local_path: str | None
    created_at: datetime
    updated_at: datetime


class NodeResponse(ApiBaseModel):
    id: int
    epic_id: int
    branch_name: str
    parent_node_id: int | None
    agent_id: int | None
    worktree_path: str | None
    primary_task_id: int | None
    github_pr_id: str | None
    linear_issue_id: str | None
    created_at: datetime
    updated_at: datetime


class AgentResponse(ApiBaseModel):
    id: int
    display_name: str
    status: AgentStatus
    last_seen_at: datetime | None
    created_at: datetime


class CommandResponse(ApiBaseModel):
    id: int
    command_type: str
    target_agent_id: int | None
    target_node_id: int | None
    payload: dict[str, Any]
    state: CommandState
    created_at: datetime
    updated_at: datetime


class BlockScopeResponse(ApiBaseModel):
    repo: bool
    from_branch: str | None
    to_branch: str | None


class ManualReleaseResponse(ApiBaseModel):
    type: Literal["manual"] = "manual"


class CommandReleaseResponse(ApiBaseModel):
    type: Literal["command"] = "command"
    command_id: int


class AckReleaseResponse(ApiBaseModel):
    type: Literal["acks"] = "acks"
    required_agent_ids: list[int]


ReleaseConditionResponse = Annotated[
    ManualReleaseResponse | CommandReleaseResponse | AckReleaseResponse,
    Field(discriminator="type"),
]


class BlockResponse(ApiBaseModel):
    id: int
    scope: BlockScopeResponse
    policy: BlockPolicy
    mode: BlockMode
    release: ReleaseConditionResponse
    reason: str | None
    created_at: datetime
    cleared_at: datetime | None
    cleared_reason: str | None


class EventResponse(ApiBaseModel):
    id: int
    event_type: str
    data: dict[str, Any]
    created_at: datetime


class BlockStatusResponse(ApiBaseModel):
    mode: BlockMode
    scope: BlockScopeResponse
    reason: str | None
    policy: BlockPolicy
    release: ReleaseConditionResponse


class ApiStatusResponse(ApiBaseModel):
    repo_root: str
    db_path: str
    default_branch: str | None
    block: BlockStatusResponse | None
