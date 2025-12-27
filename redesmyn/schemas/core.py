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
from redesmyn.schemas.base import ApiResponse


class RepositoryResponse(ApiResponse):
    id: int
    repo_root: str
    default_branch: str
    created_at: datetime


class EpicResponse(ApiResponse):
    id: int
    repository_id: int
    name: str
    slug: str
    root_branch: str
    linear_project_id: str | None
    created_at: datetime


class TaskResponse(ApiResponse):
    id: int
    epic_id: int
    title: str
    readme: str | None = Field(validation_alias="body")
    source: TaskSource
    authority: TaskAuthority
    state: TaskState
    node_id: int | None
    linear_issue_id: str | None
    github_issue_id: str | None
    local_path: str | None
    created_at: datetime
    updated_at: datetime


class NodeResponse(ApiResponse):
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


class AgentResponse(ApiResponse):
    id: int
    display_name: str
    status: AgentStatus
    last_seen_at: datetime | None
    created_at: datetime


class CommandResponse(ApiResponse):
    id: int
    command_type: str
    target_agent_id: int | None
    target_node_id: int | None
    payload: dict[str, Any]
    state: CommandState
    created_at: datetime
    updated_at: datetime


class BlockScopeResponse(ApiResponse):
    repo: bool
    from_branch: str | None
    to_branch: str | None


class ManualReleaseResponse(ApiResponse):
    type: Literal["manual"] = "manual"


class CommandReleaseResponse(ApiResponse):
    type: Literal["command"] = "command"
    command_id: int


class AckReleaseResponse(ApiResponse):
    type: Literal["acks"] = "acks"
    required_agent_ids: list[int]


ReleaseConditionResponse = Annotated[
    ManualReleaseResponse | CommandReleaseResponse | AckReleaseResponse,
    Field(discriminator="type"),
]


class BlockResponse(ApiResponse):
    id: int
    scope: BlockScopeResponse
    policy: BlockPolicy
    mode: BlockMode
    release: ReleaseConditionResponse
    reason: str | None
    created_at: datetime
    cleared_at: datetime | None
    cleared_reason: str | None


class EventResponse(ApiResponse):
    id: int
    event_type: str
    data: dict[str, Any]
    created_at: datetime


class BlockStatusResponse(ApiResponse):
    mode: BlockMode
    scope: BlockScopeResponse
    reason: str | None
    policy: BlockPolicy
    release: ReleaseConditionResponse


class ApiStatusResponse(ApiResponse):
    repo_root: str
    db_path: str
    default_branch: str | None
    block: BlockStatusResponse | None


class LinearStatusResponse(ApiResponse):
    connected: bool
    connected_at: datetime | None


class TrunkCommitResponse(ApiResponse):
    sha: str
    author_name: str | None = None
    author_email: str | None = None
    authored_at: datetime | None = None


class TrunkTimelineResponse(ApiResponse):
    base_sha: str | None
    base_commit: TrunkCommitResponse | None = None
    commits_before: list[TrunkCommitResponse]
    commits_after: list[TrunkCommitResponse]
    has_more_before: bool
    has_more_after: bool


class EpicGraphResponse(ApiResponse):
    epic: EpicResponse
    tasks: list[TaskResponse]
    nodes: list[NodeResponse]
    agents: list[AgentResponse]
    trunk: TrunkTimelineResponse | None = None
