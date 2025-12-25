from __future__ import annotations

from datetime import datetime
from typing import Any

from redesmyn.domain.enums import (
    AgentStatus,
    BarrierMode,
    BarrierState,
    CommandState,
    PauseMode,
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


class BarrierResponse(ApiBaseModel):
    id: int
    scope: str
    mode: BarrierMode
    required_acks: int
    state: BarrierState
    created_at: datetime
    fulfilled_at: datetime | None


class PauseResponse(ApiBaseModel):
    id: int
    scope: str
    mode: PauseMode
    reason: str | None
    created_at: datetime
    cleared_at: datetime | None
    cleared_reason: str | None


class EventResponse(ApiBaseModel):
    id: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


class PauseStatusResponse(ApiBaseModel):
    mode: PauseMode
    scope: str
    reason: str | None


class ApiStatusResponse(ApiBaseModel):
    repo_root: str
    db_path: str
    default_branch: str | None
    pause: PauseStatusResponse | None

