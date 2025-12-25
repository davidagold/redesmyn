from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class Repository:
    id: int
    repo_root: str
    default_branch: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Epic:
    id: int
    name: str
    slug: str
    root_branch: str
    linear_project_id: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Task:
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


@dataclass(frozen=True, slots=True)
class Node:
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


@dataclass(frozen=True, slots=True)
class Agent:
    id: int
    display_name: str
    status: AgentStatus
    last_seen_at: datetime | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Command:
    id: int
    command_type: str
    target_agent_id: int | None
    target_node_id: int | None
    payload: dict[str, Any]
    state: CommandState
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Barrier:
    id: int
    scope: str
    mode: BarrierMode
    required_acks: int
    state: BarrierState
    created_at: datetime
    fulfilled_at: datetime | None


@dataclass(frozen=True, slots=True)
class Pause:
    id: int
    scope: str
    mode: PauseMode
    reason: str | None
    created_at: datetime
    cleared_at: datetime | None
    cleared_reason: str | None


@dataclass(frozen=True, slots=True)
class Event:
    id: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime

