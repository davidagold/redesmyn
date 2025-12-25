from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, composite, mapped_column

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


class Base(DeclarativeBase):
    pass


# SQLite uses JSON; Postgres uses JSONB via variant.
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


def _enum_type(enum_cls: type[StrEnum], name: str) -> SAEnum:
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        values_callable=lambda obj: [member.value for member in obj],
    )


class CommandPayload(BaseModel):
    model_config = ConfigDict(extra="allow")


class EventData(BaseModel):
    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True, slots=True)
class PauseScope:
    repo: bool
    from_branch: str | None = None
    to_branch: str | None = None

    @classmethod
    def for_repo(cls) -> "PauseScope":
        return cls(repo=True, from_branch=None, to_branch=None)

    @classmethod
    def for_branch(cls, branch: str) -> "PauseScope":
        return cls(repo=False, from_branch=branch, to_branch=None)

    @classmethod
    def for_branch_range(cls, from_branch: str, to_branch: str) -> "PauseScope":
        return cls(repo=False, from_branch=from_branch, to_branch=to_branch)

    def __composite_values__(self) -> tuple[bool, str | None, str | None]:
        return (self.repo, self.from_branch, self.to_branch)

    def to_dict(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "from_branch": self.from_branch,
            "to_branch": self.to_branch,
        }

    def __str__(self) -> str:
        if self.repo:
            return "repo"
        if self.from_branch and self.to_branch:
            return f"branches:{self.from_branch}..{self.to_branch}"
        if self.from_branch:
            return f"branch:{self.from_branch}"
        return "scope:unknown"


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_root: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    default_branch: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Epic(Base):
    __tablename__ = "epics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    slug: Mapped[str] = mapped_column(String, nullable=False)
    root_branch: Mapped[str] = mapped_column(String, nullable=False)
    linear_project_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("repository_id", "slug", name="uq_epics_repo_slug"),
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    epic_id: Mapped[int] = mapped_column(
        ForeignKey("epics.id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[TaskSource] = mapped_column(
        _enum_type(TaskSource, "task_source"),
        default=TaskSource.Local,
        nullable=False,
    )
    authority: Mapped[TaskAuthority] = mapped_column(
        _enum_type(TaskAuthority, "task_authority"),
        default=TaskAuthority.Local,
        nullable=False,
    )
    state: Mapped[TaskState] = mapped_column(
        _enum_type(TaskState, "task_state"),
        default=TaskState.Todo,
        nullable=False,
    )
    node_id: Mapped[int | None] = mapped_column(
        ForeignKey("nodes.id"), nullable=True, index=True
    )
    linear_issue_id: Mapped[str | None] = mapped_column(String, nullable=True)
    github_issue_id: Mapped[str | None] = mapped_column(String, nullable=True)
    local_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    epic_id: Mapped[int] = mapped_column(
        ForeignKey("epics.id"), nullable=False, index=True
    )
    branch_name: Mapped[str] = mapped_column(String, nullable=False)
    parent_node_id: Mapped[int | None] = mapped_column(
        ForeignKey("nodes.id"), nullable=True, index=True
    )
    agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agents.id"), nullable=True, index=True
    )
    worktree_path: Mapped[str | None] = mapped_column(String, nullable=True)
    primary_task_id: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, index=True
    )
    github_pr_id: Mapped[str | None] = mapped_column(String, nullable=True)
    linear_issue_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("epic_id", "branch_name", name="uq_nodes_epic_branch"),
    )


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[AgentStatus] = mapped_column(
        _enum_type(AgentStatus, "agent_status"),
        default=AgentStatus.Idle,
        nullable=False,
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Command(Base):
    __tablename__ = "commands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    command_type: Mapped[str] = mapped_column(String, nullable=False)
    target_agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agents.id"), nullable=True, index=True
    )
    target_node_id: Mapped[int | None] = mapped_column(
        ForeignKey("nodes.id"), nullable=True, index=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,  # Pydantic: CommandPayload
    )
    state: Mapped[CommandState] = mapped_column(
        _enum_type(CommandState, "command_state"),
        default=CommandState.Queued,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Barrier(Base):
    __tablename__ = "barriers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[BarrierMode] = mapped_column(
        _enum_type(BarrierMode, "barrier_mode"),
        default=BarrierMode.Loose,
        nullable=False,
    )
    required_acks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    state: Mapped[BarrierState] = mapped_column(
        _enum_type(BarrierState, "barrier_state"),
        default=BarrierState.Open,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    fulfilled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Pause(Base):
    __tablename__ = "pauses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_repo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    scope_from_branch: Mapped[str | None] = mapped_column(String, nullable=True)
    scope_to_branch: Mapped[str | None] = mapped_column(String, nullable=True)
    scope: Mapped[PauseScope] = composite(
        PauseScope, scope_repo, scope_from_branch, scope_to_branch
    )
    mode: Mapped[PauseMode] = mapped_column(
        _enum_type(PauseMode, "pause_mode"),
        default=PauseMode.Lax,
        nullable=False,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    cleared_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cleared_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,  # Pydantic: EventData
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
