from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, composite, mapped_column

from redesmyn.domain.enums import (
    AgentStatus,
    AgentSessionStatus,
    BlockMode,
    BlockPolicy,
    CommandState,
    HarnessProfileSource,
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


class CommandData(BaseModel):
    model_config = ConfigDict(extra="allow")


class EventData(BaseModel):
    model_config = ConfigDict(extra="allow")


class HostCapabilities(BaseModel):
    tmux_available: bool = False
    supports_path_shim: bool = True


class HarnessProfileDefinition(BaseModel):
    argv: list[str]
    env: dict[str, str] = Field(default_factory=dict)
    working_dir: Literal["node_worktree"] = "node_worktree"
    bootstrap_prelude: str | None = None
    skill_recommendation: str | None = None


class AttachNone(BaseModel):
    type: Literal["none"] = "none"


class AttachTmux(BaseModel):
    type: Literal["tmux"] = "tmux"
    session: str
    socket_path: str | None = None


class AttachExternal(BaseModel):
    type: Literal["external"] = "external"
    hint: str


AttachInfo = Annotated[
    AttachNone | AttachTmux | AttachExternal, Field(discriminator="type")
]


class ManualRelease(BaseModel):
    type: Literal["manual"] = "manual"


class CommandRelease(BaseModel):
    type: Literal["command"] = "command"
    command_id: int


class AckRelease(BaseModel):
    type: Literal["acks"] = "acks"
    required_agent_ids: list[int]


ReleaseCondition = Annotated[
    ManualRelease | CommandRelease | AckRelease,
    Field(discriminator="type"),
]


@dataclass(frozen=True, slots=True)
class BlockScope:
    repo: bool
    from_branch: str | None = None
    to_branch: str | None = None

    @classmethod
    def for_repo(cls) -> "BlockScope":
        return cls(repo=True, from_branch=None, to_branch=None)

    @classmethod
    def for_branch(cls, branch: str) -> "BlockScope":
        return cls(repo=False, from_branch=branch, to_branch=None)

    @classmethod
    def for_branch_range(cls, from_branch: str, to_branch: str) -> "BlockScope":
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


class Host(Base):
    __tablename__ = "hosts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    host_key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    capabilities: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=lambda: HostCapabilities().model_dump(mode="python"),
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
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


class HarnessProfile(Base):
    __tablename__ = "harness_profiles"

    # String primary key so built-ins and user-defined profiles can exist without schema changes.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[HarnessProfileSource] = mapped_column(
        _enum_type(HarnessProfileSource, "harness_profile_source"),
        default=HarnessProfileSource.Builtin,
        nullable=False,
    )
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,  # Pydantic: HarnessProfileDefinition
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


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id"), nullable=False, index=True
    )
    node_id: Mapped[int | None] = mapped_column(
        ForeignKey("nodes.id"), nullable=True, index=True
    )
    host_id: Mapped[int] = mapped_column(
        ForeignKey("hosts.id"), nullable=False, index=True
    )
    harness_profile_id: Mapped[str] = mapped_column(
        ForeignKey("harness_profiles.id"), nullable=False, index=True
    )
    status: Mapped[AgentSessionStatus] = mapped_column(
        _enum_type(AgentSessionStatus, "agent_session_status"),
        default=AgentSessionStatus.Starting,
        nullable=False,
    )

    cwd_path: Mapped[str | None] = mapped_column(String, nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    attach: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=lambda: AttachNone().model_dump(mode="python"),
    )
    resolved_profile: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_TYPE,
        nullable=True,  # Pydantic: HarnessProfileDefinition
    )
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

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
        # v0 invariants:
        # - at most one active session per node (where ended_at is NULL)
        # - at most one active session per agent (where ended_at is NULL)
        Index(
            "uq_agent_sessions_active_agent",
            "agent_id",
            unique=True,
            sqlite_where=text("ended_at IS NULL"),
            postgresql_where=text("ended_at IS NULL"),
        ),
        Index(
            "uq_agent_sessions_active_node",
            "node_id",
            unique=True,
            sqlite_where=text("ended_at IS NULL AND node_id IS NOT NULL"),
            postgresql_where=text("ended_at IS NULL AND node_id IS NOT NULL"),
        ),
    )


class LinearAuth(Base):
    __tablename__ = "linear_auth"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    access_token: Mapped[str] = mapped_column(String, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(String, nullable=True)
    token_type: Mapped[str] = mapped_column(String, nullable=False, default="Bearer")
    scope: Mapped[str | None] = mapped_column(String, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
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
    data: Mapped[dict[str, Any]] = mapped_column(
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


class Block(Base):
    __tablename__ = "blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    scope_repo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    scope_from_branch: Mapped[str | None] = mapped_column(String, nullable=True)
    scope_to_branch: Mapped[str | None] = mapped_column(String, nullable=True)
    scope: Mapped[BlockScope] = composite(
        BlockScope, scope_repo, scope_from_branch, scope_to_branch
    )

    policy: Mapped[BlockPolicy] = mapped_column(
        _enum_type(BlockPolicy, "block_policy"),
        default=BlockPolicy.GitMutations,
        nullable=False,
    )
    mode: Mapped[BlockMode] = mapped_column(
        _enum_type(BlockMode, "block_mode"),
        default=BlockMode.Lax,
        nullable=False,
    )
    release: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=lambda: {"type": "manual"},  # Pydantic: ReleaseCondition
    )

    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    cleared_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cleared_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class BlockAck(Base):
    __tablename__ = "block_acks"

    block_id: Mapped[int] = mapped_column(ForeignKey("blocks.id"), primary_key=True)
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), primary_key=True)
    acked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


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
