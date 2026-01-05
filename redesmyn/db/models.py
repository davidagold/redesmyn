from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sqlalchemy import (
    CheckConstraint,
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, composite, mapped_column

from redesmyn.domain.enums import (
    AgentStatus,
    BlockMode,
    BlockPolicy,
    CommandState,
    HarnessProfileSource,
    MergeRunStatus,
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


class GitCommitEventData(BaseModel):
    task_id: int
    branch_name: str
    sha: str
    author_name: str | None = None
    author_email: str | None = None
    authored_at: str | None = None
    subject: str | None = None
    agent_id: int | None = None


class WorktreeHealthEventData(BaseModel):
    task_id: int
    branch_name: str
    worktree_path: str
    exists: bool
    current_branch: str | None = None
    dirty: bool | None = None
    branch_mismatch: bool | None = None


class MergeRunPlanStepData(BaseModel):
    index: int
    kind: Literal["rebase", "merge_ff"]
    task_id: int | None = Field(
        default=None,
        validation_alias=AliasChoices("task_id", "node_id"),
    )
    branch_name: str
    worktree_path: str
    upstream_ref: str | None = None
    base_branch: str | None = None


class MergeRunPlanData(BaseModel):
    operation: Literal["merge", "restack"] = "merge"
    base_branch: str
    base_worktree: str
    scope: Literal["descendants", "spine"]
    restack_mode: Literal["strict", "merge_then_restack"] = "strict"
    spine_task_ids: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("spine_task_ids", "spine_node_ids"),
    )
    affected_task_ids: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("affected_task_ids", "affected_node_ids"),
    )
    steps: list[MergeRunPlanStepData] = Field(default_factory=list)


class HostCapabilities(BaseModel):
    tmux_available: bool = False
    supports_path_shim: bool = True


class HarnessProfileDefinition(BaseModel):
    argv: list[str]
    env: dict[str, str] = Field(default_factory=dict)
    working_dir: Literal["node_worktree", "task_worktree"] = "task_worktree"
    bootstrap_prelude: str | None = None
    skill_recommendation: str | None = None


class AttachNone(BaseModel):
    type: Literal["none"] = "none"


class AttachTmux(BaseModel):
    type: Literal["tmux"] = "tmux"
    session: str
    socket_path: str | None = None
    log_path: str | None = None


class AttachExternal(BaseModel):
    type: Literal["external"] = "external"
    hint: str
    log_path: str | None = None


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
    workspace_id: Mapped[str] = mapped_column(String, nullable=False, default="default")
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    repo_root: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    default_branch: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "repo_id",
            name="uq_repositories_workspace_repo_id",
        ),
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


class LinearEpicDefaults(Base):
    __tablename__ = "linear_epic_defaults"

    epic_id: Mapped[int] = mapped_column(
        ForeignKey("epics.id"), primary_key=True, index=True
    )
    team_id: Mapped[str | None] = mapped_column(String, nullable=True)
    label_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    epic_id: Mapped[int] = mapped_column(
        ForeignKey("epics.id"), nullable=False, index=True
    )
    branch_name: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_task_id: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, index=True
    )
    agent_id: Mapped[int | None] = mapped_column(
        ForeignKey("agents.id"), nullable=True, index=True
    )
    worktree_path: Mapped[str | None] = mapped_column(String, nullable=True)
    github_pr_id: Mapped[str | None] = mapped_column(String, nullable=True)
    stack_in_sync: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
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
    linear_issue_id: Mapped[str | None] = mapped_column(String, nullable=True)
    linear_identifier: Mapped[str | None] = mapped_column(String, nullable=True)
    github_issue_id: Mapped[str | None] = mapped_column(String, nullable=True)
    local_path: Mapped[str | None] = mapped_column(String, nullable=True)
    merge_ready_at: Mapped[datetime | None] = mapped_column(
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
    __table_args__ = (
        CheckConstraint(
            "merge_ready_at IS NULL OR branch_name IS NOT NULL",
            name="ck_tasks_merge_ready_requires_branch",
        ),
        UniqueConstraint("epic_id", "branch_name", name="uq_tasks_epic_branch"),
    )


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)

    # NOTE: The `agents` table predates AgentSession and still contains legacy
    # "latest run" fields (status/attach/etc). These remain in the SQLite schema
    # (and Alembic baseline) for now, so we must provide defaults when creating
    # new agents or inserts will fail under NOT NULL constraints.
    status: Mapped[AgentStatus] = mapped_column(
        _enum_type(AgentStatus, "agent_status"),
        default=AgentStatus.Stopped,
        nullable=False,
    )
    attach: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        default=lambda: {"type": "none"},
        nullable=False,  # Pydantic: AttachInfo
    )

    # Best-effort pointer to the currently running session (if any). Intentionally
    # not a foreign key to avoid circular FK constraints (SQLite).
    current_session_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AgentConfig(Base):
    __tablename__ = "agent_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[int] = mapped_column(
        ForeignKey("agents.id"), nullable=False, index=True, unique=True
    )
    harness_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("harness_profiles.id"), nullable=True, index=True
    )
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
    agent_config_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_configs.id"), nullable=True, index=True
    )
    task_id: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, index=True
    )

    status: Mapped[AgentStatus] = mapped_column(
        _enum_type(AgentStatus, "agent_session_status"),
        default=AgentStatus.Stopped,
        nullable=False,
    )

    host_id: Mapped[int | None] = mapped_column(
        ForeignKey("hosts.id"), nullable=True, index=True
    )
    harness_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("harness_profiles.id"), nullable=True, index=True
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
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    prelude_rendered: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    target_task_id: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, index=True
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


class GitRefState(Base):
    __tablename__ = "git_ref_states"

    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), primary_key=True
    )
    refs: Mapped[dict[str, str]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class GitTrunkTimeline(Base):
    __tablename__ = "git_trunk_timelines"

    epic_id: Mapped[int] = mapped_column(ForeignKey("epics.id"), primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class GitMergeBase(Base):
    __tablename__ = "git_merge_bases"

    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), primary_key=True)
    merge_base_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class MergeRun(Base):
    __tablename__ = "merge_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    epic_id: Mapped[int] = mapped_column(
        ForeignKey("epics.id"), nullable=False, index=True
    )
    requested_task_id: Mapped[int] = mapped_column(
        ForeignKey("tasks.id"), nullable=False, index=True
    )
    host_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    canonical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    status: Mapped[MergeRunStatus] = mapped_column(
        _enum_type(MergeRunStatus, "merge_run_status"),
        default=MergeRunStatus.Running,
        nullable=False,
        index=True,
    )
    scope: Mapped[str] = mapped_column(String, nullable=False, default="spine")
    allow_running: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    force: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    plan: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=lambda: MergeRunPlanData(
            base_branch="",
            base_worktree="",
            scope="spine",
        ).model_dump(mode="python"),
    )

    current_step_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blocked_step_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blocked_step_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    blocked_task_id: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True, index=True
    )
    blocked_branch_name: Mapped[str | None] = mapped_column(String, nullable=True)
    blocked_worktree_path: Mapped[str | None] = mapped_column(String, nullable=True)
    blocked_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    @property
    def restack_mode(self) -> Literal["strict", "merge_then_restack"]:
        """Merge run ordering mode (stored inside the plan snapshot)."""
        try:
            value = self.plan.get("restack_mode")
        except Exception:
            value = None
        if value == "merge_then_restack":
            return "merge_then_restack"
        return "strict"

    @property
    def operation(self) -> Literal["merge", "restack"]:
        """Git operation kind (stored inside the plan snapshot)."""
        try:
            value = self.plan.get("operation")
        except Exception:
            value = None
        if value == "restack":
            return "restack"
        return "merge"


class RepoExecutorLease(Base):
    __tablename__ = "repo_executor_leases"

    workspace_id: Mapped[str] = mapped_column(String, primary_key=True)
    repo_id: Mapped[str] = mapped_column(String, primary_key=True)
    host_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
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


class GitRefStateByInstance(Base):
    __tablename__ = "git_ref_states_by_instance"

    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id"), primary_key=True
    )
    host_key: Mapped[str] = mapped_column(String, primary_key=True)
    refs: Mapped[dict[str, str]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class GitTrunkTimelineByInstance(Base):
    __tablename__ = "git_trunk_timelines_by_instance"

    epic_id: Mapped[int] = mapped_column(ForeignKey("epics.id"), primary_key=True)
    host_key: Mapped[str] = mapped_column(String, primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,
    )
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class GitMergeBaseByInstance(Base):
    __tablename__ = "git_merge_bases_by_instance"

    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), primary_key=True)
    host_key: Mapped[str] = mapped_column(String, primary_key=True)
    merge_base_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TaskStackInSyncState(Base):
    __tablename__ = "task_stack_in_sync_states"

    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"), primary_key=True)
    host_key: Mapped[str] = mapped_column(String, primary_key=True)
    stack_in_sync: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class DaemonConnection(Base):
    __tablename__ = "daemon_connections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    host_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    capabilities: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,
    )
    attached_repos: Mapped[list[dict[str, str]]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=list,
    )
    connected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    disconnected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    disconnect_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class DaemonCommand(Base):
    __tablename__ = "daemon_commands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    host_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    command_type: Mapped[str] = mapped_column(String, nullable=False)
    workspace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    repo_id: Mapped[str | None] = mapped_column(String, nullable=True)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,
    )
    ack_data: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE,
        nullable=False,
        default=dict,
    )
    state: Mapped[CommandState] = mapped_column(
        _enum_type(CommandState, "daemon_command_state"),
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
