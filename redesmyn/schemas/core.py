from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field

from redesmyn.domain.enums import (
    AgentStatus,
    BlockMode,
    BlockPolicy,
    CommandState,
    HarnessProfileSource,
    TaskAuthority,
    TaskSource,
    TaskState,
)
from redesmyn.schemas.base import ApiRequest, ApiResponse


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
    harness_profile_id: str | None = None
    resolved_profile: HarnessProfileDefinitionResponse | None = None
    status: AgentStatus
    last_seen_at: datetime | None
    created_at: datetime


class HostCapabilitiesResponse(ApiResponse):
    tmux_available: bool = False
    supports_path_shim: bool = True


class HostResponse(ApiResponse):
    id: int
    host_key: str
    display_name: str
    capabilities: HostCapabilitiesResponse
    last_seen_at: datetime | None
    created_at: datetime
    updated_at: datetime


class HarnessProfileDefinitionResponse(ApiResponse):
    argv: list[str]
    env: dict[str, str]
    working_dir: str
    bootstrap_prelude: str | None = None
    skill_recommendation: str | None = None


class HarnessProfileResponse(ApiResponse):
    id: str
    kind: str
    source: HarnessProfileSource
    display_name: str
    definition: HarnessProfileDefinitionResponse
    created_at: datetime
    updated_at: datetime


class AttachNoneResponse(ApiResponse):
    type: Literal["none"] = "none"


class AttachTmuxResponse(ApiResponse):
    type: Literal["tmux"] = "tmux"
    session: str
    socket_path: str | None = None
    log_path: str | None = None


class AttachExternalResponse(ApiResponse):
    type: Literal["external"] = "external"
    hint: str
    log_path: str | None = None


AttachInfoResponse = Annotated[
    AttachNoneResponse | AttachTmuxResponse | AttachExternalResponse,
    Field(discriminator="type"),
]


class TaskAgentStartRequest(ApiResponse):
    harness: str
    detach: bool = True


class TaskAgentRestartRequest(ApiResponse):
    harness: str | None = None
    detach: bool = True


class TaskAgentStartResponse(ApiResponse):
    task_id: int
    node_id: int
    agent_id: int
    agent_name: str
    agent_status: AgentStatus
    harness_profile_id: str
    attach: AttachInfoResponse
    resolved_profile: HarnessProfileDefinitionResponse | None
    started_at: datetime
    started: bool = True
    warnings: list[str] = Field(default_factory=list)


class TaskAgentStopResponse(ApiResponse):
    task_id: int
    node_id: int | None
    agent_id: int | None
    agent_name: str | None
    agent_status: AgentStatus | None
    stopped: bool


class HostUpsertRequest(ApiResponse):
    host_key: str
    display_name: str
    capabilities: HostCapabilitiesResponse | None = None


class HarnessProfileUpsertRequest(ApiResponse):
    id: str
    kind: str
    display_name: str
    definition: HarnessProfileDefinitionResponse
    source: HarnessProfileSource = HarnessProfileSource.User


class NodeSetAgentRequest(ApiResponse):
    agent_id: int | None


class NodeStartSessionRequest(ApiResponse):
    """Request to start a runner-owned session for a node."""

    command: str
    detach: bool = True


class NodeRestartSessionRequest(ApiResponse):
    """Request to restart a runner-owned session for a node."""

    command: str | None = None
    detach: bool = True


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


class OrchestrationFleetDefaultsResponse(ApiResponse):
    mode: Literal["fixed", "auto"]
    size: int | None = None


class OrchestrationHarnessDefaultsResponse(ApiResponse):
    command: str | None = None
    detach: bool = True
    prelude: str | None = None
    built_in_prelude_template: str
    send_prelude: bool = True
    submit_prelude: bool = True


class OrchestrationDefaultsResponse(ApiResponse):
    default_epic: str | None = None
    fleet: OrchestrationFleetDefaultsResponse
    harness: OrchestrationHarnessDefaultsResponse


class OrchestrationHarnessDefaultsUpdateRequest(ApiRequest):
    command: str | None = None
    detach: bool | None = None
    prelude: str | None = None
    send_prelude: bool | None = None
    submit_prelude: bool | None = None


class OrchestrationFleetDefaultsUpdateRequest(ApiRequest):
    mode: Literal["fixed", "auto"] | None = None
    size: int | None = None


class OrchestrationDefaultsUpdateRequest(ApiRequest):
    default_epic: str | None = None
    fleet: OrchestrationFleetDefaultsUpdateRequest | None = None
    harness: OrchestrationHarnessDefaultsUpdateRequest | None = None


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
