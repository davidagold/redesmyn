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
    MergeRunStatus,
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
    branch_name: str | None
    parent_task_id: int | None
    stack_in_sync: bool | None = None
    agent_id: int | None
    worktree_path: str | None
    github_pr_id: str | None
    title: str
    readme: str | None = Field(validation_alias="body")
    source: TaskSource
    authority: TaskAuthority
    state: TaskState
    linear_issue_id: str | None
    github_issue_id: str | None
    local_path: str | None
    merge_ready_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AgentResponse(ApiResponse):
    id: int
    display_name: str
    current_session_id: int | None = None
    created_at: datetime


class AgentSessionResponse(ApiResponse):
    id: int
    agent_id: int
    agent_name: str
    task_id: int | None
    status: AgentStatus
    harness_profile_id: str | None = None
    resolved_profile: HarnessProfileDefinitionResponse | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


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
    prelude: str | None = None


class TaskAgentRestartRequest(ApiResponse):
    harness: str | None = None
    detach: bool = True
    prelude: str | None = None


class TaskAgentStartResponse(ApiResponse):
    task_id: int
    agent_id: int
    agent_session_id: int
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
    agent_id: int | None
    agent_name: str | None
    agent_status: AgentStatus | None
    stopped: bool


class TaskAgentBulkRunRequest(ApiRequest):
    run_id: str | None = None
    start_task_ids: list[int] = Field(default_factory=list)
    restart_task_ids: list[int] = Field(default_factory=list)
    harness: str | None = None
    detach: bool = True
    prelude: str | None = None


class TaskAgentBulkRunResponse(ApiResponse):
    run_id: str
    submitted: int


class TaskAgentBulkActionItemRequest(ApiRequest):
    task_id: int
    action: Literal["start", "restart", "stop"]


class TaskAgentBulkActionRequest(ApiRequest):
    run_id: str | None = None
    actions: list[TaskAgentBulkActionItemRequest] = Field(default_factory=list)
    harness: str | None = None
    detach: bool = True
    prelude: str | None = None


class TaskAgentBulkActionResponse(ApiResponse):
    run_id: str
    submitted: int


class TaskMergeReadyRequest(ApiRequest):
    ready: bool


class TaskMergeRequest(ApiRequest):
    run_id: str | None = None
    host_key: str | None = None
    cascade: bool = False
    scope: Literal["descendants", "spine"] = "descendants"
    restack_mode: Literal["strict", "merge_then_restack"] = "strict"
    dry_run: bool = False
    allow_running: bool = False
    force: bool = False


class TaskRestackRequest(ApiRequest):
    run_id: str | None = None
    host_key: str | None = None
    scope: Literal["descendants", "spine"] = "descendants"
    dry_run: bool = False
    allow_running: bool = False


class MergeRunSummaryResponse(ApiResponse):
    run_id: str
    epic_id: int
    requested_task_id: int
    host_key: str | None = None
    canonical: bool = True
    status: MergeRunStatus
    scope: Literal["descendants", "spine"]
    operation: Literal["merge", "restack"] = "merge"
    restack_mode: Literal["strict", "merge_then_restack"] = "strict"
    allow_running: bool
    force: bool
    current_step_index: int | None = None
    blocked_step_index: int | None = None
    blocked_step_kind: str | None = None
    blocked_task_id: int | None = None
    blocked_branch_name: str | None = None
    blocked_worktree_path: str | None = None
    blocked_error: str | None = None
    created_at: datetime
    updated_at: datetime


class TaskMergePlanStepResponse(ApiResponse):
    kind: Literal["rebase", "merge_ff"]
    task_id: int | None
    branch_name: str
    worktree_path: str
    upstream_ref: str | None = None
    base_branch: str | None = None


class TaskMergeResponse(ApiResponse):
    run_id: str
    dry_run: bool = False
    base_branch: str | None = None
    steps: list[TaskMergePlanStepResponse] = Field(default_factory=list)


class TaskRestackResponse(ApiResponse):
    run_id: str
    dry_run: bool = False
    base_branch: str | None = None
    steps: list[TaskMergePlanStepResponse] = Field(default_factory=list)


class MergeRunResumeRequest(ApiRequest):
    allow_running: bool = False
    host_key: str | None = None


class MergeRunResumeResponse(ApiResponse):
    run_id: str
    base_branch: str | None = None


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


class CommandResponse(ApiResponse):
    id: int
    command_type: str
    target_agent_id: int | None
    target_task_id: int | None
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


class GitCommitEventDataResponse(ApiResponse):
    type: Literal["git.commit"] = "git.commit"
    task_id: int
    branch_name: str
    sha: str
    author_name: str | None = None
    author_email: str | None = None
    authored_at: str | None = None
    subject: str | None = None
    agent_id: int | None = None


class WorktreeHealthEventDataResponse(ApiResponse):
    type: Literal["worktree.health"] = "worktree.health"
    task_id: int
    branch_name: str
    worktree_path: str
    exists: bool
    current_branch: str | None = None
    dirty: bool | None = None
    branch_mismatch: bool | None = None


class TaskAgentSetEventDataResponse(ApiResponse):
    type: Literal["task.agent_set"] = "task.agent_set"
    task_id: int
    agent_id: int | None
    previous_agent_id: int | None


class BlockSetEventDataResponse(ApiResponse):
    type: Literal["block.set"] = "block.set"
    scope: dict[str, Any]
    policy: str
    mode: str
    release: dict[str, Any]
    reason: str | None


class BlockClearedEventDataResponse(ApiResponse):
    type: Literal["block.cleared"] = "block.cleared"
    scope: dict[str, Any]
    policy: str
    reason: str | None


class BlockAckEventDataResponse(ApiResponse):
    type: Literal["block.ack"] = "block.ack"
    scope: dict[str, Any]
    policy: str
    agent_id: int


class TaskAgentRunEventDataResponse(ApiResponse):
    type: Literal["task.agent_run"] = "task.agent_run"
    run_id: str
    task_id: int
    action: Literal["start", "restart"]
    phase: Literal["requested", "started", "failed"]
    agent_id: int | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class TaskAgentActionEventDataResponse(ApiResponse):
    type: Literal["task.agent_action"] = "task.agent_action"
    run_id: str
    task_id: int
    action: Literal["start", "restart", "stop"]
    phase: Literal["requested", "started", "stopped", "failed"]
    agent_id: int | None = None
    stopped: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class TaskMergeEventDataResponse(ApiResponse):
    type: Literal["task.merge"] = "task.merge"
    run_id: str
    operation: Literal["merge", "restack"] = "merge"
    task_id: int | None = None
    kind: Literal["rebase", "merge_ff"]
    phase: Literal["started", "finished", "failed"]
    branch_name: str
    error: str | None = None


class MergeRunEventDataResponse(ApiResponse):
    type: Literal["merge.run"] = "merge.run"
    run_id: str
    task_id: int
    epic_id: int
    requested_task_id: int
    status: MergeRunStatus
    operation: Literal["merge", "restack"] = "merge"
    blocked_step_index: int | None = None
    blocked_step_kind: str | None = None
    blocked_branch_name: str | None = None


class UnknownEventDataResponse(ApiResponse):
    type: Literal["unknown"] = "unknown"
    event_type: str
    data: dict[str, Any]


EventDataResponse = Annotated[
    GitCommitEventDataResponse
    | WorktreeHealthEventDataResponse
    | TaskAgentSetEventDataResponse
    | BlockSetEventDataResponse
    | BlockClearedEventDataResponse
    | BlockAckEventDataResponse
    | TaskAgentRunEventDataResponse
    | TaskAgentActionEventDataResponse
    | TaskMergeEventDataResponse
    | MergeRunEventDataResponse
    | UnknownEventDataResponse,
    Field(discriminator="type"),
]


class EventResponse(ApiResponse):
    id: int
    event_type: str
    data: EventDataResponse
    created_at: datetime


class RepoKeyResponse(ApiResponse):
    workspace_id: str
    repo_id: str


class RepoExecutorStatusResponse(ApiResponse):
    workspace_id: str
    repo_id: str
    primary_host_key: str | None = None
    attached_host_keys: list[str] = Field(default_factory=list)


class DaemonPresenceResponse(ApiResponse):
    host_key: str
    display_name: str | None
    capabilities: dict[str, Any]
    attached_repos: list[RepoKeyResponse]
    connected: bool
    last_seen_at: datetime | None
    connected_at: datetime | None
    disconnected_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DaemonCommandResponse(ApiResponse):
    id: int
    host_key: str
    command_type: str
    workspace_id: str | None
    repo_id: str | None
    payload: dict[str, Any] = Field(validation_alias="data")
    ack_data: dict[str, Any] = Field(default_factory=dict)
    state: CommandState
    created_at: datetime
    updated_at: datetime


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


class SandboxCapabilitiesResponse(ApiResponse):
    provider: str
    available: bool
    supports_worktree: bool
    supports_network_deny: bool
    unavailable_reason: str | None = None


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


class OrchestrationSandboxDefaultsResponse(ApiResponse):
    type: Literal["none", "worktree"] = "none"
    network: Literal["allow", "deny"] = "allow"


class OrchestrationDefaultsResponse(ApiResponse):
    default_epic: str | None = None
    fleet: OrchestrationFleetDefaultsResponse
    harness: OrchestrationHarnessDefaultsResponse
    sandbox: OrchestrationSandboxDefaultsResponse


class OrchestrationHarnessDefaultsUpdateRequest(ApiRequest):
    command: str | None = None
    detach: bool | None = None
    prelude: str | None = None
    send_prelude: bool | None = None
    submit_prelude: bool | None = None


class OrchestrationFleetDefaultsUpdateRequest(ApiRequest):
    mode: Literal["fixed", "auto"] | None = None
    size: int | None = None


class OrchestrationSandboxDefaultsUpdateRequest(ApiRequest):
    type: Literal["none", "worktree"] | None = None
    network: Literal["allow", "deny"] | None = None


class OrchestrationDefaultsUpdateRequest(ApiRequest):
    default_epic: str | None = None
    fleet: OrchestrationFleetDefaultsUpdateRequest | None = None
    harness: OrchestrationHarnessDefaultsUpdateRequest | None = None
    sandbox: OrchestrationSandboxDefaultsUpdateRequest | None = None


class LinearStatusResponse(ApiResponse):
    connected: bool
    connected_at: datetime | None


class TrunkCommitResponse(ApiResponse):
    sha: str
    author_name: str | None = None
    author_email: str | None = None
    authored_at: datetime | None = None
    committer_name: str | None = None
    committer_email: str | None = None
    committed_at: datetime | None = None
    title: str | None = None
    message: str | None = None


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
    agent_sessions: list[AgentSessionResponse]
    merge_runs: list[MergeRunSummaryResponse] = Field(default_factory=list)
    trunk: TrunkTimelineResponse | None = None
    repo_executor: RepoExecutorStatusResponse | None = None
