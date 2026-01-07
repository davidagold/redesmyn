from __future__ import annotations

import html
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from redesmyn.domain.enums import (
    AgentKind,
    AgentKindSelection,
    AgentInterfaceMode,
    AgentStatus,
    AgentSessionRuntimeKind,
    AgentTurnState,
    BlockMode,
    BlockPolicy,
    CommandState,
    LaunchConfigurationSource,
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
    worktree_path: str | None
    github_pr_id: str | None
    title: str
    readme: str | None = Field(validation_alias="body")
    source: TaskSource
    authority: TaskAuthority
    state: TaskState
    linear_issue_id: str | None
    linear_identifier: str | None = None
    linear_state_type: str | None = None
    linear_state_name: str | None = None
    linear_state_observed_at: datetime | None = None
    github_issue_id: str | None
    local_path: str | None
    merge_ready_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AgentCapabilitiesResponse(ApiResponse):
    can_detect_ready_for_input: bool = False
    can_detect_turn_complete: bool = False
    can_send_text: bool = False
    can_interrupt: bool = False
    can_receive_notifications: bool = False
    can_resume_by_id: bool = False
    can_continue_in_cwd: bool = False
    can_stream_semantic_events: bool = False


class AgentSemanticStatusResponse(ApiResponse):
    turn_state: AgentTurnState = AgentTurnState.Unknown
    detail: str | None = None


class AgentPreviewResponse(ApiResponse):
    last_assistant_message_preview: str | None = None
    last_assistant_message_at: datetime | None = None
    last_message_turn_id: str | None = None

    @model_validator(mode="after")
    def _decode_preview_entities(self) -> "AgentPreviewResponse":
        preview = self.last_assistant_message_preview
        if not preview:
            return self

        current = preview
        for _ in range(3):
            decoded = html.unescape(current)
            if decoded == current:
                break
            current = decoded
        self.last_assistant_message_preview = current
        return self


class ExternalSessionNoneResponse(ApiResponse):
    type: Literal["none"] = "none"


class ExternalSessionCodexResponse(ApiResponse):
    type: Literal["codex_thread"] = "codex_thread"
    thread_id: str
    turn_id: str | None = None


class ExternalSessionClaudeResponse(ApiResponse):
    type: Literal["claude_session"] = "claude_session"
    session_id: str


ExternalSessionRefResponse = Annotated[
    ExternalSessionNoneResponse
    | ExternalSessionCodexResponse
    | ExternalSessionClaudeResponse,
    Field(discriminator="type"),
]


class AgentSessionResponse(ApiResponse):
    id: int
    task_id: int
    agent_label: str
    status: AgentStatus
    agent_kind_selection: AgentKindSelection = AgentKindSelection.Auto
    agent_kind: AgentKind = AgentKind.Generic
    agent_interface_mode: AgentInterfaceMode = AgentInterfaceMode.Interactive
    launch_configuration_id: str | None = None
    resolved_launch_configuration: LaunchConfigurationDefinitionResponse | None = None
    agent_capabilities: AgentCapabilitiesResponse
    agent_semantic_status: AgentSemanticStatusResponse
    external_session_ref: ExternalSessionRefResponse
    agent_preview: AgentPreviewResponse = Field(default_factory=AgentPreviewResponse)
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


class LaunchConfigurationDefinitionResponse(ApiResponse):
    argv: list[str]
    env: dict[str, str]
    working_dir: str
    bootstrap_prelude: str | None = None
    skill_recommendation: str | None = None


class LaunchConfigurationResponse(ApiResponse):
    id: str
    kind: str
    source: LaunchConfigurationSource
    display_name: str
    definition: LaunchConfigurationDefinitionResponse
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
    agent_kind: AgentKindSelection | None = None
    detach: bool = True
    prelude: str | None = None


class TaskAgentRestartRequest(ApiResponse):
    harness: str | None = None
    agent_kind: AgentKindSelection | None = None
    detach: bool = True
    prelude: str | None = None


class TaskAgentStartResponse(ApiResponse):
    task_id: int
    agent_session_id: int
    agent_label: str
    agent_status: AgentStatus
    agent_kind_selection: AgentKindSelection = AgentKindSelection.Auto
    agent_kind: AgentKind = AgentKind.Generic
    launch_configuration_id: str
    attach: AttachInfoResponse
    resolved_launch_configuration: LaunchConfigurationDefinitionResponse | None
    started_at: datetime
    started: bool = True
    warnings: list[str] = Field(default_factory=list)


class TaskAgentStopResponse(ApiResponse):
    task_id: int
    agent_label: str | None
    agent_status: AgentStatus | None
    stopped: bool


class TaskAgentBulkRunRequest(ApiRequest):
    run_id: str | None = None
    start_task_ids: list[int] = Field(default_factory=list)
    restart_task_ids: list[int] = Field(default_factory=list)
    harness: str | None = None
    agent_kind: AgentKindSelection | None = None
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
    agent_kind: AgentKindSelection | None = None
    detach: bool = True
    prelude: str | None = None


class TaskAgentBulkActionResponse(ApiResponse):
    run_id: str
    submitted: int


class TaskMergeReadyRequest(ApiRequest):
    ready: bool
    scope: Literal["task", "spine"] = "task"


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


class MergeConflictAssistStatusResponse(ApiResponse):
    active: bool = False
    state: Literal[
        "inactive",
        "waiting_for_agent_ready",
        "sent_waiting_for_turn_complete",
        "waiting_for_repo_clean",
        "ready_to_resume",
        "timed_out",
        "unsupported",
        "resumed",
    ] = "inactive"
    waiting_on: list[Literal["agent_ready", "agent_turn_complete", "repo_clean"]] = (
        Field(default_factory=list)
    )
    agent_task_id: int | None = None
    agent_session_id: int | None = None
    message_sent_at: datetime | None = None
    timeout_at: datetime | None = None
    detail: str | None = None


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
    conflict_assist: MergeConflictAssistStatusResponse | None = None
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


class MergeRunCancelRequest(ApiRequest):
    host_key: str | None = None
    abort_git: bool = False


class MergeRunCancelResponse(ApiResponse):
    run_id: str
    canceled: bool = True
    aborted_git: bool = False
    detail: str | None = None


class HostUpsertRequest(ApiResponse):
    host_key: str
    display_name: str
    capabilities: HostCapabilitiesResponse | None = None


class LaunchConfigurationUpsertRequest(ApiResponse):
    id: str
    kind: str
    display_name: str
    definition: LaunchConfigurationDefinitionResponse
    source: LaunchConfigurationSource = LaunchConfigurationSource.User


class CommandResponse(ApiResponse):
    id: int
    command_type: str
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
    required_task_ids: list[int]


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
    agent_session_id: int | None = None


class WorktreeHealthEventDataResponse(ApiResponse):
    type: Literal["worktree.health"] = "worktree.health"
    task_id: int
    branch_name: str
    worktree_path: str
    exists: bool
    current_branch: str | None = None
    dirty: bool | None = None
    branch_mismatch: bool | None = None


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
    block_id: int
    task_id: int
    scope: dict[str, Any]
    policy: str


class TaskAgentRunEventDataResponse(ApiResponse):
    type: Literal["task.agent_run"] = "task.agent_run"
    run_id: str
    task_id: int
    action: Literal["start", "restart"]
    phase: Literal["requested", "started", "failed"]
    agent_session_id: int | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class TaskAgentActionEventDataResponse(ApiResponse):
    type: Literal["task.agent_action"] = "task.agent_action"
    run_id: str
    task_id: int
    action: Literal["start", "restart", "stop"]
    phase: Literal["requested", "started", "stopped", "failed"]
    agent_session_id: int | None = None
    stopped: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class TaskAgentSessionUpdateEventDataResponse(ApiResponse):
    type: Literal["task.agent_session_update"] = "task.agent_session_update"
    task_id: int
    agent_session_id: int
    agent_status: AgentStatus
    started_at: datetime | None = None
    ended_at: datetime | None = None
    attach: AttachInfoResponse
    runtime_kind: AgentSessionRuntimeKind
    agent_kind_selection: AgentKindSelection = AgentKindSelection.Auto
    agent_kind: AgentKind = AgentKind.Generic
    agent_interface_mode: AgentInterfaceMode = AgentInterfaceMode.Interactive
    agent_capabilities: AgentCapabilitiesResponse
    agent_semantic_status: AgentSemanticStatusResponse
    external_session_ref: ExternalSessionRefResponse
    agent_preview: AgentPreviewResponse = Field(default_factory=AgentPreviewResponse)


class AgentTurnStartedEventDataResponse(ApiResponse):
    type: Literal["agent.turn_started"] = "agent.turn_started"
    task_id: int
    agent_session_id: int
    external_session_ref: ExternalSessionRefResponse


class AgentTurnCompletedEventDataResponse(ApiResponse):
    type: Literal["agent.turn_completed"] = "agent.turn_completed"
    task_id: int
    agent_session_id: int
    external_session_ref: ExternalSessionRefResponse


class AgentAssistantMessageEventDataResponse(ApiResponse):
    type: Literal["agent.assistant_message"] = "agent.assistant_message"
    task_id: int
    agent_session_id: int
    text: str
    preview: str
    external_session_ref: ExternalSessionRefResponse


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
    | BlockSetEventDataResponse
    | BlockClearedEventDataResponse
    | BlockAckEventDataResponse
    | TaskAgentRunEventDataResponse
    | TaskAgentActionEventDataResponse
    | TaskAgentSessionUpdateEventDataResponse
    | AgentTurnStartedEventDataResponse
    | AgentTurnCompletedEventDataResponse
    | AgentAssistantMessageEventDataResponse
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

    @model_validator(mode="before")
    @classmethod
    def _shape_event_data(cls, value: Any) -> Any:
        # Events are stored as (event_type, data) pairs, where `data` is a free-form
        # JSON payload. API/websocket consumers expect a discriminated `data.type`
        # to drive typed decoding; for legacy/non-schema'd events, we wrap the raw
        # payload as `unknown`.
        if not isinstance(value, dict):
            value = {
                "id": getattr(value, "id", None),
                "event_type": getattr(value, "event_type", None),
                "data": getattr(value, "data", None),
                "created_at": getattr(value, "created_at", None),
            }

        event_type = value.get("event_type")
        data = value.get("data")
        if not isinstance(data, dict):
            return value
        if "type" in data:
            return value

        event_type_str = event_type if isinstance(event_type, str) else ""
        candidate = {"type": event_type_str, **data} if event_type_str else data
        adapter = TypeAdapter(EventDataResponse)
        try:
            adapter.validate_python(candidate)
        except ValidationError:
            value["data"] = {
                "type": "unknown",
                "event_type": event_type_str or "unknown",
                "data": data,
            }
        else:
            value["data"] = candidate
        return value


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
    agent_kind: AgentKindSelection = AgentKindSelection.Auto
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
    agent_kind: AgentKindSelection | None = None
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


class LinearProjectResponse(ApiResponse):
    id: str
    name: str
    slug: str | None


class LinearLabelResponse(ApiResponse):
    id: str
    name: str


class LinearMilestoneResponse(ApiResponse):
    id: str
    name: str


class EpicLinearConfigResponse(ApiResponse):
    sync_mode: Literal["label", "milestone"]
    label_id: str | None
    label_name: str | None
    milestone_id: str | None
    milestone_name: str | None


class EpicLinearConfigUpdateRequest(ApiRequest):
    label_id: str | None = None
    label_name: str | None = None
    milestone_id: str | None = None


class EpicLinearProjectUpdateRequest(ApiRequest):
    linear_project_id: str | None = None


class SyncStatsResponse(ApiResponse):
    epics_created: int
    epics_updated: int
    tasks_created: int
    tasks_updated: int
    branches_created: int
    branches_updated: int


class LinearPushStatsResponse(ApiResponse):
    issues_created: int
    issues_updated: int
    docs_updated: int
    blockers_updated: int
    blockers_skipped: int


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
