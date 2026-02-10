from __future__ import annotations

from enum import StrEnum


class TaskSource(StrEnum):
    Local = "local"
    Linear = "linear"
    Github = "github"


class TaskAuthority(StrEnum):
    Local = "local"
    Linear = "linear"
    Github = "github"


class TaskState(StrEnum):
    Todo = "todo"
    InProgress = "in_progress"
    Blocked = "blocked"
    Done = "done"


class AgentStatus(StrEnum):
    Stopped = "stopped"
    Running = "running"
    Blocked = "blocked"
    Error = "error"


class AgentSessionRuntimeKind(StrEnum):
    Tmux = "tmux"
    External = "external"
    None_ = "none"


class AgentTurnState(StrEnum):
    Unknown = "unknown"
    Ready = "ready"
    Busy = "busy"
    Blocked = "blocked"
    Completed = "completed"


class AgentAssistantMessageSource(StrEnum):
    Stream = "stream"
    LastMessageFile = "last_message_file"


class AgentKind(StrEnum):
    Generic = "generic"
    Codex = "codex"
    ClaudeCode = "claude_code"


class AgentKindSelection(StrEnum):
    Auto = "auto"
    Generic = "generic"
    Codex = "codex"
    ClaudeCode = "claude_code"


class AgentInterfaceMode(StrEnum):
    Interactive = "interactive"
    Structured = "structured"


class TaskAgentMessageConflictAction(StrEnum):
    Fail = "fail"
    InterruptTurn = "interrupt_turn"
    StopSessionAndStartNew = "stop_session_and_start_new"


class TaskAgentMessageConversationContinuity(StrEnum):
    Kept = "kept"
    Broken = "broken"


class LaunchConfigurationSource(StrEnum):
    Builtin = "builtin"
    User = "user"


class CommandState(StrEnum):
    Queued = "queued"
    Running = "running"
    Succeeded = "succeeded"
    Failed = "failed"
    Canceled = "canceled"


class BlockPolicy(StrEnum):
    GitMutations = "git_mutations"
    DaemonMutations = "daemon_mutations"


class BlockMode(StrEnum):
    Lax = "lax"
    Strict = "strict"


class MergeRunStatus(StrEnum):
    Running = "running"
    Blocked = "blocked"
    Resumable = "resumable"
    Succeeded = "succeeded"
    Failed = "failed"
    Canceled = "canceled"


class MergeQueueItemState(StrEnum):
    Draft = "draft"
    Ready = "ready"
    Gated = "gated"
    Mergeable = "mergeable"
    Merged = "merged"
    Blocked = "blocked"
    Deferred = "deferred"


class MergeQueueConductorDecision(StrEnum):
    Pending = "pending"
    Approved = "approved"
    ApprovedPending = "approved_pending"
    ChangesRequested = "changes_requested"
    Rejected = "rejected"
    Deferred = "deferred"


class MergeQueueDependencyKind(StrEnum):
    Hard = "hard"
    ApprovalPending = "approval_pending"
    FollowUp = "follow_up"


class MergeQueueActionAuthority(StrEnum):
    Director = "director"
    Conductor = "conductor"
    System = "system"


class MergeQueueActionType(StrEnum):
    Enqueue = "enqueue"
    CandidateUpdated = "candidate_updated"
    StateUpdated = "state_updated"
    DependencyAdded = "dependency_added"
    DependencyRemoved = "dependency_removed"
    Approve = "approve"
    ApprovePending = "approve_pending"
    RequestChanges = "request_changes"
    Reject = "reject"
    Defer = "defer"
    Requeue = "requeue"
    Pause = "pause"
    Resume = "resume"
    Reorder = "reorder"
    MarkMerged = "mark_merged"
    MarkBlocked = "mark_blocked"
