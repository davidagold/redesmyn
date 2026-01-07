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


class AgentKind(StrEnum):
    Generic = "generic"
    Codex = "codex"
    ClaudeCode = "claude_code"


class AgentKindSelection(StrEnum):
    Auto = "auto"
    Generic = "generic"
    Codex = "codex"
    ClaudeCode = "claude_code"


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
