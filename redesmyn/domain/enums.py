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
    Idle = "idle"
    Running = "running"
    Blocked = "blocked"
    Error = "error"


class AgentSessionStatus(StrEnum):
    Starting = "starting"
    Running = "running"
    Stopping = "stopping"
    Stopped = "stopped"
    Failed = "failed"


class HarnessProfileSource(StrEnum):
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
