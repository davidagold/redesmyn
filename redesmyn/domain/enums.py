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


class CommandState(StrEnum):
    Queued = "queued"
    Running = "running"
    Succeeded = "succeeded"
    Failed = "failed"
    Canceled = "canceled"


class BarrierMode(StrEnum):
    Loose = "loose"
    Tight = "tight"


class BarrierState(StrEnum):
    Open = "open"
    Fulfilled = "fulfilled"
    Expired = "expired"
    Canceled = "canceled"


class PauseMode(StrEnum):
    Lax = "lax"
    Strict = "strict"
