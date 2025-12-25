from __future__ import annotations

from enum import Enum


class TaskSource(str, Enum):
    local = "local"
    linear = "linear"
    github = "github"


class TaskAuthority(str, Enum):
    local = "local"
    linear = "linear"
    github = "github"


class TaskState(str, Enum):
    todo = "todo"
    in_progress = "in_progress"
    blocked = "blocked"
    done = "done"


class AgentStatus(str, Enum):
    idle = "idle"
    running = "running"
    blocked = "blocked"
    error = "error"


class CommandState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


class BarrierMode(str, Enum):
    loose = "loose"
    tight = "tight"


class BarrierState(str, Enum):
    open = "open"
    fulfilled = "fulfilled"
    expired = "expired"
    canceled = "canceled"


class PauseMode(str, Enum):
    lax = "lax"
    strict = "strict"

