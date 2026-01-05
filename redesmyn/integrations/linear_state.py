from __future__ import annotations

from redesmyn.domain.enums import TaskState


def linear_state_type_from_task_state(state: TaskState) -> str:
    match state:
        case TaskState.Todo:
            return "unstarted"
        case TaskState.InProgress:
            return "started"
        case TaskState.Blocked:
            return "blocked"
        case TaskState.Done:
            return "completed"
    return "unstarted"


def task_state_from_linear_state_type(state_type: str | None) -> TaskState:
    if not state_type:
        return TaskState.Todo
    normalized = state_type.strip().lower()
    if normalized in {"unstarted", "todo"}:
        return TaskState.Todo
    if normalized in {"started", "in_progress"}:
        return TaskState.InProgress
    if normalized == "blocked":
        return TaskState.Blocked
    if normalized in {"completed", "done"}:
        return TaskState.Done
    return TaskState.Todo
