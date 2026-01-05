from __future__ import annotations

from collections.abc import Mapping, Sequence

from redesmyn.db.models import Task
from redesmyn.domain.enums import TaskState


def resolve_spine_task_ids(
    *, tasks_by_id: Mapping[int, Task], leaf_task_id: int
) -> tuple[list[int], list[str]]:
    """
    Resolve the parent_task_id chain for `leaf_task_id`, returning root→leaf ids.

    If the chain cannot be fully resolved (cycle or missing parent), we fail
    safely by returning `[leaf_task_id]` plus a warning message.
    """

    warnings: list[str] = []
    visited: set[int] = set()

    spine_leaf_to_root: list[int] = [leaf_task_id]
    visited.add(leaf_task_id)

    cursor: int = leaf_task_id
    while True:
        task = tasks_by_id.get(cursor)
        if task is None:
            warnings.append(f"Could not resolve merge spine: task {cursor} missing.")
            return [leaf_task_id], warnings

        parent_id = task.parent_task_id
        if parent_id is None:
            break

        if parent_id in visited:
            warnings.append("Could not resolve merge spine: detected a cycle.")
            return [leaf_task_id], warnings

        if parent_id not in tasks_by_id:
            warnings.append(
                f"Could not resolve merge spine: missing parent task {parent_id}."
            )
            return [leaf_task_id], warnings

        spine_leaf_to_root.append(parent_id)
        visited.add(parent_id)
        cursor = parent_id

    spine_leaf_to_root.reverse()
    return spine_leaf_to_root, warnings


def split_merged_spine_prefix(
    *, tasks_by_id: Mapping[int, Task], spine_task_ids: Sequence[int]
) -> tuple[list[int], list[int]]:
    """
    Split a root→leaf spine into (merged_prefix, active_suffix).

    "Merged" is determined by TaskState.Done, matching the v0 merge semantics.
    """

    merged: list[int] = []
    for task_id in spine_task_ids:
        task = tasks_by_id.get(task_id)
        if task is None or task.state != TaskState.Done:
            break
        merged.append(task_id)
    active = list(spine_task_ids[len(merged) :])
    return merged, active
