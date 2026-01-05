"""Compatibility layer for scenario seed helpers.

We keep this module as a thin re-export layer to reduce churn across
tests-v0 tasks. New code should prefer importing from `tests.scenarios.seeds.*`.
"""

from __future__ import annotations

from tests.scenarios.seeds.git import (
    SeededSpine,
    SeededSpineWithDescendant,
    SeededThreeTaskChain,
    seed_active_spine_with_descendant_worktree,
    seed_active_spine_with_mid_plan_rebase_conflict,
    seed_conflicted_merge_run,
    seed_merged_parent,
    seed_running_agent,
    seed_three_task_chain_in_progress,
)

__all__ = [
    "SeededSpine",
    "SeededSpineWithDescendant",
    "SeededThreeTaskChain",
    "seed_active_spine_with_descendant_worktree",
    "seed_active_spine_with_mid_plan_rebase_conflict",
    "seed_conflicted_merge_run",
    "seed_merged_parent",
    "seed_running_agent",
    "seed_three_task_chain_in_progress",
]
