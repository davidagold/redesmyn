from __future__ import annotations

import pytest

from redesmyn.domain.enums import (
    MergeQueueActionType,
    MergeQueueConductorDecision,
    MergeQueueDependencyKind,
    MergeQueueItemState,
)
from redesmyn.schemas.core import EpicGraphResponse, MergeQueueItemResponse

from tests.scenarios.scenario import Scenario
from tests.scenarios.variants import seed_merged_parent


async def _enqueue_item(
    scenario: Scenario,
    *,
    epic_id: int,
    task_id: int,
    candidate_ref: str,
    order_index: int,
) -> MergeQueueItemResponse:
    response = await scenario.app.client.post(
        f"/v1/epics/{epic_id}/merge-queue/items",
        json={
            "task_id": task_id,
            "candidate_ref": candidate_ref,
            "state": "ready",
            "order_index": order_index,
            "authority": "director",
        },
    )
    assert response.status_code == 200, response.text
    return MergeQueueItemResponse.model_validate(response.json())


@pytest.mark.integration
async def test_merge_queue_supports_approve_pending_dependency_flow(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)

    item_b = await _enqueue_item(
        scenario,
        epic_id=seeded.epic_id,
        task_id=seeded.parent_task_id,
        candidate_ref=f"refs/heads/{seeded.parent_branch}",
        order_index=0,
    )
    item_a = await _enqueue_item(
        scenario,
        epic_id=seeded.epic_id,
        task_id=seeded.child_task_id,
        candidate_ref=f"refs/heads/{seeded.child_branch}",
        order_index=1,
    )

    add_hard_dep = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item_a.id}/dependencies",
        json={
            "depends_on_item_id": item_b.id,
            "kind": "hard",
            "authority": "director",
            "reason": "A depends on B",
        },
    )
    assert add_hard_dep.status_code == 200, add_hard_dep.text

    approve_pending = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item_a.id}/actions",
        json={
            "authority": "conductor",
            "action": "approve_pending",
            "reason": "Approve A pending B",
            "dependency_item_ids": [item_b.id],
        },
    )
    assert approve_pending.status_code == 200, approve_pending.text

    merged_b = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item_b.id}/actions",
        json={
            "authority": "director",
            "action": "mark_merged",
            "reason": "B merged",
        },
    )
    assert merged_b.status_code == 200, merged_b.text

    update_a_ref = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item_a.id}/actions",
        json={
            "authority": "director",
            "action": "candidate_updated",
            "candidate_ref": f"refs/heads/{seeded.child_branch}-updated",
            "reason": "A updated after B merge",
        },
    )
    assert update_a_ref.status_code == 200, update_a_ref.text

    make_a_mergeable = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item_a.id}/actions",
        json={
            "authority": "director",
            "action": "state_updated",
            "state": "mergeable",
        },
    )
    assert make_a_mergeable.status_code == 200, make_a_mergeable.text

    merged_a = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item_a.id}/actions",
        json={
            "authority": "director",
            "action": "mark_merged",
            "reason": "A merged",
        },
    )
    assert merged_a.status_code == 200, merged_a.text

    response = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert response.status_code == 200
    graph = EpicGraphResponse.model_validate(response.json())

    by_id = {item.id: item for item in graph.merge_queue}
    item_a_graph = by_id[item_a.id]
    item_b_graph = by_id[item_b.id]

    assert (
        item_a_graph.conductor_decision == MergeQueueConductorDecision.ApprovedPending
    )
    assert item_a_graph.approval_pending_on_item_ids == [item_b.id]
    assert item_b_graph.state == MergeQueueItemState.Merged
    assert item_a_graph.state == MergeQueueItemState.Merged
    assert any(
        action.action == MergeQueueActionType.ApprovePending
        and action.queue_item_id == item_a.id
        for action in graph.merge_queue_actions
    )


@pytest.mark.integration
async def test_merge_queue_supports_reordering_when_new_candidate_appears(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)

    item_a = await _enqueue_item(
        scenario,
        epic_id=seeded.epic_id,
        task_id=seeded.parent_task_id,
        candidate_ref=f"refs/heads/{seeded.parent_branch}",
        order_index=1,
    )
    item_b = await _enqueue_item(
        scenario,
        epic_id=seeded.epic_id,
        task_id=seeded.child_task_id,
        candidate_ref=f"refs/heads/{seeded.child_branch}",
        order_index=2,
    )
    item_c = await _enqueue_item(
        scenario,
        epic_id=seeded.epic_id,
        task_id=seeded.child_task_id,
        candidate_ref="refs/heads/task-c",
        order_index=3,
    )

    reorder_c = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item_c.id}/actions",
        json={
            "authority": "director",
            "action": "reorder",
            "order_index": 0,
            "reason": "C introduced; re-evaluate queue order",
        },
    )
    assert reorder_c.status_code == 200, reorder_c.text

    dep_b_on_c = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item_b.id}/actions",
        json={
            "authority": "director",
            "action": "dependency_added",
            "dependency_item_ids": [item_c.id],
            "dependency_kind": "hard",
            "reason": "B now depends on C",
        },
    )
    assert dep_b_on_c.status_code == 200, dep_b_on_c.text

    response = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert response.status_code == 200
    graph = EpicGraphResponse.model_validate(response.json())

    assert [item.id for item in graph.merge_queue] == [
        item_c.id,
        item_a.id,
        item_b.id,
    ]
    item_b_graph = next(item for item in graph.merge_queue if item.id == item_b.id)
    assert any(
        dep.depends_on_item_id == item_c.id
        and dep.kind == MergeQueueDependencyKind.Hard
        for dep in item_b_graph.dependencies
    )


@pytest.mark.integration
async def test_merge_queue_conductor_actions_are_durable_and_explainable(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)
    item = await _enqueue_item(
        scenario,
        epic_id=seeded.epic_id,
        task_id=seeded.child_task_id,
        candidate_ref=f"refs/heads/{seeded.child_branch}",
        order_index=0,
    )

    defer = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item.id}/actions",
        json={
            "authority": "conductor",
            "action": "defer",
            "reason": "Hold until policy update",
        },
    )
    assert defer.status_code == 200, defer.text

    requeue = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item.id}/actions",
        json={
            "authority": "conductor",
            "action": "requeue",
            "reason": "Policy updated",
        },
    )
    assert requeue.status_code == 200, requeue.text

    request_changes = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item.id}/actions",
        json={
            "authority": "conductor",
            "action": "request_changes",
            "reason": "Need follow-up wiring commit",
            "blocked_reason": "Missing follow-up wiring commit",
        },
    )
    assert request_changes.status_code == 200, request_changes.text

    response = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert response.status_code == 200
    graph = EpicGraphResponse.model_validate(response.json())

    item_graph = next(q for q in graph.merge_queue if q.id == item.id)
    assert item_graph.state == MergeQueueItemState.Blocked
    assert item_graph.conductor_decision == MergeQueueConductorDecision.ChangesRequested
    assert item_graph.blocked_reason == "Missing follow-up wiring commit"

    item_actions = [
        action
        for action in graph.merge_queue_actions
        if action.queue_item_id == item.id
    ]
    assert [(action.action, action.reason) for action in item_actions[-3:]] == [
        (MergeQueueActionType.Defer, "Hold until policy update"),
        (MergeQueueActionType.Requeue, "Policy updated"),
        (MergeQueueActionType.RequestChanges, "Need follow-up wiring commit"),
    ]


@pytest.mark.integration
async def test_merge_queue_rejects_direct_self_dependency_with_400(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)
    item = await _enqueue_item(
        scenario,
        epic_id=seeded.epic_id,
        task_id=seeded.child_task_id,
        candidate_ref=f"refs/heads/{seeded.child_branch}",
        order_index=0,
    )

    response = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item.id}/dependencies",
        json={
            "depends_on_item_id": item.id,
            "kind": "hard",
            "authority": "director",
            "reason": "invalid self dependency",
        },
    )
    assert response.status_code == 400, response.text
    detail = response.json().get("detail")
    assert isinstance(detail, str)
    assert "cannot depend on themselves" in detail


@pytest.mark.integration
async def test_merge_queue_rejects_self_dependency_via_action_helper_with_400(
    scenario: Scenario,
) -> None:
    seeded = await seed_merged_parent(scenario)
    item = await _enqueue_item(
        scenario,
        epic_id=seeded.epic_id,
        task_id=seeded.child_task_id,
        candidate_ref=f"refs/heads/{seeded.child_branch}",
        order_index=0,
    )

    response = await scenario.app.client.post(
        f"/v1/merge-queue/items/{item.id}/actions",
        json={
            "authority": "director",
            "action": "dependency_added",
            "dependency_item_ids": [item.id],
            "dependency_kind": "hard",
            "reason": "invalid self dependency",
        },
    )
    assert response.status_code == 400, response.text
    detail = response.json().get("detail")
    assert isinstance(detail, str)
    assert "cannot depend on themselves" in detail
