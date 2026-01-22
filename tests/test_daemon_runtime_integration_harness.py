from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import select

from redesmyn.db import DaemonCommand, MergeRun
from redesmyn.domain.enums import CommandState, MergeRunStatus

from tests.scenarios.daemon_harness import DaemonRuntimeHarness
from tests.scenarios.scenario import Scenario
from tests.scenarios.seeds.git import seed_merged_parent


@pytest.mark.integration
async def test_daemon_runtime_attach_and_detach_updates_repo_executor_status(
    scenario_without_primary_executor: Scenario,
) -> None:
    scenario = scenario_without_primary_executor
    seeded = await seed_merged_parent(scenario)

    before = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert before.status_code == 200
    assert before.json()["repoExecutor"]["attachedHostKeys"] == []

    harness = await DaemonRuntimeHarness.open(scenario, host_key="daemon-a")
    try:
        after = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
        assert after.status_code == 200
        assert after.json()["repoExecutor"]["attachedHostKeys"] == ["daemon-a"]
    finally:
        await harness.aclose()

    detached = await scenario.app.client.get(f"/v1/epics/{seeded.epic_id}/graph")
    assert detached.status_code == 200
    assert detached.json()["repoExecutor"]["attachedHostKeys"] == []


@pytest.mark.integration
async def test_daemon_runtime_merge_plan_command_returns_plan_snapshot(
    scenario_without_primary_executor: Scenario,
) -> None:
    scenario = scenario_without_primary_executor
    seeded = await seed_merged_parent(scenario)

    harness = await DaemonRuntimeHarness.open(scenario, host_key="daemon-a")
    try:
        command_id = await harness.create_and_send_command(
            command_type="repo.merge_run.plan",
            workspace_id=harness.daemon.attached_repos[0].workspace_id,
            repo_id=harness.daemon.attached_repos[0].repo_id,
            payload={
                "run_id": uuid4().hex,
                "task_id": seeded.child_task_id,
                "operation": "merge",
                "scope": "spine",
                "restack_mode": "strict",
                "force": False,
            },
        )

        await harness.wait_for_command_state(command_id=command_id, state=CommandState.Running)
        await harness.wait_for_server_message(expected_type="command_ack_ok")
        done = await harness.wait_for_command_state(
            command_id=command_id, state=CommandState.Succeeded
        )
        await harness.wait_for_server_message(expected_type="command_ack_ok")
        assert "plan_snapshot" in done.get("data", {})

        async with scenario.db.session() as session:
            row = await session.scalar(
                select(DaemonCommand).where(DaemonCommand.id == command_id)
            )
            assert row is not None
            assert row.state == CommandState.Succeeded
            assert isinstance(row.ack_data, dict)
            assert "plan_snapshot" in row.ack_data
            assert row.ack_data["plan_snapshot"]["base_branch"] == "main"
    finally:
        await harness.aclose()


@pytest.mark.integration
async def test_daemon_runtime_telemetry_tick_emits_git_commit_event(
    scenario_without_primary_executor: Scenario,
) -> None:
    scenario = scenario_without_primary_executor
    seeded = await seed_merged_parent(scenario)

    harness = await DaemonRuntimeHarness.open(scenario, host_key="daemon-a")
    try:
        await harness.telemetry_tick()

        sha = scenario.repo.commit_file(
            worktree_path=seeded.child_worktree,
            relpath="child-2.txt",
            content="child-2\n",
            message="child change 2",
        )
        await harness.telemetry_tick()

        msg = await harness.wait_for_event(
            event_type="git.commit",
            predicate=lambda m: m.get("data", {}).get("sha") == sha,
        )
        assert msg["workspace_id"] == harness.daemon.attached_repos[0].workspace_id
        assert msg["repo_id"] == harness.daemon.attached_repos[0].repo_id
        assert msg["data"]["task_id"] == seeded.child_task_id
        assert msg["data"]["branch_name"] == seeded.child_branch
        assert msg["data"]["sha"] == sha
    finally:
        await harness.aclose()


@pytest.mark.integration
async def test_daemon_runtime_merge_run_start_executes_happy_path(
    scenario_without_primary_executor: Scenario,
) -> None:
    scenario = scenario_without_primary_executor
    seeded = await seed_merged_parent(scenario)

    main_before = scenario.repo.git(["rev-parse", "main"], cwd=scenario.ctx.repo_root)
    child_head = scenario.repo.git(
        ["rev-parse", seeded.child_branch], cwd=scenario.ctx.repo_root
    )
    assert main_before != child_head

    run_id = uuid4().hex
    async with scenario.db.session() as session:
        session.add(
            MergeRun(
                run_id=run_id,
                epic_id=seeded.epic_id,
                requested_task_id=seeded.child_task_id,
                canonical=True,
                status=MergeRunStatus.Running,
                scope="spine",
            )
        )
        await session.commit()

    harness = await DaemonRuntimeHarness.open(scenario, host_key="daemon-a")
    try:
        command_id = await harness.create_and_send_command(
            command_type="repo.merge_run.start",
            workspace_id=harness.daemon.attached_repos[0].workspace_id,
            repo_id=harness.daemon.attached_repos[0].repo_id,
            payload={
                "run_id": run_id,
                "task_id": seeded.child_task_id,
                "operation": "merge",
                "scope": "spine",
                "restack_mode": "strict",
                "allow_running": False,
                "force": False,
                "canonical": True,
            },
        )

        await harness.wait_for_command_state(command_id=command_id, state=CommandState.Running)
        await harness.wait_for_server_message(expected_type="command_ack_ok")
        await harness.wait_for_command_state(
            command_id=command_id, state=CommandState.Succeeded, timeout_s=10.0
        )
        await harness.wait_for_server_message(expected_type="command_ack_ok", timeout_s=10.0)

        succeeded = await harness.wait_for_event(
            event_type="merge.run",
            predicate=lambda m: m.get("data", {}).get("status") == MergeRunStatus.Succeeded,
            timeout_s=10.0,
        )
        assert succeeded["data"]["run_id"] == run_id
        assert succeeded["data"]["status"] == MergeRunStatus.Succeeded

        main_after = scenario.repo.git(["rev-parse", "main"], cwd=scenario.ctx.repo_root)
        assert main_after == child_head

        # Wait for the control plane to ingest the final merge.run event.
        end = asyncio.get_running_loop().time() + 5.0
        while True:
            async with scenario.db.session() as session:
                row = await session.scalar(
                    select(MergeRun).where(MergeRun.run_id == run_id)
                )
                if row is not None and row.status == MergeRunStatus.Succeeded:
                    assert row.host_key == "daemon-a"
                    break
            if asyncio.get_running_loop().time() >= end:
                raise AssertionError("Timed out waiting for merge run to succeed")
            await asyncio.sleep(0)
    finally:
        await harness.aclose()


@pytest.mark.integration
async def test_lease_enforcement_rejects_canonical_resume_on_non_primary_host(
    scenario_without_primary_executor: Scenario,
) -> None:
    scenario = scenario_without_primary_executor
    seeded = await seed_merged_parent(scenario)

    run_id = uuid4().hex
    async with scenario.db.session() as session:
        session.add(
            MergeRun(
                run_id=run_id,
                epic_id=seeded.epic_id,
                requested_task_id=seeded.child_task_id,
                canonical=True,
                status=MergeRunStatus.Resumable,
                scope="spine",
                host_key="daemon-a",
            )
        )
        await session.commit()

    harness = await DaemonRuntimeHarness.open(scenario, host_key="daemon-a")
    try:
        resp = await scenario.app.client.post(
            f"/v1/merge-runs/{run_id}/resume",
            json={"allow_running": False, "host_key": "daemon-b"},
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "Primary executor is" in detail
        assert "daemon-b" in detail
    finally:
        await harness.aclose()
