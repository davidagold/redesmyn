from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import pytest

from redesmyn.api import create_app
from redesmyn.context import build_repo_context
from redesmyn.host_identity import HostIdentity, host_identity_path
from redesmyn.orchestrator import init_repo
from redesmyn.settings import RedesmynSettings

from tests.scenarios.scenario import (
    Scenario,
    ScenarioApp,
    ScenarioDB,
    ScenarioDaemon,
    ScenarioRepo,
)
from tests.scenarios.variants import (
    seed_conflicted_merge_run,
    seed_merged_parent,
    seed_running_agent,
)


def _write_host_identity(*, repo_root: Path, host_key: str) -> None:
    ctx = build_repo_context(repo_root=repo_root)
    host_identity_path(ctx).write_text(
        HostIdentity(host_key=host_key, display_name="Redesmyn Tests").model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )


async def _make_scenario(
    tmp_path: Path, *, runner_mode: Literal["local", "remote"]
) -> Scenario:
    repo = ScenarioRepo.init(tmp_path)
    ctx = build_repo_context(
        repo_root=repo.repo_root, worktree_root=repo.worktrees_root
    )
    ctx.state_dir.mkdir(parents=True, exist_ok=True)
    _write_host_identity(repo_root=ctx.repo_root, host_key="test-host-key")

    await init_repo(ctx, migrate=True)

    db = await ScenarioDB.connect(db_path=ctx.db_path)
    api_app = create_app(
        settings=RedesmynSettings(
            repo_root=ctx.repo_root,
            worktree_root=ctx.worktree_root,
            db_path=ctx.db_path,
            runner_mode=runner_mode,
        )
    )
    app = await ScenarioApp.open(api_app)
    scenario = Scenario(
        ctx=ctx,
        repo=repo,
        db=db,
        app=app,
        daemon=ScenarioDaemon.from_app(app.app),
        host_key="test-host-key",
    )
    return scenario


@pytest.fixture
async def scenario(tmp_path: Path) -> AsyncIterator[Scenario]:
    scenario_ = await _make_scenario(tmp_path, runner_mode="local")
    try:
        yield scenario_
    finally:
        await scenario_.aclose()


@pytest.fixture
async def scenario_with_merged_parent(scenario: Scenario) -> Scenario:
    await seed_merged_parent(scenario)
    return scenario


@pytest.fixture
async def scenario_with_running_agent(scenario: Scenario) -> Scenario:
    await seed_running_agent(scenario)
    return scenario


@pytest.fixture
async def scenario_with_conflicted_merge_run(scenario: Scenario) -> Scenario:
    await seed_conflicted_merge_run(scenario)
    return scenario


@pytest.fixture
async def scenario_without_primary_executor(
    tmp_path: Path,
) -> AsyncIterator[Scenario]:
    scenario_ = await _make_scenario(tmp_path, runner_mode="remote")
    try:
        yield scenario_
    finally:
        await scenario_.aclose()
