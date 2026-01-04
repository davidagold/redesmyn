from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from redesmyn.context import build_repo_context
from redesmyn.host_identity import HostIdentity, host_identity_path
from redesmyn.orchestrator import init_repo

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


def _configure_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repo_root: Path,
    worktree_root: Path,
    db_path: Path,
    runner_mode: str,
) -> None:
    monkeypatch.setenv("REDESMYN_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("REDESMYN_WORKTREE_ROOT", str(worktree_root))
    monkeypatch.setenv("REDESMYN_DB_PATH", str(db_path))
    monkeypatch.setenv("REDESMYN_RUNNER_MODE", runner_mode)
    monkeypatch.delenv("REDESMYN_DB_URL", raising=False)


async def _make_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, runner_mode: str
) -> Scenario:
    repo = ScenarioRepo.init(tmp_path)
    ctx = build_repo_context(
        repo_root=repo.repo_root, worktree_root=repo.worktrees_root
    )
    ctx.state_dir.mkdir(parents=True, exist_ok=True)
    _write_host_identity(repo_root=ctx.repo_root, host_key="test-host-key")

    await init_repo(ctx, migrate=True)

    _configure_env(
        monkeypatch,
        repo_root=ctx.repo_root,
        worktree_root=ctx.worktree_root,
        db_path=ctx.db_path,
        runner_mode=runner_mode,
    )

    db = await ScenarioDB.connect(db_path=ctx.db_path)
    app = await ScenarioApp.open()
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
async def scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Scenario]:
    scenario_ = await _make_scenario(tmp_path, monkeypatch, runner_mode="local")
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Scenario]:
    scenario_ = await _make_scenario(tmp_path, monkeypatch, runner_mode="remote")
    try:
        yield scenario_
    finally:
        await scenario_.aclose()
