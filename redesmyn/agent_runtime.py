from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.context import RepoContext
from redesmyn.db import (
    Agent,
    Epic,
    HarnessProfile,
    Host,
    Node,
    Task,
    create_engine,
    create_sessionmaker,
)
from redesmyn.db.models import (
    AttachExternal,
    AttachInfo,
    AttachTmux,
    HarnessProfileDefinition,
    HostCapabilities,
)
from redesmyn.domain.enums import AgentStatus, HarnessProfileSource
from redesmyn.repo import GitCommandError, current_branch, git_worktree_add
from redesmyn.strings import slugify


class RunnerHostConfig(BaseModel):
    host_key: str
    display_name: str


def runner_host_config_path(ctx: RepoContext) -> Path:
    return ctx.state_dir / "runner-host.json"


def load_or_create_runner_host_config(ctx: RepoContext) -> RunnerHostConfig:
    path = runner_host_config_path(ctx)
    if path.exists():
        return RunnerHostConfig.model_validate_json(path.read_text(encoding="utf-8"))

    config = RunnerHostConfig(host_key=str(uuid4()), display_name=platform.node())
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return config


def has_tmux() -> bool:
    return shutil.which("tmux") is not None


@dataclass(frozen=True, slots=True)
class GitShim:
    dir: Path

    @property
    def git_path(self) -> Path:
        return self.dir / "git"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def create_git_shim(*, ctx: RepoContext, agent_id: int) -> GitShim:
    shim_dir = ctx.state_dir / "shims" / f"agent-{agent_id}"
    shim_dir.mkdir(parents=True, exist_ok=True)
    # Best-effort enforcement: route `git ...` through `rn git ...` so blocks apply.
    #
    # IMPORTANT: `rn git` itself spawns `git`. If we leave the shim on PATH, it would recurse.
    # So the shim removes itself from PATH before delegating.
    _write_executable(
        shim_dir / "git",
        "#!/bin/sh\n"
        'SHIM_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        "NEW_PATH=\n"
        "OLD_IFS=$IFS\n"
        "IFS=:\n"
        "for p in $PATH; do\n"
        '  if [ "$p" != "$SHIM_DIR" ]; then\n'
        '    if [ -z "$NEW_PATH" ]; then\n'
        "      NEW_PATH=$p\n"
        "    else\n"
        "      NEW_PATH=$NEW_PATH:$p\n"
        "    fi\n"
        "  fi\n"
        "done\n"
        "IFS=$OLD_IFS\n"
        "export PATH=$NEW_PATH\n"
        'exec rn git "$@"\n',
    )
    return GitShim(dir=shim_dir)


def agent_dir(ctx: RepoContext, *, agent_id: int) -> Path:
    return ctx.state_dir / "agents" / str(agent_id)


def agent_log_path(ctx: RepoContext, *, agent_id: int) -> Path:
    return agent_dir(ctx, agent_id=agent_id) / "output.log"


def write_agent_launcher(
    *,
    ctx: RepoContext,
    agent_id: int,
    argv: list[str],
    env: dict[str, str],
) -> Path:
    run_dir = agent_dir(ctx, agent_id=agent_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    lines = ["#!/bin/sh", "set -eu"]
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.append(f"exec {shlex.join(argv)}")
    script_path = run_dir / "run.sh"
    _write_executable(script_path, "\n".join(lines) + "\n")
    return script_path


def _tmux_new_session(*, name: str, cwd: Path, script_path: Path) -> None:
    proc = subprocess.run(
        ["tmux", "new-session", "-d", "-s", name, "-c", str(cwd), str(script_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "tmux new-session failed")


def _tmux_pipe_to_log(*, name: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"cat >> {shlex.quote(str(log_path))}"
    proc = subprocess.run(
        ["tmux", "pipe-pane", "-t", name, "-o", cmd],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "tmux pipe-pane failed")


def _tmux_kill_session(*, name: str) -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _tmux_has_session(*, name: str) -> bool:
    proc = subprocess.run(
        ["tmux", "has-session", "-t", name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=1,
    )
    return proc.returncode == 0


def _tmux_attach(*, name: str) -> int:
    proc = subprocess.run(["tmux", "attach-session", "-t", name])
    return proc.returncode


async def ensure_host_row(session: AsyncSession, ctx: RepoContext) -> Host:
    config = load_or_create_runner_host_config(ctx)
    capabilities = HostCapabilities(
        tmux_available=has_tmux(),
        supports_path_shim=shutil.which("rn") is not None,
    ).model_dump(mode="python")

    host = await session.scalar(select(Host).where(Host.host_key == config.host_key))
    if host is None:
        host = Host(
            host_key=config.host_key,
            display_name=config.display_name,
            capabilities=capabilities,
            last_seen_at=datetime.now(UTC),
        )
        session.add(host)
        await session.flush()
        return host

    host.display_name = config.display_name
    host.capabilities = capabilities
    host.last_seen_at = datetime.now(UTC)
    return host


def tmux_session_name_for_task(*, task_id: int) -> str:
    return f"rn-a-{task_id}"


def _parse_harness_command(command: str) -> list[str]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("Harness command is empty")
    return argv


async def ensure_harness_profile_row(
    session: AsyncSession,
    *,
    profile_id: str,
    kind: str,
    definition: HarnessProfileDefinition,
) -> HarnessProfile:
    profile = await session.get(HarnessProfile, profile_id)
    if profile is not None:
        return profile

    profile = HarnessProfile(
        id=profile_id,
        kind=kind,
        source=HarnessProfileSource.Builtin,
        display_name=kind,
        definition=definition.model_dump(mode="python"),
    )
    session.add(profile)
    await session.flush()
    return profile


def default_worktree_path(ctx: RepoContext, *, branch: str) -> Path:
    safe_parts: list[str] = []
    for part in branch.split("/"):
        if part in {"", ".", ".."}:
            continue
        safe_parts.append(part.replace(":", "_"))

    safe_rel = Path(*safe_parts) if safe_parts else Path(branch.replace(":", "_"))
    return ctx.state_dir / "worktrees" / safe_rel


def _find_existing_worktree_path_for_branch(
    repo_root: Path, *, branch: str
) -> Path | None:
    proc = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    if proc.returncode != 0:
        return None

    current_path: Path | None = None
    current_branch: str | None = None

    def commit_current() -> Path | None:
        if current_path is None or current_branch is None:
            return None
        if current_branch == branch:
            return current_path
        return None

    for line in proc.stdout.splitlines():
        if not line.strip():
            match = commit_current()
            if match is not None:
                return match
            current_path = None
            current_branch = None
            continue

        if line.startswith("worktree "):
            current_path = Path(line.removeprefix("worktree ").strip())
            continue

        if line.startswith("branch "):
            ref = line.removeprefix("branch ").strip()
            if ref.startswith("refs/heads/"):
                current_branch = ref.removeprefix("refs/heads/")
            else:
                current_branch = ref
            continue

    return commit_current()


def harness_profile_id_for_definition(
    kind: str, definition: HarnessProfileDefinition
) -> str:
    normalized = json.dumps(
        definition.model_dump(mode="python"), sort_keys=True
    ).encode("utf-8")
    digest = sha256(normalized).hexdigest()[:12]
    return f"{slugify(kind)}/sha256-{digest}"


async def ensure_node_worktree(
    session: AsyncSession,
    ctx: RepoContext,
    *,
    node: Node,
    epic: Epic,
) -> Path:
    if node.worktree_path:
        path = Path(node.worktree_path)
        if path.exists():
            return path
        node.worktree_path = None
        await session.flush()

    parent_node: Node | None = None
    if node.parent_node_id is not None:
        parent_node = await session.get(Node, node.parent_node_id)

    base_ref = parent_node.branch_name if parent_node is not None else epic.root_branch

    existing_path = _find_existing_worktree_path_for_branch(
        ctx.repo_root, branch=node.branch_name
    )
    if existing_path is not None:
        node.worktree_path = str(existing_path)
        await session.flush()
        return existing_path

    worktree_path = default_worktree_path(ctx, branch=node.branch_name)

    if worktree_path.exists():
        branch = current_branch(cwd=worktree_path)
        if branch != node.branch_name:
            raise RuntimeError(
                f"Worktree path already exists but is on {branch!r} "
                f"(expected {node.branch_name!r}): {worktree_path}"
            )
    else:
        try:
            git_worktree_add(
                ctx.repo_root,
                worktree_path=worktree_path,
                branch_name=node.branch_name,
                base_ref=base_ref,
            )
        except GitCommandError as e:
            raise RuntimeError(str(e)) from e

    node.worktree_path = str(worktree_path)
    await session.flush()
    return worktree_path


async def load_node_agent(session: AsyncSession, *, node: Node) -> Agent | None:
    if node.agent_id is None:
        return None
    agent = await session.get(Agent, node.agent_id)
    if agent is None:
        raise RuntimeError("Node has an invalid agent_id")
    return agent


async def get_or_create_task_agent(
    session: AsyncSession, *, task: Task, node: Node
) -> Agent:
    agent = await load_node_agent(session, node=node)
    if agent is not None:
        return agent

    expected = f"a-{task.id}"
    agent = await session.scalar(select(Agent).where(Agent.display_name == expected))
    if agent is None:
        agent = Agent(display_name=expected)
        session.add(agent)
        await session.flush()
    else:
        other_node = await session.scalar(
            select(Node).where(Node.agent_id == agent.id, Node.id != node.id).limit(1)
        )
        if other_node is not None:
            raise RuntimeError(
                f"Agent {expected!r} is already assigned to node {other_node.id} ({other_node.branch_name})"
            )

    node.agent_id = agent.id
    await session.flush()
    return agent


@dataclass(frozen=True, slots=True)
class StartAgentResult:
    agent: Agent
    attach: AttachInfo
    started: bool = True
    warnings: tuple[str, ...] = ()


async def start_tmux_session(
    ctx: RepoContext,
    *,
    task_id: int,
    agent_id: int,
    worktree_path: Path,
    argv: list[str],
    env: dict[str, str],
) -> AttachTmux:
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    log_path = agent_log_path(ctx, agent_id=agent_id)
    script_path = write_agent_launcher(ctx=ctx, agent_id=agent_id, argv=argv, env=env)

    _tmux_kill_session(name=tmux_name)
    _tmux_new_session(name=tmux_name, cwd=worktree_path, script_path=script_path)
    _tmux_pipe_to_log(name=tmux_name, log_path=log_path)
    return AttachTmux(session=tmux_name, socket_path=None, log_path=str(log_path))


async def _load_task_and_node(
    session: AsyncSession,
    *,
    task_id: int,
) -> tuple[Task, Node, Epic]:
    task = await session.get(Task, task_id)
    if task is None:
        raise RuntimeError(f"Unknown task id: {task_id}")
    if task.node_id is None:
        raise RuntimeError("Task has no node backing; sync docs to create nodes first")

    node = await session.get(Node, task.node_id)
    if node is None:
        raise RuntimeError("Node not found for task")

    epic = await session.get(Epic, node.epic_id)
    if epic is None:
        raise RuntimeError("Epic not found for task")

    return task, node, epic


async def start_task_agent(
    ctx: RepoContext,
    *,
    task_id: int,
    harness_command: str,
    detach: bool,
) -> StartAgentResult:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            task, node, epic = await _load_task_and_node(session, task_id=task_id)
            agent = await get_or_create_task_agent(
                session=session, task=task, node=node
            )

            if not detach:
                raise RuntimeError(
                    "v0 requires tmux-backed detached agents (omit --no-detach)"
                )
            if not has_tmux():
                raise RuntimeError(
                    "tmux is required for v0 agents; install tmux or set up a tmux-capable runner"
                )

            tmux_name = tmux_session_name_for_task(task_id=task.id)
            if _tmux_has_session(name=tmux_name):
                now = datetime.now(UTC)
                agent.status = AgentStatus.Running
                agent.last_seen_at = now
                if agent.started_at is None:
                    agent.started_at = now
                agent.ended_at = None

                log_path = agent_log_path(ctx, agent_id=agent.id)
                try:
                    _tmux_pipe_to_log(name=tmux_name, log_path=log_path)
                except RuntimeError:
                    pass

                attach = AttachTmux(
                    session=tmux_name,
                    socket_path=None,
                    log_path=str(log_path),
                )
                agent.attach = attach.model_dump(mode="python")
                await session.commit()
                await session.refresh(agent)
                return StartAgentResult(
                    agent=agent,
                    attach=attach,
                    started=False,
                    warnings=(),
                )

            host = await ensure_host_row(session, ctx)
            worktree_path = await ensure_node_worktree(
                session, ctx, node=node, epic=epic
            )

            argv = _parse_harness_command(harness_command)
            definition = HarnessProfileDefinition(argv=argv)
            profile_id = harness_profile_id_for_definition(argv[0], definition)
            profile = await ensure_harness_profile_row(
                session,
                profile_id=profile_id,
                kind=argv[0],
                definition=definition,
            )

            warnings: list[str] = []
            shim_path = shutil.which("rn")
            runtime_env = {
                "REDESMYN_AGENT_ID": str(agent.id),
                "REDESMYN_TASK_ID": str(task.id),
                "REDESMYN_NODE_ID": str(node.id),
                "REDESMYN_HOST_KEY": host.host_key,
                "PATH": os.environ.get("PATH", ""),
            }
            if shim_path is not None:
                shim = create_git_shim(ctx=ctx, agent_id=agent.id)
                runtime_env["PATH"] = f"{shim.dir}{os.pathsep}{runtime_env['PATH']}"
            else:
                warnings.append(
                    "`rn` not found on PATH; skipping git shim injection (agent git will bypass blocks)"
                )
            runtime_env |= definition.env

            attach = await start_tmux_session(
                ctx,
                task_id=task.id,
                agent_id=agent.id,
                worktree_path=worktree_path,
                argv=definition.argv,
                env=runtime_env,
            )
            pid = None

            now = datetime.now(UTC)
            agent.status = AgentStatus.Running
            agent.last_seen_at = now
            agent.host_id = host.id
            agent.harness_profile_id = profile.id
            agent.cwd_path = str(worktree_path)
            agent.pid = pid
            agent.attach = attach.model_dump(mode="python")
            agent.resolved_profile = definition.model_dump(mode="python")
            agent.exit_code = None
            agent.started_at = now
            agent.ended_at = None

            await session.commit()
            await session.refresh(agent)
            return StartAgentResult(
                agent=agent,
                attach=attach,
                started=True,
                warnings=tuple(warnings),
            )
    finally:
        await engine.dispose()


async def stop_task_agent(
    ctx: RepoContext,
    *,
    task_id: int,
) -> bool:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            task, node, _ = await _load_task_and_node(session, task_id=task_id)
            agent = await load_node_agent(session, node=node)
            if agent is None:
                return False

            tmux_name = tmux_session_name_for_task(task_id=task.id)
            was_running = has_tmux() and _tmux_has_session(name=tmux_name)
            if was_running:
                _tmux_kill_session(name=tmux_name)
            else:
                if agent.status in {AgentStatus.Running, AgentStatus.Blocked}:
                    now = datetime.now(UTC)
                    agent.status = AgentStatus.Error
                    agent.last_seen_at = now
                    agent.ended_at = now
                    agent.pid = None
                    await session.commit()
                return False

            now = datetime.now(UTC)
            agent.status = AgentStatus.Idle
            agent.last_seen_at = now
            agent.ended_at = now
            agent.pid = None

            await session.commit()
            return True
    finally:
        await engine.dispose()


async def restart_task_agent(
    ctx: RepoContext,
    *,
    task_id: int,
    harness_command: str | None,
    detach: bool,
) -> StartAgentResult:
    if harness_command is None:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                task, node, _ = await _load_task_and_node(session, task_id=task_id)
                agent = await load_node_agent(session, node=node)
                if agent is None:
                    raise RuntimeError(
                        "No prior agent run found; pass --harness to restart"
                    )
                if agent.resolved_profile is None:
                    raise RuntimeError(
                        "No prior agent run found; pass --harness to restart"
                    )
                definition = TypeAdapter(HarnessProfileDefinition).validate_python(
                    agent.resolved_profile
                )
                harness_command = shlex.join(definition.argv)
        finally:
            await engine.dispose()

    if not detach:
        raise RuntimeError("v0 requires tmux-backed detached agents (omit --no-detach)")
    if not has_tmux():
        raise RuntimeError(
            "tmux is required for v0 agents; install tmux or set up a tmux-capable runner"
        )
    _tmux_kill_session(name=tmux_session_name_for_task(task_id=task_id))
    return await start_task_agent(
        ctx,
        task_id=task_id,
        harness_command=harness_command,
        detach=detach,
    )


async def load_task_agent(
    ctx: RepoContext,
    *,
    task_id: int,
    active_only: bool = True,
) -> Agent | None:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            task, node, _ = await _load_task_and_node(session, task_id=task_id)
            agent = await load_node_agent(session, node=node)
            if agent is None:
                return None
            if active_only:
                tmux_name = tmux_session_name_for_task(task_id=task.id)
                if not (has_tmux() and _tmux_has_session(name=tmux_name)):
                    return None
            return agent
    finally:
        await engine.dispose()


def agent_log_path_for_row(ctx: RepoContext, *, agent_row: Agent) -> Path:
    attach = TypeAdapter(AttachInfo).validate_python(agent_row.attach)
    if isinstance(attach, (AttachExternal, AttachTmux)) and attach.log_path:
        return Path(attach.log_path)
    return agent_log_path(ctx, agent_id=agent_row.id)


def attach_agent(*, agent_row: Agent) -> int:
    attach = TypeAdapter(AttachInfo).validate_python(agent_row.attach)
    if isinstance(attach, AttachTmux):
        return _tmux_attach(name=attach.session)
    raise RuntimeError("Attach is not available for this agent")
