from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.context import RepoContext
from redesmyn.db import (
    Agent,
    AgentSession,
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
    AttachNone,
    AttachTmux,
    HarnessProfileDefinition,
    HostCapabilities,
)
from redesmyn.domain.enums import AgentSessionStatus, AgentStatus, HarnessProfileSource
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


def create_git_shim(*, ctx: RepoContext, session_id: int) -> GitShim:
    shim_dir = ctx.state_dir / "shims" / f"session-{session_id}"
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


def session_dir(ctx: RepoContext, *, session_id: int) -> Path:
    return ctx.state_dir / "sessions" / str(session_id)


def session_log_path(ctx: RepoContext, *, session_id: int) -> Path:
    return session_dir(ctx, session_id=session_id) / "output.log"


def write_session_launcher(
    *,
    ctx: RepoContext,
    session_id: int,
    argv: list[str],
    env: dict[str, str],
) -> Path:
    run_dir = session_dir(ctx, session_id=session_id)
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


async def get_active_session_for_node(
    session: AsyncSession, *, node_id: int
) -> AgentSession | None:
    return await session.scalar(
        select(AgentSession)
        .where(AgentSession.node_id == node_id, AgentSession.ended_at.is_(None))
        .order_by(AgentSession.id.desc())
        .limit(1)
    )


async def get_active_session_for_agent(
    session: AsyncSession, *, agent_id: int
) -> AgentSession | None:
    return await session.scalar(
        select(AgentSession)
        .where(AgentSession.agent_id == agent_id, AgentSession.ended_at.is_(None))
        .order_by(AgentSession.id.desc())
        .limit(1)
    )


async def ensure_task_agent(*, session: AsyncSession, task: Task, node: Node) -> Agent:
    expected = f"a-{task.id}"

    if node.agent_id is not None:
        agent = await session.get(Agent, node.agent_id)
        if agent is None:
            raise RuntimeError("Node has an invalid agent_id")
        if agent.display_name != expected:
            raise RuntimeError(
                f"Node is assigned to {agent.display_name!r} but expected {expected!r}"
            )
        return agent

    agent = await session.scalar(select(Agent).where(Agent.display_name == expected))
    if agent is None:
        agent = Agent(display_name=expected)
        session.add(agent)
        await session.flush()
    else:
        other = await session.scalar(
            select(Node).where(Node.agent_id == agent.id, Node.id != node.id).limit(1)
        )
        if other is not None:
            raise RuntimeError(
                f"Agent {expected!r} is already assigned to node {other.id} ({other.branch_name})"
            )

    node.agent_id = agent.id
    await session.flush()
    return agent


@dataclass(frozen=True, slots=True)
class StartSessionResult:
    session: AgentSession
    attach: AttachInfo
    started: bool = True
    warnings: tuple[str, ...] = ()


async def start_tmux_session(
    ctx: RepoContext,
    *,
    task_id: int,
    session_id: int,
    worktree_path: Path,
    argv: list[str],
    env: dict[str, str],
) -> AttachTmux:
    tmux_name = tmux_session_name_for_task(task_id=task_id)
    log_path = session_log_path(ctx, session_id=session_id)
    script_path = write_session_launcher(
        ctx=ctx, session_id=session_id, argv=argv, env=env
    )

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


async def start_agent_session_for_node(
    ctx: RepoContext,
    *,
    epic_slug: str | None,
    node_id: int,
    harness: str,
    argv: list[str] | None,
    detach: bool,
) -> StartSessionResult:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            node = await session.get(Node, node_id)
            if node is None:
                raise RuntimeError(f"Unknown node id: {node_id}")

            epic = await session.get(Epic, node.epic_id)
            if epic is None:
                raise RuntimeError("Epic not found for node")
            if epic_slug and epic.slug != epic_slug:
                raise RuntimeError(f"Node {node_id} is not in epic {epic_slug!r}")

            if node.agent_id is None:
                raise RuntimeError("Node has no agent assigned; assign one first")

            agent = await session.get(Agent, node.agent_id)
            if agent is None:
                raise RuntimeError("Assigned agent not found")

            existing = await get_active_session_for_node(session, node_id=node.id)
            if existing is not None:
                raise RuntimeError(
                    f"Node already has an active session (id={existing.id})"
                )

            host = await ensure_host_row(session, ctx)
            worktree_path = await ensure_node_worktree(
                session, ctx, node=node, epic=epic
            )

            definition = HarnessProfileDefinition(argv=argv or [harness])
            profile_id = harness_profile_id_for_definition(harness, definition)
            profile = await ensure_harness_profile_row(
                session,
                profile_id=profile_id,
                kind=harness,
                definition=definition,
            )

            session_row = AgentSession(
                agent_id=agent.id,
                node_id=node.id,
                host_id=host.id,
                harness_profile_id=profile.id,
                status=AgentSessionStatus.Starting,
                cwd_path=str(worktree_path),
                pid=None,
                attach=AttachNone().model_dump(mode="python"),
                resolved_profile=definition.model_dump(mode="python"),
                started_at=datetime.now(UTC),
                ended_at=None,
            )
            session.add(session_row)
            try:
                await session.flush()
            except IntegrityError as e:
                raise RuntimeError("Agent or node already has an active session") from e

            shim = create_git_shim(ctx=ctx, session_id=session_row.id)
            runtime_env = {
                "REDESMYN_AGENT_ID": str(agent.id),
                "REDESMYN_NODE_ID": str(node.id),
                "REDESMYN_SESSION_ID": str(session_row.id),
                "REDESMYN_HOST_KEY": host.host_key,
                "PATH": f"{shim.dir}{os.pathsep}{os.environ.get('PATH', '')}",
            }
            runtime_env |= definition.env

            attach: AttachInfo
            if detach and has_tmux():
                attach = await start_tmux_session(
                    ctx,
                    task_id=node.primary_task_id or node.id,
                    session_id=session_row.id,
                    worktree_path=worktree_path,
                    argv=definition.argv,
                    env=runtime_env,
                )
            else:
                log_path = session_log_path(ctx, session_id=session_row.id)
                script_path = write_session_launcher(
                    ctx=ctx,
                    session_id=session_row.id,
                    argv=definition.argv,
                    env=runtime_env,
                )
                with log_path.open("ab") as log_fp:
                    proc = subprocess.Popen(
                        [str(script_path)],
                        cwd=str(worktree_path),
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        env=os.environ.copy(),
                    )
                attach = AttachExternal(
                    hint=f"Started pid {proc.pid} (no tmux); logs: {log_path}",
                    log_path=str(log_path),
                )
                session_row.pid = proc.pid

            session_row.attach = attach.model_dump(mode="python")
            session_row.status = AgentSessionStatus.Running
            agent.status = AgentStatus.Running
            agent.last_seen_at = datetime.now(UTC)

            await session.commit()
            await session.refresh(session_row)
            return StartSessionResult(session=session_row, attach=attach)
    finally:
        await engine.dispose()


async def start_task_agent_session(
    ctx: RepoContext,
    *,
    task_id: int,
    harness_command: str,
    detach: bool,
) -> StartSessionResult:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            task, node, epic = await _load_task_and_node(session, task_id=task_id)
            agent = await ensure_task_agent(session=session, task=task, node=node)

            existing = await get_active_session_for_node(session, node_id=node.id)
            if existing is not None:
                attach = TypeAdapter(AttachInfo).validate_python(existing.attach)
                return StartSessionResult(
                    session=existing,
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

            session_row = AgentSession(
                agent_id=agent.id,
                node_id=node.id,
                host_id=host.id,
                harness_profile_id=profile.id,
                status=AgentSessionStatus.Starting,
                cwd_path=str(worktree_path),
                pid=None,
                attach=AttachNone().model_dump(mode="python"),
                resolved_profile=definition.model_dump(mode="python"),
                started_at=datetime.now(UTC),
                ended_at=None,
            )
            session.add(session_row)
            try:
                await session.flush()
            except IntegrityError as e:
                raise RuntimeError("Agent or node already has an active session") from e

            warnings: list[str] = []
            shim_path = shutil.which("rn")
            runtime_env = {
                "REDESMYN_AGENT_ID": str(agent.id),
                "REDESMYN_TASK_ID": str(task.id),
                "REDESMYN_NODE_ID": str(node.id),
                "REDESMYN_SESSION_ID": str(session_row.id),
                "REDESMYN_HOST_KEY": host.host_key,
                "PATH": os.environ.get("PATH", ""),
            }
            if shim_path is not None:
                shim = create_git_shim(ctx=ctx, session_id=session_row.id)
                runtime_env["PATH"] = f"{shim.dir}{os.pathsep}{runtime_env['PATH']}"
            else:
                warnings.append(
                    "`rn` not found on PATH; skipping git shim injection (agent git will bypass blocks)"
                )
            runtime_env |= definition.env

            attach: AttachInfo
            if detach and has_tmux():
                attach = await start_tmux_session(
                    ctx,
                    task_id=task.id,
                    session_id=session_row.id,
                    worktree_path=worktree_path,
                    argv=definition.argv,
                    env=runtime_env,
                )
            else:
                log_path = session_log_path(ctx, session_id=session_row.id)
                script_path = write_session_launcher(
                    ctx=ctx,
                    session_id=session_row.id,
                    argv=definition.argv,
                    env=runtime_env,
                )
                with log_path.open("ab") as log_fp:
                    proc = subprocess.Popen(
                        [str(script_path)],
                        cwd=str(worktree_path),
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        env=os.environ.copy(),
                    )
                attach = AttachExternal(
                    hint=f"Started pid {proc.pid} (no tmux); logs: {log_path}",
                    log_path=str(log_path),
                )
                session_row.pid = proc.pid

            session_row.attach = attach.model_dump(mode="python")
            session_row.status = AgentSessionStatus.Running
            agent.status = AgentStatus.Running
            agent.last_seen_at = datetime.now(UTC)

            await session.commit()
            await session.refresh(session_row)
            return StartSessionResult(
                session=session_row,
                attach=attach,
                started=True,
                warnings=tuple(warnings),
            )
    finally:
        await engine.dispose()


async def stop_agent_session(
    ctx: RepoContext,
    *,
    session_id: int | None = None,
    node_id: int | None = None,
) -> AgentSession | None:
    if session_id is None and node_id is None:
        raise ValueError("Pass session_id or node_id")

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            row: AgentSession | None = None
            if session_id is not None:
                row = await session.get(AgentSession, session_id)
            elif node_id is not None:
                row = await get_active_session_for_node(session, node_id=node_id)

            if row is None or row.ended_at is not None:
                return None

            attach = TypeAdapter(AttachInfo).validate_python(row.attach)
            if isinstance(attach, AttachTmux):
                _tmux_kill_session(name=attach.session)
            else:
                if row.pid is not None:
                    try:
                        os.kill(row.pid, signal.SIGTERM)
                    except OSError:
                        pass

            row.status = AgentSessionStatus.Stopped
            row.ended_at = datetime.now(UTC)

            agent = await session.get(Agent, row.agent_id)
            if agent is not None:
                agent.status = AgentStatus.Idle
                agent.last_seen_at = datetime.now(UTC)

            await session.commit()
            await session.refresh(row)
            return row
    finally:
        await engine.dispose()


async def stop_task_agent_session(
    ctx: RepoContext,
    *,
    task_id: int,
) -> AgentSession | None:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            _, node, _ = await _load_task_and_node(session, task_id=task_id)
            row = await get_active_session_for_node(session, node_id=node.id)
            if row is None:
                return None

            attach = TypeAdapter(AttachInfo).validate_python(row.attach)
            if isinstance(attach, AttachTmux):
                _tmux_kill_session(name=attach.session)
            else:
                if row.pid is not None:
                    try:
                        os.kill(row.pid, signal.SIGTERM)
                    except OSError:
                        pass

            row.status = AgentSessionStatus.Stopped
            row.ended_at = datetime.now(UTC)

            agent = await session.get(Agent, row.agent_id)
            if agent is not None:
                agent.status = AgentStatus.Idle
                agent.last_seen_at = datetime.now(UTC)

            await session.commit()
            await session.refresh(row)
            return row
    finally:
        await engine.dispose()


async def restart_task_agent_session(
    ctx: RepoContext,
    *,
    task_id: int,
    harness_command: str | None,
    detach: bool,
) -> StartSessionResult:
    if harness_command is None:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                _, node, _ = await _load_task_and_node(session, task_id=task_id)
                last = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.node_id == node.id)
                    .order_by(AgentSession.id.desc())
                    .limit(1)
                )
                if last is None or last.resolved_profile is None:
                    raise RuntimeError(
                        "No prior agent run found; pass --harness to restart"
                    )
                definition = TypeAdapter(HarnessProfileDefinition).validate_python(
                    last.resolved_profile
                )
                harness_command = shlex.join(definition.argv)
        finally:
            await engine.dispose()

    await stop_task_agent_session(ctx, task_id=task_id)
    return await start_task_agent_session(
        ctx,
        task_id=task_id,
        harness_command=harness_command,
        detach=detach,
    )


async def load_agent_session(
    ctx: RepoContext,
    *,
    session_id: int | None = None,
    node_id: int | None = None,
    active_only: bool = True,
) -> AgentSession | None:
    if session_id is None and node_id is None:
        raise ValueError("Pass session_id or node_id")

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            if session_id is not None:
                row = await session.get(AgentSession, session_id)
                if row is None:
                    return None
                if active_only and row.ended_at is not None:
                    return None
                return row

            row = await get_active_session_for_node(session, node_id=node_id or 0)
            if row is None:
                return None
            if active_only and row.ended_at is not None:
                return None
            return row
    finally:
        await engine.dispose()


async def load_task_agent_session(
    ctx: RepoContext,
    *,
    task_id: int,
    active_only: bool = True,
) -> AgentSession | None:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            _, node, _ = await _load_task_and_node(session, task_id=task_id)
            if active_only:
                return await get_active_session_for_node(session, node_id=node.id)

            return await session.scalar(
                select(AgentSession)
                .where(AgentSession.node_id == node.id)
                .order_by(AgentSession.id.desc())
                .limit(1)
            )
    finally:
        await engine.dispose()


def session_log_path_for_row(ctx: RepoContext, *, session_row: AgentSession) -> Path:
    attach = TypeAdapter(AttachInfo).validate_python(session_row.attach)
    if isinstance(attach, (AttachExternal, AttachTmux)) and attach.log_path:
        return Path(attach.log_path)
    return session_log_path(ctx, session_id=session_row.id)


def attach_agent_session(*, session_row: AgentSession) -> int:
    attach = TypeAdapter(AttachInfo).validate_python(session_row.attach)
    if isinstance(attach, AttachTmux):
        return _tmux_attach(name=attach.session)
    raise RuntimeError("Attach is not available for this session type")
