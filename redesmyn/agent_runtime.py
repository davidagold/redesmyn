from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from pydantic import TypeAdapter
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.agent_kind import resolve_agent_kind
from redesmyn.agent_label import agent_label_for_task_id
from redesmyn.context import RepoContext
from redesmyn.host_identity import HostIdentity, load_or_create_host_identity
from redesmyn.db import (
    AgentSession,
    Epic,
    LaunchConfiguration,
    Host,
    Task,
    create_engine,
    create_sessionmaker,
)
from redesmyn.db.models import (
    AttachExternal,
    AttachInfo,
    AttachTmux,
    LaunchConfigurationDefinition,
    HostCapabilities,
)
from redesmyn.agent_prelude import DEFAULT_AGENT_PRELUDE_TEMPLATE
from redesmyn.domain.enums import (
    AgentKindSelection,
    AgentStatus,
    LaunchConfigurationSource,
    TaskState,
)
from redesmyn.integrations.linear_automation import maybe_push_task_state_to_linear
from redesmyn.orchestration_config import load_orchestration_defaults
from redesmyn.repo import (
    GitCommandError,
    current_branch,
    git_is_ancestor,
    git_worktree_add,
)
from redesmyn.sandbox import (
    NullSandboxPolicy,
    WorktreeSandboxPolicy,
    make_sandbox_provider,
)
from redesmyn.strings import slugify


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


def create_git_shim(*, ctx: RepoContext, task_id: int) -> GitShim:
    shim_dir = ctx.state_dir / "shims" / f"task-{task_id}"
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


def task_dir(ctx: RepoContext, *, task_id: int) -> Path:
    return ctx.state_dir / "tasks" / str(task_id)


def task_agent_runtime_dir(ctx: RepoContext, *, task_id: int) -> Path:
    return task_dir(ctx, task_id=task_id) / "agent-runtime"


def agent_session_dir(ctx: RepoContext, *, task_id: int, session_id: int) -> Path:
    return task_dir(ctx, task_id=task_id) / "agent-sessions" / str(session_id)


def agent_session_log_path(ctx: RepoContext, *, task_id: int, session_id: int) -> Path:
    return agent_session_dir(ctx, task_id=task_id, session_id=session_id) / "output.log"


def _tail_text(path: Path, *, max_bytes: int = 8192) -> str | None:
    try:
        with path.open("rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                offset = max(0, size - max_bytes)
                f.seek(offset, os.SEEK_SET)
            except OSError:
                pass
            data = f.read()
    except OSError:
        return None
    if not data:
        return None
    return data.decode("utf-8", errors="replace")


def _sanitize_path_for_sandbox(path_value: str) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for raw in path_value.split(os.pathsep):
        entry = raw.strip()
        if not entry or entry in seen:
            continue
        path = Path(entry)
        if not path.is_dir():
            continue
        # Codex may try to write under certain PATH entries; drop ephemeral
        # temp paths that are non-writable under worktree sandboxing.
        if entry.startswith("/var/folders/") or entry.startswith(
            "/private/var/folders/"
        ):
            continue
        parts.append(entry)
        seen.add(entry)

    for required in ("/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"):
        if required not in seen and Path(required).is_dir():
            parts.append(required)
            seen.add(required)

    return os.pathsep.join(parts)


def _seed_codex_home(*, src_dir: Path, dst_dir: Path, warnings: list[str]) -> None:
    """Best-effort seed of Codex auth/config into a sandboxed home."""
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        warnings.append(f"Failed to create CODEX_HOME directory: {e}")
        return

    if not src_dir.is_dir():
        return

    for name in (
        "auth.json",
        "config.toml",
        "version.json",
    ):
        src_file = src_dir / name
        dst_file = dst_dir / name
        if not src_file.is_file() or dst_file.exists():
            continue
        try:
            shutil.copy2(src_file, dst_file)
        except OSError as e:
            warnings.append(f"Failed to copy {name} into CODEX_HOME: {e}")

    # Ensure Playwright MCP uses isolated mode so multiple sandboxes/agents can
    # use browser tooling concurrently without profile locking.
    config_path = dst_dir / "config.toml"
    if config_path.exists():
        try:
            text = config_path.read_text(encoding="utf-8")
            if "--isolated" not in text and "mcp_servers.playwright" in text:
                text = text.replace(
                    'args = ["@playwright/mcp@latest"]',
                    'args = ["@playwright/mcp@latest", "--isolated"]',
                )
                config_path.write_text(text, encoding="utf-8")
        except OSError as e:
            warnings.append(f"Failed to patch CODEX_HOME config.toml: {e}")


def write_agent_launcher(
    *,
    ctx: RepoContext,
    task_id: int,
    session_id: int,
    argv: list[str],
    env: dict[str, str],
) -> Path:
    run_dir = agent_session_dir(ctx, task_id=task_id, session_id=session_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "output.log"

    lines = [
        "#!/bin/sh",
        "set -eu",
        f"LOG_PATH={shlex.quote(str(log_path))}",
        # Capture launcher output (including early failures) in the log file, then
        # restore the harness stdout/stderr so interactive harnesses keep a TTY.
        "exec 3>&1 4>&2",
        'exec >>"$LOG_PATH" 2>&1',
    ]
    for key, value in env.items():
        lines.append(f"export {key}={shlex.quote(value)}")

    # Give tmux a beat to attach pipe-pane so we don't miss very early harness
    # output in the log.
    lines.append("sleep 0.2")
    lines.append(f"exec {shlex.join(argv)} 1>&3 2>&4")
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


def _tmux_send_literal(*, name: str, text: str) -> None:
    proc = subprocess.run(
        ["tmux", "send-keys", "-t", name, "-l", "--", text],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "tmux send-keys failed")


def _tmux_send_enter(*, name: str) -> None:
    proc = subprocess.run(
        ["tmux", "send-keys", "-t", name, "Enter"],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "tmux send-keys Enter failed")


def _tmux_send_lines(*, name: str, lines: list[str]) -> None:
    for line in lines:
        if line:
            _tmux_send_literal(name=name, text=line)
        _tmux_send_enter(name=name)


def _tmux_capture_pane(*, name: str) -> str:
    proc = subprocess.run(
        ["tmux", "capture-pane", "-p", "-t", name],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "tmux capture-pane failed")
    return proc.stdout


async def _wait_for_agent_input_ready(*, name: str, timeout_s: float) -> None:
    """Best-effort readiness check.

    Some harness UIs (notably Codex TUI) may buffer/ignore early keystrokes during
    startup. We wait briefly for recognizable prompts before sending the prelude.
    """

    deadline = asyncio.get_running_loop().time() + max(0.0, timeout_s)
    while True:
        if not _tmux_has_session(name=name):
            return
        try:
            pane = _tmux_capture_pane(name=name)
        except RuntimeError:
            pane = ""
        if (
            "OpenAI Codex" in pane
            or "100% context left" in pane
            or "\n> " in pane
            or ">_" in pane
        ):
            return
        now = asyncio.get_running_loop().time()
        if now >= deadline:
            return
        await asyncio.sleep(0.1)


async def ensure_host_row(session: AsyncSession, ctx: RepoContext) -> Host:
    config: HostIdentity = load_or_create_host_identity(ctx)
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
    prefix = os.environ.get("REDESMYN_TMUX_SESSION_PREFIX", "rn-a").strip()
    if not prefix:
        prefix = "rn-a"
    return f"{prefix}-{task_id}"


def _parse_harness_command(command: str) -> list[str]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("Harness command is empty")
    return argv


async def ensure_launch_configuration_row(
    session: AsyncSession,
    *,
    profile_id: str,
    kind: str,
    definition: LaunchConfigurationDefinition,
) -> LaunchConfiguration:
    profile = await session.get(LaunchConfiguration, profile_id)
    if profile is not None:
        return profile

    profile = LaunchConfiguration(
        id=profile_id,
        kind=kind,
        source=LaunchConfigurationSource.Builtin,
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


def find_existing_worktree_path_for_branch(
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


def launch_configuration_id_for_definition(
    kind: str, definition: LaunchConfigurationDefinition
) -> str:
    normalized = json.dumps(
        definition.model_dump(mode="python"), sort_keys=True
    ).encode("utf-8")
    digest = sha256(normalized).hexdigest()[:12]
    return f"{slugify(kind)}/sha256-{digest}"


_TASK_TITLE_ID_RE = re.compile(r"^(T-\d+)\b")


class _BranchNameTask(Protocol):
    id: int
    title: str
    linear_identifier: str | None


def _default_branch_name_for_task(*, epic_slug: str, task: _BranchNameTask) -> str:
    identifier = task.linear_identifier
    if not identifier:
        match = _TASK_TITLE_ID_RE.match(task.title.strip())
        identifier = match.group(1) if match else f"task-{task.id}"

    title = (task.title or "").strip()
    title_remainder = title
    if identifier and title.startswith(identifier):
        title_remainder = title[len(identifier) :].strip()
    if title_remainder.startswith("-"):
        title_remainder = title_remainder[1:].strip()

    # Prefer a concise "task abbreviation" (similar to the UI branch label):
    # - drop parenthetical detail
    # - take the first '+'-separated segment
    # - drop common boilerplate suffixes
    title_remainder = re.sub(r"\([^)]*\)", " ", title_remainder).strip()
    title_remainder = title_remainder.split("+", 1)[0].strip()
    title_remainder = re.sub(r"\bimplementation\b", "", title_remainder, flags=re.I)
    title_remainder = re.sub(r"\s+", " ", title_remainder).strip()

    short = slugify(title_remainder or title, fallback="task")[:60].strip("-") or "task"
    return f"rn/{epic_slug}/{identifier}-{short}"


async def ensure_task_worktree(
    session: AsyncSession,
    ctx: RepoContext,
    *,
    task: Task,
    epic: Epic,
) -> Path:
    if task.branch_name is None:
        branch = _default_branch_name_for_task(epic_slug=epic.slug, task=task)
        existing = await session.scalar(
            select(Task).where(
                Task.epic_id == epic.id,
                Task.branch_name == branch,
                Task.id != task.id,
            )
        )
        if existing is not None:
            branch = f"{branch}-{task.id}"
        task.branch_name = branch
        await session.flush()

    branch_name = task.branch_name
    if branch_name is None:
        raise RuntimeError("Task branch name missing after branch assignment")

    if task.worktree_path:
        path = Path(task.worktree_path)
        if path.exists():
            return path
        task.worktree_path = None
        await session.flush()

    parent_task: Task | None = None
    if task.parent_task_id is not None:
        parent_task = await session.get(Task, task.parent_task_id)

    base_ref = epic.root_branch
    base_task = parent_task
    while base_task is not None and base_task.branch_name is not None:
        if git_is_ancestor(ctx.repo_root, base_task.branch_name, epic.root_branch):
            base_task = (
                await session.get(Task, base_task.parent_task_id)
                if base_task.parent_task_id is not None
                else None
            )
            continue

        if base_task.state == TaskState.Done:
            raise RuntimeError(
                f"Parent task {base_task.id} is marked done but {epic.root_branch!r} "
                f"does not contain {base_task.branch_name!r}. Fast-forward the base branch."
            )

        base_ref = base_task.branch_name
        break

    branch_name = task.branch_name
    if branch_name is None:
        raise RuntimeError("Task branch_name missing after allocation")

    existing_path = find_existing_worktree_path_for_branch(
        ctx.repo_root, branch=branch_name
    )
    if existing_path is not None:
        task.worktree_path = str(existing_path)
        await session.flush()
        return existing_path

    worktree_path = default_worktree_path(ctx, branch=branch_name)

    if worktree_path.exists():
        branch = current_branch(cwd=worktree_path)
        if branch != branch_name:
            raise RuntimeError(
                f"Worktree path already exists but is on {branch!r} "
                f"(expected {branch_name!r}): {worktree_path}"
            )
    else:
        try:
            git_worktree_add(
                ctx.repo_root,
                worktree_path=worktree_path,
                branch_name=branch_name,
                base_ref=base_ref,
            )
        except GitCommandError as e:
            raise RuntimeError(str(e)) from e

    task.worktree_path = str(worktree_path)
    await session.flush()
    return worktree_path


async def load_latest_task_agent_session_row(
    session: AsyncSession, *, task: Task
) -> AgentSession | None:
    return await session.scalar(
        select(AgentSession)
        .where(AgentSession.task_id == task.id)
        .order_by(desc(AgentSession.id))
        .limit(1)
    )


async def load_active_task_agent_session_row(
    session: AsyncSession, *, task_id: int
) -> AgentSession | None:
    return await session.scalar(
        select(AgentSession)
        .where(AgentSession.task_id == task_id)
        .where(AgentSession.ended_at.is_(None))
        .order_by(desc(AgentSession.id))
        .limit(1)
    )


async def ensure_active_task_agent_session_row(
    session: AsyncSession, *, task_id: int, now: datetime
) -> AgentSession:
    current = await load_active_task_agent_session_row(session, task_id=task_id)
    if current is None:
        current = AgentSession(
            task_id=task_id,
            status=AgentStatus.Running,
            started_at=now,
            ended_at=None,
        )
        session.add(current)
        await session.flush()
        return current

    if current.started_at is None:
        current.started_at = now
    current.ended_at = None
    return current


@dataclass(frozen=True, slots=True)
class StartAgentResult:
    agent_label: str
    agent_session: AgentSession
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
    log_path = agent_session_log_path(ctx, task_id=task_id, session_id=session_id)
    script_path = write_agent_launcher(
        ctx=ctx, task_id=task_id, session_id=session_id, argv=argv, env=env
    )

    _tmux_kill_session(name=tmux_name)
    _tmux_new_session(name=tmux_name, cwd=worktree_path, script_path=script_path)
    _tmux_pipe_to_log(name=tmux_name, log_path=log_path)
    return AttachTmux(session=tmux_name, socket_path=None, log_path=str(log_path))


async def _load_task_and_epic(
    session: AsyncSession,
    *,
    task_id: int,
) -> tuple[Task, Epic]:
    task = await session.get(Task, task_id)
    if task is None:
        raise RuntimeError(f"Unknown task id: {task_id}")

    epic = await session.get(Epic, task.epic_id)
    if epic is None:
        raise RuntimeError("Epic not found for task")

    return task, epic


async def checkout_task_worktree(ctx: RepoContext, *, task_id: int) -> Path:
    """Ensure a task's worktree exists and is recorded on the Task row."""
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            task, epic = await _load_task_and_epic(session, task_id=task_id)
            path = await ensure_task_worktree(session, ctx, task=task, epic=epic)
            await session.commit()
            return path
    finally:
        await engine.dispose()


def _agent_prelude_lines(
    *,
    task: Task,
    epic: Epic,
    worktree_path: Path,
    template: str | None,
) -> list[str]:
    default_template = DEFAULT_AGENT_PRELUDE_TEMPLATE

    class SafeDict(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    context = SafeDict(
        task_id=str(task.id),
        task_title=task.title,
        task_doc=task.local_path or "(unknown path)",
        epic_slug=epic.slug,
        epic_readme=f"epics/{epic.slug}/README.md",
        branch=task.branch_name or "(no branch)",
        worktree=str(worktree_path),
    )

    chosen = template or default_template
    try:
        rendered = chosen.format_map(context)
    except Exception:
        rendered = chosen

    lines = rendered.strip("\n").splitlines()
    return ["", *lines, ""]


async def send_agent_prelude(
    *,
    task: Task,
    epic: Epic,
    worktree_path: Path,
    template: str | None,
    submit: bool,
    delay_s: float = 1.0,
) -> None:
    if delay_s > 0:
        await asyncio.sleep(delay_s)

    tmux_name = tmux_session_name_for_task(task_id=task.id)
    if not (has_tmux() and _tmux_has_session(name=tmux_name)):
        return

    await _wait_for_agent_input_ready(name=tmux_name, timeout_s=3.0)

    _tmux_send_lines(
        name=tmux_name,
        lines=_agent_prelude_lines(
            task=task,
            epic=epic,
            worktree_path=worktree_path,
            template=template,
        ),
    )
    if submit:
        _tmux_send_enter(name=tmux_name)


async def start_task_agent(
    ctx: RepoContext,
    *,
    task_id: int,
    harness_command: str,
    detach: bool,
    prelude_override: str | None = None,
    agent_kind_selection: AgentKindSelection = AgentKindSelection.Auto,
    external_session_ref_hint: dict[str, Any] | None = None,
) -> StartAgentResult:
    argv = _parse_harness_command(harness_command)
    if not detach:
        raise RuntimeError("v0 requires tmux-backed detached agents (omit --no-detach)")
    if not has_tmux():
        raise RuntimeError(
            "tmux is required for v0 agents; install tmux or set up a tmux-capable runner"
        )

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            task, epic = await _load_task_and_epic(session, task_id=task_id)
            host = await ensure_host_row(session, ctx)
            agent_label = agent_label_for_task_id(task.id)

            tmux_name = tmux_session_name_for_task(task_id=task.id)
            if _tmux_has_session(name=tmux_name):
                now = datetime.now(UTC)
                agent_session = await ensure_active_task_agent_session_row(
                    session, task_id=task.id, now=now
                )
                agent_session.status = AgentStatus.Running
                agent_session.host_id = host.id
                if task.state != TaskState.Done:
                    task.state = TaskState.InProgress

                log_path = agent_session_log_path(
                    ctx, task_id=task.id, session_id=agent_session.id
                )
                try:
                    _tmux_pipe_to_log(name=tmux_name, log_path=log_path)
                except RuntimeError:
                    pass

                attach = AttachTmux(
                    session=tmux_name,
                    socket_path=None,
                    log_path=str(log_path),
                )
                agent_session.attach = attach.model_dump(mode="python")
                await session.commit()
                await session.refresh(agent_session)
                await maybe_push_task_state_to_linear(
                    ctx,
                    sessionmaker=sessionmaker,
                    task_id=task.id,
                    desired_task_state=TaskState.InProgress,
                )
                return StartAgentResult(
                    agent_label=agent_label,
                    agent_session=agent_session,
                    attach=attach,
                    started=False,
                    warnings=(),
                )

            warnings: list[str] = []
            worktree_path = await ensure_task_worktree(
                session, ctx, task=task, epic=epic
            )

            definition = LaunchConfigurationDefinition(argv=argv)
            profile_id = launch_configuration_id_for_definition(argv[0], definition)
            profile = await ensure_launch_configuration_row(
                session,
                profile_id=profile_id,
                kind=argv[0],
                definition=definition,
            )

            resolved_agent_kind = resolve_agent_kind(
                agent_kind_selection,
                argv,
                external_session_ref_hint=external_session_ref_hint,
            )

            parent_task = (
                await session.get(Task, task.parent_task_id)
                if task.parent_task_id is not None
                else None
            )
            base_ref = epic.root_branch
            base_task = parent_task
            while base_task is not None and base_task.branch_name is not None:
                if git_is_ancestor(
                    ctx.repo_root, base_task.branch_name, epic.root_branch
                ):
                    base_task = (
                        await session.get(Task, base_task.parent_task_id)
                        if base_task.parent_task_id is not None
                        else None
                    )
                    continue

                if base_task.state == TaskState.Done:
                    warnings.append(
                        f"Parent task {base_task.id} is marked done but {epic.root_branch!r} "
                        f"does not contain {base_task.branch_name!r}; fast-forward the base branch."
                    )
                    base_task = (
                        await session.get(Task, base_task.parent_task_id)
                        if base_task.parent_task_id is not None
                        else None
                    )
                    continue

                base_ref = base_task.branch_name
                break

            if not git_is_ancestor(ctx.repo_root, base_ref, task.branch_name or ""):
                if base_ref == epic.root_branch:
                    warnings.append(
                        f"Branch {task.branch_name!r} does not include the latest base tip "
                        f"{base_ref!r}; consider rebasing before starting new work."
                    )
                else:
                    warnings.append(
                        f"Branch {task.branch_name!r} does not include the latest parent tip "
                        f"{base_ref!r}; consider rebasing before starting new work."
                    )
            shim_path = shutil.which("rn")
            runtime_env = {
                "REDESMYN_AGENT_SESSION_ID": "",
                "REDESMYN_TASK_ID": str(task.id),
                "REDESMYN_HOST_KEY": host.host_key,
                "PATH": os.environ.get("PATH", ""),
            }
            if shim_path is not None:
                shim = create_git_shim(ctx=ctx, task_id=task.id)
                runtime_env["PATH"] = f"{shim.dir}{os.pathsep}{runtime_env['PATH']}"
            else:
                warnings.append(
                    "`rn` not found on PATH; skipping git shim injection (agent git will bypass blocks)"
                )
            runtime_env |= definition.env

            defaults = None
            try:
                defaults = load_orchestration_defaults(ctx)
            except RuntimeError:
                defaults = None

            sandbox_policy = NullSandboxPolicy()
            if defaults is not None and defaults.sandbox.type == "worktree":
                shared_state_paths: list[Path] = [ctx.state_dir]
                # Git worktrees share metadata under the main repo `.git/`
                # directory (e.g., `.git/worktrees/<name>/index.lock`). Allowing
                # writes there enables basic git operations within a sandboxed
                # worktree without broadening permissions elsewhere.
                repo_git_dir = ctx.repo_root / ".git"
                if repo_git_dir.exists():
                    shared_state_paths.append(repo_git_dir)
                # Many tools expect to write to a temp directory (on macOS this is
                # typically under `/var/folders/...`). Allowing the per-user temp
                # directory keeps the worktree sandbox usable without widening it
                # to all writable paths.
                tmp_root = Path(tempfile.gettempdir())
                if tmp_root.is_dir():
                    shared_state_paths.append(tmp_root)

                sandbox_policy = WorktreeSandboxPolicy(
                    worktree_path=worktree_path,
                    shared_state_paths=shared_state_paths,
                    network=defaults.sandbox.network,
                )

            if isinstance(sandbox_policy, WorktreeSandboxPolicy):
                run_dir = task_agent_runtime_dir(ctx, task_id=task.id)
                tmp_dir = run_dir / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                runtime_env["TMPDIR"] = str(tmp_dir)
                runtime_env["TMP"] = str(tmp_dir)
                runtime_env["TEMP"] = str(tmp_dir)
                runtime_env["PATH"] = _sanitize_path_for_sandbox(
                    runtime_env.get("PATH", "")
                )

                # Keep harness state (caches/config) inside Redesmyn state so it
                # remains writable when sandboxing is enabled.
                runtime_env["HOME"] = str(run_dir / "home")
                runtime_env.setdefault(
                    "CODEX_HOME", str(Path(runtime_env["HOME"]) / ".codex")
                )
                runtime_env.setdefault("XDG_CONFIG_HOME", str(run_dir / "xdg-config"))
                runtime_env.setdefault("XDG_CACHE_HOME", str(run_dir / "xdg-cache"))
                runtime_env.setdefault("XDG_DATA_HOME", str(run_dir / "xdg-data"))
                runtime_env.setdefault("XDG_STATE_HOME", str(run_dir / "xdg-state"))

                for key in (
                    "HOME",
                    "CODEX_HOME",
                    "XDG_CONFIG_HOME",
                    "XDG_CACHE_HOME",
                    "XDG_DATA_HOME",
                    "XDG_STATE_HOME",
                ):
                    value = runtime_env.get(key)
                    if not value:
                        continue
                    path = Path(value)
                    if path.is_relative_to(ctx.state_dir):
                        path.mkdir(parents=True, exist_ok=True)

                if Path(definition.argv[0]).name == "codex":
                    src_codex = Path.home() / ".codex"
                    _seed_codex_home(
                        src_dir=src_codex,
                        dst_dir=Path(runtime_env["CODEX_HOME"]),
                        warnings=warnings,
                    )

            sandbox_provider = make_sandbox_provider()
            now = datetime.now(UTC)
            agent_session = AgentSession(
                task_id=task.id,
                status=AgentStatus.Running,
                agent_kind_selection=agent_kind_selection,
                agent_kind=resolved_agent_kind,
                host_id=host.id,
                launch_configuration_id=profile.id,
                cwd_path=str(worktree_path),
                pid=None,
                resolved_launch_configuration=definition.model_dump(mode="python"),
                exit_code=None,
                started_at=now,
                ended_at=None,
            )
            session.add(agent_session)
            await session.flush()
            runtime_env["REDESMYN_AGENT_SESSION_ID"] = str(agent_session.id)

            wrapped = sandbox_provider.wrap(
                argv=definition.argv,
                cwd=worktree_path,
                env=runtime_env,
                policy=sandbox_policy,
            )

            attach = await start_tmux_session(
                ctx,
                task_id=task.id,
                session_id=agent_session.id,
                worktree_path=worktree_path,
                argv=wrapped.argv,
                env=wrapped.env,
            )
            agent_session.attach = attach.model_dump(mode="python")

            await asyncio.sleep(0.25)
            if not _tmux_has_session(name=tmux_name):
                now = datetime.now(UTC)
                agent_session.status = AgentStatus.Error
                agent_session.ended_at = now
                log_path = agent_session_log_path(
                    ctx, task_id=task.id, session_id=agent_session.id
                )
                tail = _tail_text(log_path)
                if tail:
                    warnings.append(
                        "Harness exited immediately:\n" + tail.strip("\n")[-4000:]
                    )
                else:
                    warnings.append("Harness exited immediately (no logs captured).")
                await session.commit()
                await session.refresh(agent_session)
                return StartAgentResult(
                    agent_label=agent_label,
                    agent_session=agent_session,
                    attach=attach,
                    started=False,
                    warnings=tuple(warnings),
                )

            now = datetime.now(UTC)
            agent_session.status = AgentStatus.Running
            if task.state != TaskState.Done:
                task.state = TaskState.InProgress

            await session.commit()
            await session.refresh(agent_session)
            try:
                prelude = None
                send_prelude = True
                submit_prelude = True
                try:
                    if defaults is None:
                        defaults = load_orchestration_defaults(ctx)
                    prelude = prelude_override or defaults.harness.prelude
                    send_prelude = defaults.harness.send_prelude
                    submit_prelude = defaults.harness.submit_prelude
                except RuntimeError:
                    prelude = prelude_override or None
                if send_prelude:
                    prelude_lines = _agent_prelude_lines(
                        task=task,
                        epic=epic,
                        worktree_path=worktree_path,
                        template=prelude,
                    )
                    agent_session.prelude_rendered = "\n".join(prelude_lines)
                    await session.commit()
                    await send_agent_prelude(
                        task=task,
                        epic=epic,
                        worktree_path=worktree_path,
                        template=prelude,
                        submit=submit_prelude,
                    )
            except RuntimeError as e:
                warnings.append(f"Failed to send agent prelude: {e}")
            await maybe_push_task_state_to_linear(
                ctx,
                sessionmaker=sessionmaker,
                task_id=task.id,
                desired_task_state=TaskState.InProgress,
            )
            return StartAgentResult(
                agent_label=agent_label,
                agent_session=agent_session,
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
            task, _ = await _load_task_and_epic(session, task_id=task_id)
            agent_session = await load_latest_task_agent_session_row(session, task=task)

            tmux_name = tmux_session_name_for_task(task_id=task.id)
            was_running = has_tmux() and _tmux_has_session(name=tmux_name)
            if was_running:
                _tmux_kill_session(name=tmux_name)
            else:
                if (
                    agent_session is not None
                    and agent_session.status
                    in {AgentStatus.Running, AgentStatus.Blocked}
                    and agent_session.ended_at is None
                ):
                    now = datetime.now(UTC)
                    agent_session.status = AgentStatus.Error
                    agent_session.ended_at = now
                    agent_session.pid = None
                    await session.commit()
                return False

            now = datetime.now(UTC)
            current = await load_active_task_agent_session_row(session, task_id=task.id)
            if current is None:
                current = agent_session
            if current is None:
                current = AgentSession(
                    task_id=task.id,
                    status=AgentStatus.Stopped,
                    started_at=now,
                    ended_at=now,
                    pid=None,
                )
                session.add(current)
            else:
                current.status = AgentStatus.Stopped
                current.ended_at = now
                current.pid = None

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
    prelude_override: str | None = None,
    agent_kind_selection_override: AgentKindSelection | None = None,
    default_agent_kind_selection: AgentKindSelection = AgentKindSelection.Auto,
) -> StartAgentResult:
    external_session_ref_hint: dict[str, Any] | None = None
    agent_kind_selection = agent_kind_selection_override
    if harness_command is None or agent_kind_selection is None:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                task, _ = await _load_task_and_epic(session, task_id=task_id)
                latest = await load_latest_task_agent_session_row(session, task=task)
                resolved_launch_configuration: dict[str, Any] | None = (
                    None if latest is None else latest.resolved_launch_configuration
                )
                if latest is not None:
                    external_session_ref_hint = latest.external_session_ref
                    if agent_kind_selection is None:
                        agent_kind_selection = latest.agent_kind_selection
                if harness_command is None:
                    if resolved_launch_configuration is None:
                        raise RuntimeError(
                            "No prior agent config found; pass --harness to restart"
                        )
                    definition = TypeAdapter(
                        LaunchConfigurationDefinition
                    ).validate_python(resolved_launch_configuration)
                    harness_command = shlex.join(definition.argv)
        finally:
            await engine.dispose()
    if agent_kind_selection is None:
        agent_kind_selection = default_agent_kind_selection

    if not detach:
        raise RuntimeError("v0 requires tmux-backed detached agents (omit --no-detach)")
    if not has_tmux():
        raise RuntimeError(
            "tmux is required for v0 agents; install tmux or set up a tmux-capable runner"
        )
    await stop_task_agent(ctx, task_id=task_id)
    if harness_command is None:
        raise RuntimeError("No harness command available for restart")
    return await start_task_agent(
        ctx,
        task_id=task_id,
        harness_command=harness_command,
        detach=detach,
        prelude_override=prelude_override,
        agent_kind_selection=agent_kind_selection,
        external_session_ref_hint=external_session_ref_hint,
    )


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
            task, _ = await _load_task_and_epic(session, task_id=task_id)
            agent_session = await load_latest_task_agent_session_row(session, task=task)
            if active_only:
                tmux_name = tmux_session_name_for_task(task_id=task.id)
                if has_tmux() and _tmux_has_session(name=tmux_name):
                    if agent_session is None or agent_session.ended_at is not None:
                        now = datetime.now(UTC)
                        agent_session = await ensure_active_task_agent_session_row(
                            session, task_id=task.id, now=now
                        )
                        agent_session.status = AgentStatus.Running
                        log_path = agent_session_log_path(
                            ctx, task_id=task.id, session_id=agent_session.id
                        )
                        agent_session.attach = AttachTmux(
                            session=tmux_name,
                            socket_path=None,
                            log_path=str(log_path),
                        ).model_dump(mode="python")
                        await session.commit()
                        await session.refresh(agent_session)
                else:
                    return None
            return agent_session
    finally:
        await engine.dispose()


def agent_log_path_for_session_row(
    ctx: RepoContext, *, agent_session_row: AgentSession
) -> Path:
    attach = TypeAdapter(AttachInfo).validate_python(agent_session_row.attach)
    if isinstance(attach, (AttachExternal, AttachTmux)) and attach.log_path:
        return Path(attach.log_path)
    return agent_session_log_path(
        ctx, task_id=agent_session_row.task_id, session_id=agent_session_row.id
    )


def attach_agent_session(*, agent_session_row: AgentSession) -> int:
    attach = TypeAdapter(AttachInfo).validate_python(agent_session_row.attach)
    if isinstance(attach, AttachTmux):
        return _tmux_attach(name=attach.session)
    raise RuntimeError("Attach is not available for this agent")
