from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

import click
import typer
from sqlalchemy import delete, desc, select

from redesmyn import __version__
from redesmyn.agent_runtime import (
    attach_agent_session,
    agent_log_path_for_session_row,
    checkout_task_worktree,
    find_existing_worktree_path_for_branch,
    has_tmux,
    load_task_agent_session,
    restart_task_agent,
    start_task_agent,
    stop_task_agent,
    tmux_session_name_for_task,
)
from redesmyn.blocks import (
    NotInitializedError,
    clear_block,
    get_effective_block,
    list_blocks,
    set_manual_block,
)
from redesmyn.context import RepoContext, get_repo_context
from redesmyn.db import (
    DatabaseMigrationRequiredError,
    DatabaseNotInitializedError,
    Epic,
    LinearAuth,
    Repository,
    Task,
    create_engine,
    create_sessionmaker,
    init_db,
)
from redesmyn.docs.markdown import (
    MarkdownSectionError,
    extract_fenced_block_after_heading,
    parse_yaml_block,
)
from redesmyn.docs.loader import DocLoadError, load_epic_doc, load_task_doc
from redesmyn.docs.metadata import TaskMetadata
from redesmyn.docs.writer import upsert_metadata_yaml, upsert_synced_section
from redesmyn.domain.enums import (
    AgentKindSelection,
    BlockPolicy,
    MergeRunStatus,
    TaskAuthority,
    TaskSource,
    TaskState,
)
from redesmyn.integrations.linear import (
    LinearApiError,
    LinearClient,
    LinearIssue,
    LinearIssueRelation,
    LinearMilestone,
    LinearProject,
    create_issue,
    ensure_issue_has_label,
    exchange_code_for_token,
    fetch_label_by_name,
    fetch_issue,
    fetch_issue_project_milestone_id,
    fetch_issue_team_id,
    fetch_project,
    fetch_project_issue_relations,
    fetch_project_issues,
    fetch_project_issues_by_label,
    fetch_project_issues_by_milestone,
    fetch_project_milestones,
    fetch_projects,
    linear_authorize_url,
    linear_redirect_uri,
    new_oauth_state,
    new_pkce_verifier,
    pkce_code_challenge,
    refresh_access_token,
    resolve_team_state_id,
    set_issue_blockers,
    update_issue,
)
from redesmyn.integrations.linear_credentials import (
    LinearCredentials,
    default_linear_credential_store,
    is_expiring_soon,
)
from redesmyn.integrations.linear_state import (
    linear_state_type_from_task_state,
    task_state_from_linear_state_type,
)
from redesmyn.integrations.linear_write_defaults import (
    ensure_linear_write_defaults,
    load_linear_write_defaults,
)
from redesmyn.git_proxy import (
    READ_ONLY_SUBCOMMANDS,
    detect_git_subcommand,
    does_block_git,
)
from redesmyn.git_telemetry import update_git_projections
from redesmyn.logging_config import configure_logging
from redesmyn.orchestration_config import (
    global_config_path,
    load_orchestration_defaults,
    read_config_file,
    render_config_toml,
    repo_config_path,
    set_config_value,
    write_config,
)
from redesmyn.orchestrator import init_repo
from redesmyn.repo_identity import compute_repo_id
from redesmyn.repo import (
    GitCommandError,
    NotAGitRepositoryError,
    branch_exists,
    current_branch,
    git_rename_current_branch,
    git_worktree_add,
)
from redesmyn.git_mechanics_v0 import (
    MergeBlockedByRunningAgents,
    MergePlanError,
    build_restack_plan,
    build_merge_cascade_plan,
    execute_restack_plan,
    execute_merge_cascade_plan,
    format_merge_plan,
    format_restack_plan,
    format_running_agents_warning,
)
from redesmyn.host_identity import load_or_create_host_identity
from redesmyn.merge_runs import (
    emit_merge_run_event,
    emit_task_merge_event,
    merge_run_plan_snapshot,
    record_merge_run_step_update,
    set_merge_run_status,
    upsert_merge_run,
)
from redesmyn.repo_observer import run_repo_observer
from redesmyn.sandbox import make_sandbox_provider
from redesmyn.settings import RedesmynSettings
from redesmyn.strings import slugify

if TYPE_CHECKING:
    from redesmyn.docs.loader import TaskDoc

_DEBUG = False

app = typer.Typer(
    add_completion=False,
    help="Redesmyn CLI (`rn`).",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)
daemon_app = typer.Typer(add_completion=False, help="Daemon management.")
server_app = typer.Typer(add_completion=False, help="Control plane server management.")
config_app = typer.Typer(add_completion=False, help="Defaults and settings.")
sandbox_app = typer.Typer(add_completion=False, help="Sandbox configuration + health.")
block_app = typer.Typer(
    add_completion=False,
    help="Block controls.",
)
epic_app = typer.Typer(add_completion=False, help="Epic management.")
task_app = typer.Typer(add_completion=False, help="Task management.")
agent_app = typer.Typer(add_completion=False, help="Agent management.")
linear_app = typer.Typer(add_completion=False, help="Linear integration.")
observer_app = typer.Typer(add_completion=False, help="Repo observer + telemetry.")


@app.callback()
def _global_options(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show full traceback on errors.",
        is_eager=True,
    ),
) -> None:
    global _DEBUG
    _DEBUG = debug


@dataclass(slots=True)
class SyncStats:
    epics_created: int = 0
    epics_updated: int = 0
    tasks_created: int = 0
    tasks_updated: int = 0
    branches_created: int = 0
    branches_updated: int = 0


@dataclass(slots=True)
class LinearPushStats:
    issues_created: int = 0
    issues_updated: int = 0
    docs_updated: int = 0
    blockers_updated: int = 0
    blockers_skipped: int = 0


class SyncOutputFormat(StrEnum):
    Text = "text"
    Json = "json"


@app.command()
def sync(
    from_: str | None = typer.Option(
        None,
        "--from",
        help="Sync source (local|linear).",
        show_choices=True,
        case_sensitive=False,
    ),
    to: str | None = typer.Option(
        None,
        "--to",
        help="Sync target (linear).",
        show_choices=True,
        case_sensitive=False,
    ),
    epic: str | None = typer.Option(
        None, help="Epic slug or id (defaults if only one epic)."
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Linear project id, slug, or name (when syncing from/to Linear).",
    ),
    create_branches: bool = typer.Option(
        True,
        "--create-branches/--no-create-branches",
        help="Create/update the branch graph for tasks.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan changes without writing to Linear."
    ),
    output_format: SyncOutputFormat = typer.Option(
        SyncOutputFormat.Text,
        "--format",
        case_sensitive=False,
        show_choices=True,
        help="Output format for --dry-run (text|json).",
    ),
) -> None:
    """Synchronize between local task docs, Linear, and the DB projection."""
    if from_ and to:
        typer.echo("error: pass only one of --from or --to", err=True)
        raise typer.Exit(2)
    if not from_ and not to:
        typer.echo(
            "error: missing direction; pass --from local|linear or --to linear",
            err=True,
        )
        raise typer.Exit(2)

    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if from_:
        if dry_run:
            typer.echo(
                "error: --dry-run is only supported with --to linear (v0)", err=True
            )
            raise typer.Exit(2)
        if output_format != SyncOutputFormat.Text:
            typer.echo(
                "error: --format is only supported with --dry-run (v0)", err=True
            )
            raise typer.Exit(2)
        src = from_.lower()
        if src == "local":
            stats = asyncio.run(
                _sync_from_local(ctx, epic=epic, create_branches=create_branches)
            )
            typer.echo(
                "Synced from local: "
                f"epics +{stats.epics_created}/~{stats.epics_updated}, "
                f"tasks +{stats.tasks_created}/~{stats.tasks_updated}, "
                f"branches +{stats.branches_created}/~{stats.branches_updated}"
            )
            return
        if src == "linear":
            stats = asyncio.run(
                _sync_from_linear(
                    ctx,
                    epic=epic,
                    project=project,
                    create_branches=create_branches,
                )
            )
            typer.echo(
                "Synced from Linear: "
                f"epics +{stats.epics_created}/~{stats.epics_updated}, "
                f"tasks +{stats.tasks_created}/~{stats.tasks_updated}, "
                f"branches +{stats.branches_created}/~{stats.branches_updated}"
            )
            return
        typer.echo("error: --from must be one of: local, linear", err=True)
        raise typer.Exit(2)

    dst = (to or "").lower()
    if dst == "linear":
        if not dry_run and output_format != SyncOutputFormat.Text:
            typer.echo(
                "error: --format is only supported with --dry-run (v0)", err=True
            )
            raise typer.Exit(2)
        stats = asyncio.run(
            _sync_to_linear(
                ctx,
                epic=epic,
                project=project,
                create_branches=create_branches,
                dry_run=dry_run,
                output_format=output_format,
            )
        )
        if not dry_run:
            typer.echo(
                "Synced to Linear: "
                f"issues +{stats.issues_created}/~{stats.issues_updated}, "
                f"docs ~{stats.docs_updated}, "
                f"blockers ~{stats.blockers_updated} (skipped {stats.blockers_skipped})"
            )
        return
    typer.echo("error: --to must be one of: linear", err=True)
    raise typer.Exit(2)


@app.command()
def version() -> None:
    print(__version__)


@app.command()
def init(
    cwd: Path | None = typer.Option(None, help="Run from this directory."),
) -> None:
    """Initialize Redesmyn state for this repo."""
    try:
        ctx = get_repo_context(cwd=cwd)
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    migrate = not ctx.db_path.exists()
    asyncio.run(init_repo(ctx, migrate=migrate))
    typer.echo(f"Initialized: {ctx.state_dir}")
    typer.echo(f"DB: {ctx.db_path}")


def _parse_epic_task_num(raw: str) -> tuple[int, str]:
    trimmed = raw.strip()
    match = re.match(r"^(?:[Tt]-?)?(\d+)$", trimmed)
    if not match:
        raise ValueError(
            f"Invalid task number: {raw!r} (expected a number like 4 or T-4)"
        )
    num = int(match.group(1))
    if num <= 0:
        raise ValueError(f"Invalid task number: {raw!r} (must be > 0)")
    return num, f"T-{num}"


def _infer_task_label_from_local_path(local_path: str | None) -> str | None:
    if not local_path:
        return None
    match = re.search(r"/tasks/(T-\d+)/README\.md$", local_path)
    return match.group(1) if match else None


async def _resolve_task_for_shell(
    ctx: RepoContext,
    *,
    epic: str | None,
    task: str | None,
    task_id: int | None,
) -> tuple[Task, Epic, str | None]:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            if task_id is not None:
                row = await session.get(Task, task_id)
                if row is None:
                    raise RuntimeError(f"Unknown task id: {task_id}")
                epic_row = await session.get(Epic, row.epic_id)
                if epic_row is None:
                    raise RuntimeError("Epic not found for task")
                return row, epic_row, _infer_task_label_from_local_path(row.local_path)

            if not task:
                raise RuntimeError("Pass either --task-id or -t/--task.")

            epic_row = await _resolve_epic(ctx, epic=epic)
            _, task_label = _parse_epic_task_num(task)
            expected_rel_path = f"epics/{epic_row.slug}/tasks/{task_label}/README.md"

            row = await session.scalar(
                select(Task).where(
                    Task.epic_id == epic_row.id,
                    Task.local_path == expected_rel_path,
                )
            )
            if row is not None:
                return row, epic_row, task_label

            # Fallback: locate by reading docs metadata, in case the task folder isn't `T-<n>`.
            matches: list[Task] = []
            tasks = list(
                await session.scalars(select(Task).where(Task.epic_id == epic_row.id))
            )
            for candidate in tasks:
                if not candidate.local_path:
                    continue
                path = ctx.worktree_root / candidate.local_path
                if not path.exists():
                    continue
                try:
                    doc = load_task_doc(path)
                except DocLoadError:
                    continue
                if doc.metadata.id == task_label:
                    matches.append(candidate)

            if not matches:
                raise RuntimeError(
                    f"Unknown task {task_label} in epic {epic_row.slug!r}. "
                    "Run `rn sync --from local` to import docs."
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Ambiguous task ref {task_label!r}; matches multiple tasks."
                )
            return matches[0], epic_row, task_label
    finally:
        await engine.dispose()


@app.command()
def shell(
    epic: str | None = typer.Option(
        None,
        "--epic",
        "-e",
        help="Epic slug or id (defaults if only one epic).",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Epic-scoped task number (e.g. 4 or T-4).",
    ),
    task_id: int | None = typer.Option(
        None,
        "--task-id",
        help="Developer: task id (DB primary key).",
    ),
    no_create: bool = typer.Option(
        False,
        "--no-create",
        help="Do not create a worktree; error if it doesn't exist.",
    ),
    print_only: bool = typer.Option(
        False,
        "--print",
        help="Print the worktree path and exit.",
    ),
    nested: bool = typer.Option(
        False,
        "--nested",
        help="Allow opening an `rn shell` subshell from within another one.",
    ),
) -> None:
    """Open a subshell rooted at a task's worktree."""
    if task_id is not None and (epic is not None or task is not None):
        typer.echo("error: pass only one of --task-id or --epic/--task", err=True)
        raise typer.Exit(2)

    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    try:
        resolved_task, resolved_epic, task_label = asyncio.run(
            _resolve_task_for_shell(ctx, epic=epic, task=task, task_id=task_id)
        )
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if resolved_task.branch_name is None:
        typer.echo(
            "error: task has no branch backing; run `rn sync --from local` to create branches",
            err=True,
        )
        raise typer.Exit(2)

    path: Path | None = None
    if resolved_task.worktree_path:
        candidate = Path(resolved_task.worktree_path)
        if candidate.exists():
            path = candidate

    if path is None:
        existing = find_existing_worktree_path_for_branch(
            ctx.repo_root, branch=resolved_task.branch_name
        )
        if existing is not None and existing.exists():
            path = existing

    if path is None and no_create:
        typer.echo(
            "error: worktree does not exist (pass without --no-create)", err=True
        )
        raise typer.Exit(2)

    if path is None:
        try:
            path = asyncio.run(checkout_task_worktree(ctx, task_id=resolved_task.id))
        except RuntimeError as e:
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(2)

    if print_only:
        typer.echo(str(path))
        return

    parent_cwd = os.environ.get("RN_PARENT_CWD")
    if parent_cwd and not nested:
        typer.echo(
            "error: already in an `rn shell` subshell; run `exit` first (or pass --nested)",
            err=True,
        )
        raise typer.Exit(2)

    safe_title = (resolved_task.title or "").replace("\n", " ").strip()
    label = task_label or _infer_task_label_from_local_path(resolved_task.local_path)
    num = None
    if label:
        parsed = re.match(r"^T-(\d+)$", label)
        if parsed:
            num = parsed.group(1)
    suffix = f" {label}" if label else ""
    typer.echo(f'Worktree: {resolved_epic.slug}{suffix} "{safe_title}" → {path}')

    env = os.environ.copy()
    env["RN_EPIC_SLUG"] = resolved_epic.slug
    env["RN_TASK_ID"] = str(resolved_task.id)
    env["RN_TASK_TITLE"] = safe_title
    env["RN_BRANCH_NAME"] = resolved_task.branch_name
    env["RN_WORKTREE_PATH"] = str(path)
    env["RN_PARENT_CWD"] = os.getcwd()
    if label:
        env["RN_TASK_LABEL"] = label
    if num:
        env["RN_TASK_NUM"] = num

    shell_path = env.get("SHELL") or "/bin/zsh"
    try:
        subprocess.run([shell_path], cwd=str(path), env=env, check=False)
    except OSError as e:
        typer.echo(f"error: failed to start shell {shell_path!r}: {e}", err=True)
        raise typer.Exit(2)


async def _resolve_task_id_for_current_context(
    *,
    sessionmaker: object,
    explicit_task_id: int | None,
) -> int:
    if explicit_task_id is not None:
        return explicit_task_id

    if raw := os.environ.get("RN_TASK_ID"):
        try:
            return int(raw)
        except ValueError as e:
            raise MergePlanError(
                f"Invalid RN_TASK_ID={raw!r} (expected an integer)"
            ) from e

    branch = current_branch(cwd=Path.cwd())
    if branch == "HEAD":
        raise MergePlanError("Pass --task (or run from rn shell / a task worktree).")

    async with sessionmaker() as session:  # type: ignore[misc]
        matches = list(
            await session.scalars(select(Task).where(Task.branch_name == branch))
        )

    if not matches:
        raise MergePlanError(
            f"No task matches current branch {branch!r}. "
            "Pass --task (or run from rn shell / a task worktree)."
        )
    if len(matches) > 1:
        ids = ", ".join(str(t.id) for t in matches)
        raise MergePlanError(
            f"Ambiguous branch {branch!r}; assigned to multiple tasks ({ids}). "
            "Pass --task to disambiguate."
        )
    return matches[0].id


@app.command()
def merge(
    task_id: int | None = typer.Option(
        None,
        "--task",
        help=(
            "Task id (DB primary key). Defaults from RN_TASK_ID (rn shell) or the "
            "current git branch (when run from a task worktree)."
        ),
    ),
    cascade: bool = typer.Option(
        False,
        "--cascade",
        help="Also rebase downstream branches to preserve the stack.",
    ),
    scope: str = typer.Option(
        "descendants",
        "--scope",
        help="Cascade scope (descendants|spine). Ignored unless --cascade is set.",
        show_choices=True,
    ),
    restack_mode: str = typer.Option(
        "strict",
        "--restack-mode",
        help="Restack ordering (strict|merge_then_restack). Ignored unless --cascade is set.",
        show_choices=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the merge plan without performing any git operations.",
    ),
    yes: bool = typer.Option(
        False,
        "-y",
        "--yes",
        help="Proceed without prompting (including when running agents are detected).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Merge even if the task is not marked ready.",
    ),
) -> None:
    """Merge a task branch into the epic base branch (optionally cascading through the stack)."""
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    normalized_scope = scope.strip().lower()
    if normalized_scope not in {"descendants", "spine"}:
        typer.echo("error: --scope must be one of: descendants, spine", err=True)
        raise typer.Exit(2)

    normalized_restack_mode = restack_mode.strip().lower()
    if normalized_restack_mode not in {"strict", "merge_then_restack"}:
        typer.echo(
            "error: --restack-mode must be one of: strict, merge_then_restack",
            err=True,
        )
        raise typer.Exit(2)

    selected_scope = normalized_scope if cascade else "spine"
    selected_restack_mode = normalized_restack_mode if cascade else "strict"

    engine = create_engine(ctx.db_path)
    sessionmaker = create_sessionmaker(engine)
    try:
        host_key = load_or_create_host_identity(ctx).host_key
        resolved_task_id = asyncio.run(
            _resolve_task_id_for_current_context(
                sessionmaker=sessionmaker,
                explicit_task_id=task_id,
            )
        )
        run_id = f"cli-{uuid4().hex}"
        plan = asyncio.run(
            build_merge_cascade_plan(
                ctx=ctx,
                sessionmaker=sessionmaker,
                task_id=resolved_task_id,
                run_id=run_id,
                scope=selected_scope,  # type: ignore[arg-type]
                restack_mode=selected_restack_mode,  # type: ignore[arg-type]
                force=force,
            )
        )

        typer.echo("Merge plan:")
        for line in format_merge_plan(plan).splitlines():
            typer.echo(f"  {line}" if line else "")
        if dry_run:
            return

        allow_running = yes
        if not yes:
            if plan.running_agents:
                typer.echo("")
                typer.echo(format_running_agents_warning(plan.running_agents))
                typer.echo("")
            if not typer.confirm("Proceed with merge?", default=False):
                raise typer.Exit(1)
            allow_running = bool(plan.running_agents)

        plan_snapshot = merge_run_plan_snapshot(plan=plan, operation="merge")
        merge_run_task_id = (
            plan.spine_task_ids[-1] if plan.spine_task_ids else resolved_task_id
        )
        asyncio.run(
            upsert_merge_run(
                sessionmaker=sessionmaker,
                run_id=plan.run_id,
                epic_id=plan.epic_id,
                task_id=resolved_task_id,
                host_key=host_key,
                canonical=True,
                scope=plan.scope,
                allow_running=allow_running,
                force=force,
                plan_snapshot=plan_snapshot,
            )
        )
        asyncio.run(
            emit_merge_run_event(
                sessionmaker=sessionmaker,
                run_id=plan.run_id,
                task_id=merge_run_task_id,
                epic_id=plan.epic_id,
                requested_task_id=resolved_task_id,
                status=MergeRunStatus.Running,
                host_key=host_key,
                operation="merge",
            )
        )

        try:
            asyncio.run(
                execute_merge_cascade_plan(
                    ctx=ctx,
                    sessionmaker=sessionmaker,
                    plan=plan,
                    allow_running=allow_running,
                    emit_event=lambda payload: emit_task_merge_event(
                        sessionmaker=sessionmaker,
                        payload={**payload, "operation": "merge"},
                        host_key=host_key,
                    ),
                    update_run=lambda update: record_merge_run_step_update(
                        sessionmaker=sessionmaker,
                        run_id=plan.run_id,
                        update=update,
                    ),
                )
            )
        except GitCommandError:
            # `record_merge_run_step_update` already recorded blocked/failed status.
            raise
        except Exception as e:
            asyncio.run(
                set_merge_run_status(
                    sessionmaker=sessionmaker,
                    run_id=plan.run_id,
                    status=MergeRunStatus.Failed,
                    error=str(e),
                )
            )
            raise
        else:
            asyncio.run(
                set_merge_run_status(
                    sessionmaker=sessionmaker,
                    run_id=plan.run_id,
                    status=MergeRunStatus.Succeeded,
                )
            )

        typer.echo(f"Merged into {plan.base_branch}")
    except MergePlanError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except MergeBlockedByRunningAgents:
        typer.echo(
            "error: merge affects running tasks; pass --yes to proceed", err=True
        )
        raise typer.Exit(2)
    except GitCommandError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    finally:
        asyncio.run(engine.dispose())


@app.command()
def restack(
    task_id: int | None = typer.Option(
        None,
        "--task",
        help=(
            "Task id (DB primary key). Defaults from RN_TASK_ID (rn shell) or the "
            "current git branch (when run from a task worktree)."
        ),
    ),
    scope: str = typer.Option(
        "descendants",
        "--scope",
        help="Restack scope (descendants|spine).",
        show_choices=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the restack plan without performing any git operations.",
    ),
    yes: bool = typer.Option(
        False,
        "-y",
        "--yes",
        help="Proceed without prompting (including when running agents are detected).",
    ),
) -> None:
    """Rebase a task stack to preserve parent-child branch relationships."""
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    normalized_scope = scope.strip().lower()
    if normalized_scope not in {"descendants", "spine"}:
        typer.echo("error: --scope must be one of: descendants, spine", err=True)
        raise typer.Exit(2)

    engine = create_engine(ctx.db_path)
    sessionmaker = create_sessionmaker(engine)
    try:
        host_key = load_or_create_host_identity(ctx).host_key
        resolved_task_id = asyncio.run(
            _resolve_task_id_for_current_context(
                sessionmaker=sessionmaker,
                explicit_task_id=task_id,
            )
        )
        run_id = f"cli-{uuid4().hex}"
        plan = asyncio.run(
            build_restack_plan(
                ctx=ctx,
                sessionmaker=sessionmaker,
                task_id=resolved_task_id,
                run_id=run_id,
                scope=normalized_scope,  # type: ignore[arg-type]
            )
        )

        typer.echo("Restack plan:")
        for line in format_restack_plan(plan).splitlines():
            typer.echo(f"  {line}" if line else "")
        if dry_run:
            return

        allow_running = yes
        if not yes:
            if plan.running_agents:
                typer.echo("")
                typer.echo(format_running_agents_warning(plan.running_agents))
                typer.echo("")
            if not typer.confirm("Proceed with restack?", default=False):
                raise typer.Exit(1)
            allow_running = bool(plan.running_agents)

        plan_snapshot = merge_run_plan_snapshot(plan=plan, operation="restack")
        asyncio.run(
            upsert_merge_run(
                sessionmaker=sessionmaker,
                run_id=plan.run_id,
                epic_id=plan.epic_id,
                task_id=resolved_task_id,
                host_key=host_key,
                canonical=True,
                scope=plan.scope,
                allow_running=allow_running,
                force=False,
                plan_snapshot=plan_snapshot,
            )
        )
        asyncio.run(
            emit_merge_run_event(
                sessionmaker=sessionmaker,
                run_id=plan.run_id,
                task_id=resolved_task_id,
                epic_id=plan.epic_id,
                requested_task_id=resolved_task_id,
                status=MergeRunStatus.Running,
                host_key=host_key,
                operation="restack",
            )
        )

        try:
            asyncio.run(
                execute_restack_plan(
                    ctx=ctx,
                    sessionmaker=sessionmaker,
                    plan=plan,
                    allow_running=allow_running,
                    emit_event=lambda payload: emit_task_merge_event(
                        sessionmaker=sessionmaker,
                        payload={**payload, "operation": "restack"},
                        host_key=host_key,
                    ),
                    update_run=lambda update: record_merge_run_step_update(
                        sessionmaker=sessionmaker,
                        run_id=plan.run_id,
                        update=update,
                    ),
                )
            )
        except GitCommandError:
            raise
        except Exception as e:
            asyncio.run(
                set_merge_run_status(
                    sessionmaker=sessionmaker,
                    run_id=plan.run_id,
                    status=MergeRunStatus.Failed,
                    error=str(e),
                )
            )
            raise
        else:
            asyncio.run(
                set_merge_run_status(
                    sessionmaker=sessionmaker,
                    run_id=plan.run_id,
                    status=MergeRunStatus.Succeeded,
                )
            )

        typer.echo("Restacked.")
    except MergePlanError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except MergeBlockedByRunningAgents:
        typer.echo(
            "error: restack affects running tasks; pass --yes to proceed", err=True
        )
        raise typer.Exit(2)
    except GitCommandError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    finally:
        asyncio.run(engine.dispose())


@app.command()
def attach(
    task_id: int | None = typer.Option(
        None,
        "--task",
        help=(
            "Task id (DB primary key). Defaults from RN_TASK_ID (rn shell) or the "
            "current git branch (when run from a task worktree)."
        ),
    ),
) -> None:
    """Attach to the current task's detached agent session (tmux)."""
    try:
        repo_ctx = get_repo_context()
        _ensure_initialized(repo_ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    engine = create_engine(repo_ctx.db_path)
    sessionmaker = create_sessionmaker(engine)
    try:
        resolved_task_id = asyncio.run(
            _resolve_task_id_for_current_context(
                sessionmaker=sessionmaker,
                explicit_task_id=task_id,
            )
        )
    except MergePlanError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    finally:
        asyncio.run(engine.dispose())

    _attach_agent_for_task_id(repo_ctx, task_id=resolved_task_id)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise typer.BadParameter(f"Expected a boolean, got {value!r}")


@config_app.command("get")
def config_get() -> None:
    """Print the effective (global + repo) orchestration defaults."""
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    defaults = load_orchestration_defaults(ctx)
    typer.echo(f"Global: {global_config_path()}")
    typer.echo(f"Repo:   {repo_config_path(ctx)}")
    typer.echo("")
    rendered = render_config_toml(defaults.model_dump(exclude_none=True))
    typer.echo(rendered if rendered else "(empty)")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        ...,
        help="Config key (e.g. default_epic, fleet.mode, fleet.size, harness.command, harness.detach, harness.prelude, harness.send_prelude, harness.submit_prelude, sandbox.type, sandbox.network).",
    ),
    value: str = typer.Argument(
        ...,
        help="Value. Use 'null' to unset (falls back to the next layer).",
    ),
    scope: str = typer.Option(
        "repo",
        "--scope",
        help="Where to write (repo|global).",
        show_choices=True,
        case_sensitive=False,
    ),
) -> None:
    """Set a config value in the repo or global layer."""
    allowed = {
        "default_epic",
        "fleet.mode",
        "fleet.size",
        "harness.command",
        "harness.detach",
        "harness.prelude",
        "harness.send_prelude",
        "harness.submit_prelude",
        "sandbox.type",
        "sandbox.network",
    }
    if key not in allowed:
        raise typer.BadParameter(f"Unknown key: {key!r}")

    normalized_scope = scope.strip().lower()
    if normalized_scope not in {"repo", "global"}:
        raise typer.BadParameter("--scope must be one of: repo, global")

    value_raw = value.strip()
    unset = value_raw.lower() in {"null", "none", "(null)", "(none)"}

    parsed: object | None
    if unset:
        parsed = None
    elif key == "fleet.mode":
        mode = value_raw.lower()
        if mode not in {"fixed", "auto"}:
            raise typer.BadParameter("fleet.mode must be fixed or auto")
        parsed = mode
    elif key == "fleet.size":
        parsed_int = int(value_raw)
        if parsed_int <= 0:
            raise typer.BadParameter("fleet.size must be > 0 (or null)")
        parsed = parsed_int
    elif key == "sandbox.type":
        sandbox_type = value_raw.lower()
        if sandbox_type not in {"none", "worktree"}:
            raise typer.BadParameter("sandbox.type must be none or worktree")
        parsed = sandbox_type
    elif key == "sandbox.network":
        sandbox_network = value_raw.lower()
        if sandbox_network not in {"allow", "deny"}:
            raise typer.BadParameter("sandbox.network must be allow or deny")
        parsed = sandbox_network
    elif key == "harness.detach":
        parsed = _parse_bool(value_raw)
    elif key in {"harness.send_prelude", "harness.submit_prelude"}:
        parsed = _parse_bool(value_raw)
    else:
        parsed = value_raw

    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    path = repo_config_path(ctx) if normalized_scope == "repo" else global_config_path()
    data = read_config_file(path)
    set_config_value(data, key, parsed)
    write_config(path, data)
    typer.echo(f"Wrote: {path}")


@sandbox_app.command("doctor")
def sandbox_doctor(
    json_output: bool = typer.Option(False, "--json", help="Print JSON for scripting."),
) -> None:
    """Show sandbox provider capability + current defaults."""
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    defaults = load_orchestration_defaults(ctx)
    capabilities = make_sandbox_provider().capabilities()
    payload = {
        "capabilities": capabilities.model_dump(mode="python"),
        "defaults": {
            "type": defaults.sandbox.type,
            "network": defaults.sandbox.network,
        },
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo(f"Provider: {capabilities.provider}")
    available = "yes" if capabilities.available else "no"
    if capabilities.unavailable_reason:
        available = f"{available} ({capabilities.unavailable_reason})"
    typer.echo(f"Available: {available}")
    typer.echo(
        "Supports worktree sandbox: "
        + ("yes" if capabilities.supports_worktree else "no")
    )
    typer.echo(
        f"Supports network deny: {'yes' if capabilities.supports_network_deny else 'no'}"
    )
    typer.echo("")
    typer.echo(f"Configured type: {defaults.sandbox.type}")
    typer.echo(f"Configured network: {defaults.sandbox.network}")


@app.command()
def run(
    epic: str | None = typer.Option(
        None, help="Epic slug or id (defaults from config.default_epic)."
    ),
    fleet_size: int | None = typer.Option(
        None,
        "--fleet-size",
        help="Start agents for the top N eligible tasks in the epic (ordered parent-first). Overrides config.",
    ),
    task_ids: list[int] | None = typer.Option(
        None,
        "--task",
        help="Task id(s) to start (repeatable). Mutually exclusive with --fleet-size.",
    ),
    harness: str | None = typer.Option(
        None,
        "--harness",
        help="Harness command (shell-like). Defaults from config.harness.command.",
    ),
    detach: bool | None = typer.Option(
        None,
        "--detach/--no-detach",
        help="Run in a detached session (tmux if available). Defaults from config.harness.detach.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan only."),
    restart: bool = typer.Option(
        False,
        "--restart",
        help="Stop and restart agents for selected tasks (still respects eligibility).",
    ),
) -> None:
    """Fleet start agents (or start a specific set of tasks)."""
    if task_ids and fleet_size is not None:
        typer.echo("error: --task is mutually exclusive with --fleet-size", err=True)
        raise typer.Exit(2)

    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
        asyncio.run(init_repo(ctx, migrate=False))
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    defaults = load_orchestration_defaults(ctx)
    effective_harness = harness or defaults.harness.command
    if effective_harness is None:
        typer.echo(
            "error: Missing --harness (or set config.harness.command via `rn config set harness.command …`)",
            err=True,
        )
        raise typer.Exit(2)

    effective_detach = detach if detach is not None else defaults.harness.detach
    effective_epic = epic or defaults.default_epic

    async def _plan() -> tuple[
        list[Task], list[tuple[int, str]], list[Task], Epic | None
    ]:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                epic_row: Epic | None = None
                if task_ids is None:
                    epic_row = await _resolve_epic(ctx, epic=effective_epic)
                    tasks = list(
                        await session.scalars(
                            select(Task)
                            .where(Task.epic_id == epic_row.id)
                            .order_by(Task.id)
                        )
                    )
                else:
                    tasks = []
                    for task_id in task_ids:
                        task = await session.get(Task, task_id)
                        if task is None:
                            continue
                        tasks.append(task)
                task_by_id = {t.id: t for t in tasks}

                def task_depth(task_id: int, cache: dict[int, int]) -> int:
                    if task_id in cache:
                        return cache[task_id]
                    task = task_by_id.get(task_id)
                    if task is None or task.parent_task_id is None:
                        cache[task_id] = 0
                        return 0
                    value = 1 + task_depth(task.parent_task_id, cache)
                    cache[task_id] = value
                    return value

                eligible: list[Task] = []
                skipped: list[tuple[int, str]] = []
                if task_ids is not None:
                    existing_ids = {t.id for t in tasks}
                    for task_id in task_ids:
                        if task_id not in existing_ids:
                            skipped.append((task_id, "not found"))

                for task in tasks:
                    if task.branch_name is None:
                        skipped.append((task.id, "no branch backing"))
                        continue
                    if task.state in {TaskState.Blocked, TaskState.Done}:
                        skipped.append((task.id, f"state={task.state.value}"))
                        continue
                    eligible.append(task)

                active_task_ids: set[int] = set()
                if eligible:
                    tmux_sessions: set[str] = set()
                    if has_tmux():
                        proc = subprocess.run(
                            ["tmux", "list-sessions", "-F", "#S"],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=1,
                        )
                        if proc.returncode == 0:
                            tmux_sessions = {
                                line.strip()
                                for line in proc.stdout.splitlines()
                                if line.strip()
                            }

                    for task in eligible:
                        tmux_name = tmux_session_name_for_task(task_id=task.id)
                        if tmux_name in tmux_sessions:
                            active_task_ids.add(task.id)

                if epic_row is not None:
                    cache: dict[int, int] = {}
                    eligible.sort(key=lambda t: (task_depth(t.id, cache), t.id))

                if task_ids is None:
                    effective_fleet_size: int | None = fleet_size
                    if effective_fleet_size is None:
                        if defaults.fleet.mode == "auto":
                            effective_fleet_size = len(eligible)
                        else:
                            effective_fleet_size = defaults.fleet.size

                    if effective_fleet_size is None:
                        raise typer.BadParameter(
                            "Missing --fleet-size (or set config.fleet.mode/config.fleet.size)"
                        )

                    if restart:
                        selected = eligible[:effective_fleet_size]
                    else:
                        selected = [t for t in eligible if t.id not in active_task_ids][
                            :effective_fleet_size
                        ]
                else:
                    if restart:
                        selected = eligible
                    else:
                        selected = [t for t in eligible if t.id not in active_task_ids]

                if not restart:
                    already_running = [t for t in eligible if t.id in active_task_ids]
                    for t in already_running:
                        skipped.append((t.id, "already running"))

                return selected, skipped, eligible, epic_row
        finally:
            await engine.dispose()

    try:
        selected, skipped, eligible, epic_row = asyncio.run(_plan())
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if epic_row is not None:
        typer.echo(f"Epic: {epic_row.slug} (id={epic_row.id})")

    typer.echo(f"Eligible: {len(eligible)}")
    typer.echo(f"Selected: {len(selected)}")
    if skipped:
        typer.echo("Skipped:")
        for task_id, reason in skipped:
            typer.echo(f"  - {task_id}: {reason}")

    if dry_run:
        return

    failures: list[tuple[int, str]] = []
    for task in selected:
        try:
            if restart:
                result = asyncio.run(
                    restart_task_agent(
                        ctx,
                        task_id=task.id,
                        harness_command=effective_harness,
                        detach=effective_detach,
                    )
                )
            else:
                result = asyncio.run(
                    start_task_agent(
                        ctx,
                        task_id=task.id,
                        harness_command=effective_harness,
                        detach=effective_detach,
                    )
                )
            for warning in result.warnings:
                typer.echo(f"warning: {warning}", err=True)
            verb = (
                "Restarted"
                if restart
                else ("Started" if result.started else "Already running")
            )
            typer.echo(f"{verb}: {task.id} ({task.title})")
            typer.echo(f"  rn agent attach --task {task.id}")
            typer.echo(f"  rn agent logs --task {task.id}")
            typer.echo(f"  rn shell --task-id {task.id}")
        except RuntimeError as e:
            failures.append((task.id, str(e)))

    if failures:
        typer.echo("Failures:", err=True)
        for task_id, error in failures:
            typer.echo(f"  - {task_id}: {error}", err=True)
        raise typer.Exit(1)


def _ensure_initialized(ctx: RepoContext) -> None:
    """Check that the database is initialized. Safe to call from sync or async contexts."""
    if not ctx.db_path.exists():
        raise NotInitializedError(
            "Redesmyn is not initialized in this repo. Run `rn init`."
        )

    async def _validate() -> None:
        engine = create_engine(ctx.db_path)
        try:
            await init_db(engine, migrate=False)
        finally:
            await engine.dispose()

    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in async context - create a task and let it run
            # We can't block here, so just check the sync parts
            # The async validation will happen naturally when the caller awaits
            return
        else:
            asyncio.run(_validate())
    except (DatabaseNotInitializedError, DatabaseMigrationRequiredError) as e:
        raise NotInitializedError(str(e)) from e


async def _load_repository_row(ctx: RepoContext) -> Repository | None:
    if not ctx.db_path.exists():
        return None

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            return await session.scalar(
                select(Repository).where(Repository.repo_root == str(ctx.repo_root))
            )
    finally:
        await engine.dispose()


async def _require_repo_row(ctx: RepoContext) -> Repository:
    repo = await _load_repository_row(ctx)
    if repo is None:
        raise NotInitializedError(
            "Redesmyn is not initialized in this repo. Run `rn init`."
        )
    return repo


async def _resolve_epic(ctx: RepoContext, *, epic: str | None) -> Epic:
    _ensure_initialized(ctx)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            repo = await session.scalar(
                select(Repository).where(Repository.repo_root == str(ctx.repo_root))
            )
            if repo is None:
                raise NotInitializedError(
                    "Redesmyn is not initialized in this repo. Run `rn init`."
                )

            if epic is None:
                epics = list(
                    await session.scalars(
                        select(Epic)
                        .where(Epic.repository_id == repo.id)
                        .order_by(Epic.id)
                    )
                )
                if len(epics) == 1:
                    return epics[0]
                if not epics:
                    raise typer.BadParameter(
                        "No epics found. Create one with `rn epic create`."
                    )
                raise typer.BadParameter("Multiple epics found; pass --epic <slug|id>.")

            if epic.isdigit():
                row = await session.get(Epic, int(epic))
            else:
                row = await session.scalar(
                    select(Epic).where(
                        Epic.repository_id == repo.id,
                        Epic.slug == epic,
                    )
                )
            if row is None:
                raise typer.BadParameter(f"Unknown epic: {epic}")
            return row
    finally:
        await engine.dispose()


def _infer_single_epic_slug_from_fs(repo_root: Path) -> str | None:
    epics_dir = repo_root / "epics"
    if not epics_dir.exists():
        return None
    slugs = [
        p.name for p in epics_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    ]
    if len(slugs) == 1:
        return slugs[0]
    return None


async def _sync_from_local(
    ctx: RepoContext,
    *,
    epic: str | None,
    create_branches: bool,
) -> SyncStats:
    epic_fs = _infer_single_epic_slug_from_fs(ctx.worktree_root)
    requested = epic or epic_fs

    engine = create_engine(ctx.db_path)
    stats = SyncStats()
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            repo = await session.scalar(
                select(Repository).where(Repository.repo_root == str(ctx.repo_root))
            )
            if repo is None:
                raise NotInitializedError(
                    "Redesmyn is not initialized in this repo. Run `rn init`."
                )

            epic_row: Epic | None = None
            if requested is None:
                epics = list(
                    await session.scalars(
                        select(Epic)
                        .where(Epic.repository_id == repo.id)
                        .order_by(Epic.id)
                    )
                )
                if len(epics) == 1:
                    epic_row = epics[0]
                elif len(epics) > 1:
                    raise typer.BadParameter(
                        "Multiple epics found; pass --epic <slug|id>."
                    )
                else:
                    raise typer.BadParameter(
                        "No epics found; create one with `rn epic create`."
                    )

            if epic_row is None and requested is not None:
                if requested.isdigit():
                    epic_row = await session.get(Epic, int(requested))
                else:
                    epic_row = await session.scalar(
                        select(Epic).where(
                            Epic.repository_id == repo.id,
                            Epic.slug == requested,
                        )
                    )

            epic_slug = epic_row.slug if epic_row is not None else requested
            if not epic_slug:
                raise typer.BadParameter("Could not infer epic; pass --epic <slug|id>.")

            epic_readme = ctx.worktree_root / "epics" / epic_slug / "README.md"
            if not epic_readme.exists():
                raise typer.BadParameter(f"Epic doc not found: {epic_readme}")

            try:
                epic_doc = load_epic_doc(epic_readme)
            except DocLoadError as e:
                raise typer.BadParameter(str(e)) from e

            if epic_row is None:
                epic_row = Epic(
                    repository_id=repo.id,
                    name=epic_doc.metadata.name,
                    slug=epic_doc.metadata.slug,
                    root_branch=epic_doc.metadata.root_branch,
                    linear_project_id=epic_doc.metadata.linear_project_id,
                )
                session.add(epic_row)
                await session.flush()
                stats.epics_created += 1
            else:
                updated = False
                if epic_row.name != epic_doc.metadata.name:
                    epic_row.name = epic_doc.metadata.name
                    updated = True
                if epic_row.root_branch != epic_doc.metadata.root_branch:
                    epic_row.root_branch = epic_doc.metadata.root_branch
                    updated = True
                if (
                    epic_doc.metadata.linear_project_id is not None
                    and epic_row.linear_project_id
                    != epic_doc.metadata.linear_project_id
                ):
                    epic_row.linear_project_id = epic_doc.metadata.linear_project_id
                    updated = True
                if updated:
                    stats.epics_updated += 1

            tasks_dir = ctx.worktree_root / "epics" / epic_row.slug / "tasks"
            task_readmes = (
                sorted(tasks_dir.glob("*/README.md")) if tasks_dir.exists() else []
            )

            task_docs = []
            for readme in task_readmes:
                try:
                    task_docs.append(load_task_doc(readme))
                except DocLoadError as e:
                    raise typer.BadParameter(str(e)) from e

            created_task_ids: set[int] = set()
            updated_task_ids: set[int] = set()
            task_by_ref: dict[str, Task] = {}
            task_by_path: dict[Path, Task] = {}

            def _maybe_rename_task_worktree_branch(
                *, task: Task, desired_branch: str
            ) -> None:
                if not task.worktree_path:
                    return
                worktree_path = Path(task.worktree_path)
                if not worktree_path.exists():
                    return

                current = current_branch(cwd=worktree_path)
                if current == desired_branch:
                    return
                if current == "HEAD":
                    raise typer.BadParameter(
                        f"Task {task.id} worktree is not on a branch (detached HEAD): {worktree_path}"
                    )
                if branch_exists(ctx.repo_root, desired_branch):
                    raise typer.BadParameter(
                        "Cannot rename task worktree branch; target branch already exists: "
                        f"{desired_branch!r} (task {task.id}, worktree {worktree_path})"
                    )

                try:
                    git_rename_current_branch(worktree_path, new_name=desired_branch)
                except GitCommandError as e:
                    raise typer.BadParameter(
                        f"Failed to rename task {task.id} worktree branch {current!r} -> {desired_branch!r}: {e}"
                    ) from e

            for doc in task_docs:
                if doc.title is None:
                    raise typer.BadParameter(f"Task doc missing title (H1): {doc.path}")

                meta = doc.metadata
                linear_issue_id = meta.linear.issue_id if meta.linear else None
                linear_identifier = meta.linear.identifier if meta.linear else None
                rel_path = str(doc.path.relative_to(ctx.worktree_root))

                task: Task | None = None
                if linear_issue_id:
                    task = await session.scalar(
                        select(Task).where(
                            Task.epic_id == epic_row.id,
                            Task.linear_issue_id == linear_issue_id,
                        )
                    )

                if task is None:
                    task = await session.scalar(
                        select(Task).where(
                            Task.epic_id == epic_row.id,
                            Task.local_path == rel_path,
                        )
                    )

                if task is None:
                    task = Task(
                        epic_id=epic_row.id,
                        title=doc.title,
                        body=doc.markdown,
                        source=TaskSource.Linear
                        if linear_issue_id
                        else TaskSource.Local,
                        state=TaskState.Todo,
                        linear_issue_id=linear_issue_id,
                        linear_identifier=linear_identifier,
                        local_path=rel_path,
                    )
                    session.add(task)
                    await session.flush()
                    created_task_ids.add(task.id)
                    stats.tasks_created += 1
                else:
                    updated = False
                    if task.title != doc.title:
                        task.title = doc.title
                        updated = True
                    if task.body != doc.markdown:
                        task.body = doc.markdown
                        updated = True
                    if task.local_path != rel_path:
                        task.local_path = rel_path
                        updated = True
                    if linear_issue_id and task.linear_issue_id != linear_issue_id:
                        task.linear_issue_id = linear_issue_id
                        task.source = TaskSource.Linear
                        updated = True
                    if (
                        linear_identifier
                        and task.linear_identifier != linear_identifier
                    ):
                        task.linear_identifier = linear_identifier
                        updated = True
                    if updated:
                        updated_task_ids.add(task.id)

                task_by_path[doc.path] = task

                for ref in (meta.id, linear_identifier, linear_issue_id):
                    if not ref:
                        continue
                    existing_ref = task_by_ref.get(ref)
                    if existing_ref is not None and existing_ref.id != task.id:
                        raise typer.BadParameter(
                            f"Ambiguous task ref {ref!r}; matches multiple tasks in docs (v0)"
                        )
                    task_by_ref[ref] = task

                if not create_branches:
                    continue

                branch = meta.node.branch if meta.node and meta.node.branch else None
                if not branch:
                    identifier = linear_identifier or meta.id or f"task-{task.id}"
                    short = (
                        slugify(doc.title, fallback="task")[:60].strip("-") or "task"
                    )
                    branch = f"rn/{epic_row.slug}/{identifier}-{short}"

                existing = await session.scalar(
                    select(Task).where(
                        Task.epic_id == epic_row.id,
                        Task.branch_name == branch,
                        Task.id != task.id,
                    )
                )
                if existing is not None:
                    raise typer.BadParameter(
                        f"Ambiguous branch {branch!r}; assigned to multiple tasks ({existing.id} and {task.id})"
                    )

                _maybe_rename_task_worktree_branch(task=task, desired_branch=branch)
                if task.branch_name != branch:
                    if task.branch_name is None and branch is not None:
                        stats.branches_created += 1
                    else:
                        stats.branches_updated += 1
                    task.branch_name = branch
                    if task.id not in created_task_ids:
                        updated_task_ids.add(task.id)

            for doc in task_docs:
                task = task_by_path.get(doc.path)
                if task is None:
                    continue
                parent_ref = doc.metadata.stacked_on
                if not parent_ref:
                    if task.parent_task_id is not None:
                        task.parent_task_id = None
                        if task.id not in created_task_ids:
                            updated_task_ids.add(task.id)
                    continue
                parent_task = task_by_ref.get(parent_ref)
                if parent_task is None:
                    raise typer.BadParameter(
                        f"Unknown stacked_on ref {parent_ref!r} in {doc.path}"
                    )
                if task.parent_task_id != parent_task.id:
                    task.parent_task_id = parent_task.id
                    if task.id not in created_task_ids:
                        updated_task_ids.add(task.id)

            stats.tasks_updated = len(updated_task_ids)
            await session.commit()
            return stats
    finally:
        await engine.dispose()


async def _sync_from_linear(
    ctx: RepoContext,
    *,
    epic: str | None,
    project: str | None,
    create_branches: bool,
) -> SyncStats:
    creds = await _require_fresh_linear_credentials(ctx=ctx)
    client = LinearClient(access_token=creds.access_token)

    epic_row = await _resolve_epic(ctx, epic=epic)

    epic_readme = ctx.worktree_root / "epics" / epic_row.slug / "README.md"
    epic_doc = None
    if epic_readme.exists():
        try:
            epic_doc = load_epic_doc(epic_readme)
        except DocLoadError:
            epic_doc = None

    project_id = project or epic_row.linear_project_id
    if not project_id and epic_doc is not None:
        project_id = epic_doc.metadata.linear_project_id

    milestone_raw = (
        epic_doc.metadata.linear.milestone_id
        if epic_doc is not None and epic_doc.metadata.linear is not None
        else None
    )

    if not project_id:
        raise typer.BadParameter(
            "Missing Linear project id. Pass --project <id|slug|name>, set it in the epic doc metadata, "
            "or run `rn linear projects` to discover ids."
        )
    try:
        project_id = await _resolve_linear_project_id(client, project_id)
    except typer.BadParameter:
        raise
    except Exception as e:
        raise typer.BadParameter(str(e)) from e

    milestone: LinearMilestone | None = None
    if milestone_raw:
        milestone = await _resolve_linear_milestone(
            client, project_id=project_id, raw=milestone_raw
        )

    def _load_metadata_dict(markdown: str) -> dict[str, Any]:
        try:
            block = extract_fenced_block_after_heading(
                markdown, heading="Metadata", allowed_langs={"yaml", "yml"}
            )
        except MarkdownSectionError:
            return {}
        return parse_yaml_block(block.content)

    def _parse_task_number(task_id: str | None) -> int | None:
        if not task_id:
            return None
        prefix = "T-"
        if not task_id.startswith(prefix):
            return None
        suffix = task_id[len(prefix) :]
        if not suffix.isdigit():
            return None
        return int(suffix)

    if milestone is not None:
        issues = await fetch_project_issues_by_milestone(
            client, project_id=project_id, milestone_id=milestone.id
        )
    else:
        issues = await fetch_project_issues_by_label(
            client, project_id=project_id, label_name=epic_row.slug
        )
    try:
        relations = await fetch_project_issue_relations(client, project_id=project_id)
    except Exception as e:
        typer.echo(
            f"warning: could not fetch issue relations; parent inference disabled ({e})",
            err=True,
        )
        relations = []

    issues_by_id = {i.id: i for i in issues}
    blockers_by_issue: dict[str, list[str]] = {i.id: [] for i in issues}
    for rel in relations:
        rel_type = rel.type.lower()
        if rel_type == "blocks":
            if rel.related_issue_id in blockers_by_issue:
                blockers_by_issue[rel.related_issue_id].append(rel.issue_id)
            continue
        if (
            rel_type in {"blocked_by", "blockedby"}
            and rel.issue_id in blockers_by_issue
        ):
            blockers_by_issue[rel.issue_id].append(rel.related_issue_id)

    for issue_id, blockers in blockers_by_issue.items():
        blockers_by_issue[issue_id] = list(dict.fromkeys(blockers))

    epic_dir = ctx.worktree_root / "epics" / epic_row.slug
    tasks_dir = epic_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    epic_readme = epic_dir / "README.md"
    if epic_readme.exists():
        markdown = epic_readme.read_text(encoding="utf-8")
        yaml_data = _load_metadata_dict(markdown)
        yaml_data["slug"] = epic_row.slug
        yaml_data["name"] = epic_row.name
        yaml_data["root_branch"] = epic_row.root_branch
        linear_meta = yaml_data.get("linear")
        if not isinstance(linear_meta, dict):
            linear_meta = {}
            yaml_data["linear"] = linear_meta
        linear_meta["project_id"] = project_id
        epic_readme.write_text(
            upsert_metadata_yaml(markdown, yaml_data=yaml_data), encoding="utf-8"
        )

    def _branch_name(identifier: str, title: str) -> str:
        short = slugify(title, fallback="task")[:60].strip("-") or "task"
        return f"rn/{epic_row.slug}/{identifier}-{short}"

    existing_readmes = sorted(tasks_dir.glob("*/README.md"))
    existing_readme_by_linear_issue_id: dict[str, Path] = {}
    existing_task_id_by_linear_issue_id: dict[str, str] = {}
    used_task_numbers: set[int] = set()

    for readme in existing_readmes:
        try:
            doc = load_task_doc(readme)
        except DocLoadError as e:
            raise typer.BadParameter(str(e)) from e

        meta = doc.metadata
        dir_name = readme.parent.name
        meta_task_id = meta.id
        task_num = _parse_task_number(meta_task_id)
        task_id = meta_task_id
        if task_num is None:
            task_num = _parse_task_number(dir_name)
            if task_num is not None:
                task_id = dir_name
        if task_num is not None:
            used_task_numbers.add(task_num)

        linear_issue_id = meta.linear.issue_id if meta.linear else None
        if not linear_issue_id:
            continue
        existing = existing_readme_by_linear_issue_id.get(linear_issue_id)
        if existing is not None and existing != readme:
            raise typer.BadParameter(
                f"Multiple task docs found for Linear issue {linear_issue_id} (v0)"
            )
        existing_readme_by_linear_issue_id[linear_issue_id] = readme
        if task_id:
            existing_task_id_by_linear_issue_id[linear_issue_id] = task_id

    next_task_number = (max(used_task_numbers) + 1) if used_task_numbers else 1
    local_task_id_by_issue_id: dict[str, str] = {}
    for issue in sorted(issues, key=lambda x: x.identifier):
        if existing_id := existing_task_id_by_linear_issue_id.get(issue.id):
            local_task_id_by_issue_id[issue.id] = existing_id
            continue
        while next_task_number in used_task_numbers:
            next_task_number += 1
        local_task_id_by_issue_id[issue.id] = f"T-{next_task_number}"
        used_task_numbers.add(next_task_number)
        next_task_number += 1

    def _choose_parent_issue_id(
        *,
        issue: LinearIssue,
        blocker_issue_ids: list[str],
        existing_stacked_on: str | None,
        existing_must_land_after: list[str],
    ) -> str | None:
        imported_blocker_ids = sorted(
            [bid for bid in blocker_issue_ids if bid in issues_by_id],
            key=lambda bid: issues_by_id[bid].identifier,
        )
        if not imported_blocker_ids:
            return None
        if len(imported_blocker_ids) == 1:
            return imported_blocker_ids[0]

        if existing_stacked_on:
            for bid in imported_blocker_ids:
                if existing_stacked_on in {
                    local_task_id_by_issue_id.get(bid),
                    issues_by_id[bid].identifier,
                    bid,
                }:
                    return bid

        if existing_stacked_on is None:
            for ref in existing_must_land_after:
                for bid in imported_blocker_ids:
                    if ref in {
                        local_task_id_by_issue_id.get(bid),
                        issues_by_id[bid].identifier,
                        bid,
                    }:
                        return None

        if not sys.stdin.isatty():
            raise typer.BadParameter(
                f"Linear issue {issue.identifier} has multiple blockers; run in a TTY to choose a parent (v0)"
            )

        typer.echo("")
        typer.echo(
            f"Linear issue {issue.identifier} has multiple blockers; choose `stacked_on`:"
        )
        typer.echo("  0) No parent")
        for idx, bid in enumerate(imported_blocker_ids, start=1):
            blocker = issues_by_id[bid]
            local_id = local_task_id_by_issue_id.get(bid)
            if local_id:
                typer.echo(
                    f"  {idx}) {local_id} ({blocker.identifier}) {blocker.title}"
                )
            else:
                typer.echo(f"  {idx}) {blocker.identifier} {blocker.title}")

        while True:
            raw = typer.prompt("Parent", default="0").strip()
            if not raw:
                return None
            if not raw.isdigit():
                typer.echo("Enter a number from the list above.", err=True)
                continue
            choice = int(raw)
            if choice == 0:
                return None
            if 1 <= choice <= len(imported_blocker_ids):
                return imported_blocker_ids[choice - 1]
            typer.echo("Enter a number from the list above.", err=True)

    for issue in sorted(issues, key=lambda x: x.identifier):
        local_task_id = local_task_id_by_issue_id[issue.id]

        target_dir = tasks_dir / local_task_id
        target_readme = target_dir / "README.md"

        existing_readme = existing_readme_by_linear_issue_id.get(issue.id)
        if existing_readme:
            existing_dir = existing_readme.parent
            if existing_dir != target_dir and target_dir.exists():
                raise typer.BadParameter(
                    f"Task path collision for {local_task_id}: {target_dir} already exists (v0)"
                )
            if existing_dir != target_dir:
                existing_dir.rename(target_dir)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

        if target_readme.exists():
            markdown = target_readme.read_text(encoding="utf-8")
        else:
            markdown = f"# {issue.identifier} {issue.title}\n\n## Brief (local)\n\n"

        existing_meta = _load_metadata_dict(markdown)
        existing_stacked_on = None
        existing_must_land_after: list[str] = []
        try:
            doc = load_task_doc(target_readme) if target_readme.exists() else None
        except DocLoadError:
            doc = None
        if doc is not None:
            existing_stacked_on = doc.metadata.stacked_on
            existing_must_land_after = doc.metadata.must_land_after

        blocker_issue_ids = blockers_by_issue.get(issue.id, [])
        parent_issue_id = _choose_parent_issue_id(
            issue=issue,
            blocker_issue_ids=blocker_issue_ids,
            existing_stacked_on=existing_stacked_on,
            existing_must_land_after=existing_must_land_after,
        )
        stacked_on = (
            local_task_id_by_issue_id[parent_issue_id] if parent_issue_id else None
        )

        imported_blockers = sorted(
            [bid for bid in blocker_issue_ids if bid in local_task_id_by_issue_id],
            key=lambda bid: issues_by_id[bid].identifier
            if bid in issues_by_id
            else bid,
        )
        external_blockers = sorted(
            [bid for bid in blocker_issue_ids if bid not in issues_by_id]
        )
        must_land_after: list[str] = []
        for bid in [*imported_blockers, *external_blockers]:
            ref = local_task_id_by_issue_id.get(bid) or bid
            if stacked_on and ref == stacked_on:
                continue
            must_land_after.append(ref)
        must_land_after_dedup: list[str] = []
        seen = set()
        for ref in must_land_after:
            if ref in seen:
                continue
            seen.add(ref)
            must_land_after_dedup.append(ref)

        existing_meta["id"] = local_task_id
        existing_meta["stacked_on"] = stacked_on
        existing_meta["must_land_after"] = must_land_after_dedup

        linear_meta = existing_meta.get("linear")
        if not isinstance(linear_meta, dict):
            linear_meta = {}
            existing_meta["linear"] = linear_meta
        linear_meta["issue_id"] = issue.id
        linear_meta["identifier"] = issue.identifier

        node_meta = existing_meta.get("node")
        if not isinstance(node_meta, dict):
            node_meta = {}
            existing_meta["node"] = node_meta
        branch = node_meta.get("branch")
        if not isinstance(branch, str) or not branch.strip():
            node_meta["branch"] = _branch_name(issue.identifier, issue.title)

        markdown = upsert_metadata_yaml(markdown, yaml_data=existing_meta)

        synced_lines: list[str] = []
        if issue.state_type:
            synced_lines.append(f"State: {issue.state_type}")
        if issue.description:
            synced_lines.append("")
            synced_lines.append(issue.description.strip())

        markdown = upsert_synced_section(
            markdown,
            title="Synced (from Linear)",
            content="\n".join(synced_lines).rstrip(),
        )
        target_readme.write_text(markdown, encoding="utf-8")

    stats = await _sync_from_local(
        ctx, epic=epic_row.slug, create_branches=create_branches
    )

    issue_state_type_by_id = {issue.id: issue.state_type for issue in issues}
    issue_state_name_by_id = {issue.id: issue.state_name for issue in issues}
    issue_state_issue_ids = sorted(
        set(issue_state_type_by_id.keys()) | set(issue_state_name_by_id.keys())
    )
    if issue_state_issue_ids:
        observed_at = datetime.now(UTC)
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                rows = list(
                    await session.scalars(
                        select(Task).where(
                            Task.epic_id == epic_row.id,
                            Task.linear_issue_id.in_(issue_state_issue_ids),
                        )
                    )
                )
                for task_row in rows:
                    if not task_row.linear_issue_id:
                        continue
                    task_row.linear_state_type = issue_state_type_by_id.get(
                        task_row.linear_issue_id
                    )
                    task_row.linear_state_name = issue_state_name_by_id.get(
                        task_row.linear_issue_id
                    )
                    task_row.linear_state_observed_at = observed_at
                await session.commit()
        finally:
            await engine.dispose()

    return stats


def _extract_markdown_section(markdown: str, *, heading: str) -> str | None:
    lines = markdown.splitlines()
    target = heading.strip().lower()

    heading_index: int | None = None
    heading_level: int | None = None
    for idx, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        hashes, sep, title = line.partition(" ")
        if not sep or not hashes or not set(hashes) <= {"#"}:
            continue
        if title.strip().lower() != target:
            continue
        heading_index = idx
        heading_level = len(hashes)
        break

    if heading_index is None or heading_level is None:
        return None

    body: list[str] = []
    for line in lines[heading_index + 1 :]:
        if line.startswith("#"):
            hashes, sep, _ = line.partition(" ")
            if sep and hashes and set(hashes) <= {"#"} and len(hashes) <= heading_level:
                break
        body.append(line)

    content = "\n".join(body).strip()
    return content or None


def _looks_like_uuid(value: str) -> bool:
    # Linear issue ids are UUIDs. We accept UUID-shaped values for best-effort
    # external blocker refs.
    parts = value.split("-")
    if len(parts) != 5:
        return False
    expected_lens = [8, 4, 4, 4, 12]
    for part, expected_len in zip(parts, expected_lens, strict=True):
        if len(part) != expected_len:
            return False
        try:
            int(part, 16)
        except ValueError:
            return False
    return True


def _format_linear_project(project: LinearProject) -> str:
    slug = project.slug or ""
    teams = ", ".join(f"{t.key} {t.name}" for t in project.teams)
    return f"{project.id}\t{slug}\t{project.name}\t{teams}".rstrip()


def _format_linear_milestone(milestone: LinearMilestone) -> str:
    return f"{milestone.id}\t{milestone.name}".rstrip()


async def _resolve_linear_milestone(
    client: LinearClient, *, project_id: str, raw: str
) -> LinearMilestone:
    value = raw.strip()
    if not value:
        raise typer.BadParameter("Missing Linear milestone value")

    try:
        milestones = await fetch_project_milestones(client, project_id=project_id)
    except Exception as e:
        raise typer.BadParameter(f"Could not list Linear milestones: {e}") from e

    if _looks_like_uuid(value):
        for m in milestones:
            if m.id == value:
                return m
        raise typer.BadParameter(
            f"Unknown Linear milestone {raw!r}. Run `rn linear milestones --project ...` to list valid ids."
        )

    candidates = [m for m in milestones if m.name == value]
    if not candidates:
        lowered = value.lower()
        candidates = [m for m in milestones if m.name.lower() == lowered]

    if not candidates:
        raise typer.BadParameter(
            f"Unknown Linear milestone {raw!r}. Run `rn linear milestones --project ...` to list valid ids."
        )
    if len(candidates) == 1:
        return candidates[0]

    preview = "\n".join(f"  {_format_linear_milestone(m)}" for m in candidates[:15])
    suffix = "\n  …" if len(candidates) > 15 else ""
    raise typer.BadParameter(
        f"Multiple Linear milestones match {raw!r}:\n{preview}{suffix}\n"
        "Run `rn linear milestones --project ...` to pick a unique id."
    )


async def _resolve_linear_project_id(client: LinearClient, raw: str) -> str:
    value = raw.strip()
    if not value:
        raise typer.BadParameter("Missing Linear project value")

    if _looks_like_uuid(value):
        try:
            project = await fetch_project(client, project_id=value)
        except Exception:
            project = None
        if project is None:
            raise typer.BadParameter(
                f"Unknown Linear project {raw!r}. Run `rn linear projects` to list valid ids."
            )
        return project.id

    try:
        projects = await fetch_projects(client)
    except Exception as e:
        raise typer.BadParameter(f"Could not list Linear projects: {e}") from e

    def _match_slug(v: str) -> list[LinearProject]:
        return [p for p in projects if p.slug == v]

    def _match_name(v: str) -> list[LinearProject]:
        return [p for p in projects if p.name == v]

    candidates = _match_slug(value)
    if not candidates:
        candidates = _match_name(value)
    if not candidates:
        lowered = value.lower()
        candidates = [p for p in projects if (p.slug or "").lower() == lowered]
    if not candidates:
        lowered = value.lower()
        candidates = [p for p in projects if p.name.lower() == lowered]
    if not candidates and "-" not in value:
        # Linear project URLs look like `/project/v0-<slugId>/...`. Let users
        # paste either the full slug (`v0-...`) or the short suffix.
        candidates = _match_slug(f"v0-{value}")

    if not candidates:
        raise typer.BadParameter(
            f"Unknown Linear project {raw!r}. Run `rn linear projects` to list valid ids."
        )
    if len(candidates) == 1:
        return candidates[0].id

    preview = "\n".join(f"  {_format_linear_project(p)}" for p in candidates[:15])
    suffix = "\n  …" if len(candidates) > 15 else ""
    raise typer.BadParameter(
        f"Multiple Linear projects match {raw!r}:\n{preview}{suffix}\n"
        "Run `rn linear projects` to pick a unique id."
    )


def _task_ref_from_doc(doc_path: Path, meta: TaskMetadata) -> str | None:
    if meta.id:
        return meta.id
    name = doc_path.parent.name
    if name.startswith("T-") and name[2:].isdigit():
        return name
    return None


def _linear_title_from_doc(
    title: str, *, identifier: str | None, local_id: str | None
) -> str:
    t = title.strip()
    if identifier and t.startswith(f"{identifier} "):
        t = t.removeprefix(f"{identifier} ").strip()
    if local_id and t.startswith(f"{local_id} "):
        t = t.removeprefix(f"{local_id} ").strip()
    return t or title.strip()


async def _sync_to_linear(
    ctx: RepoContext,
    *,
    epic: str | None,
    project: str | None,
    create_branches: bool,
    dry_run: bool,
    output_format: SyncOutputFormat,
) -> LinearPushStats:
    # Make sure local docs/DB are in sync before we push, and so we have Task rows
    # (including state) for every task doc.
    requested_epic = epic or _infer_single_epic_slug_from_fs(ctx.worktree_root)
    # For --dry-run we avoid branch graph churn, but still want the DB projection
    # up to date so state/title refs are current.
    await _sync_from_local(
        ctx, epic=requested_epic, create_branches=(create_branches and not dry_run)
    )

    epic_row = await _resolve_epic(ctx, epic=requested_epic)
    creds = await _require_fresh_linear_credentials(ctx=ctx)
    client = LinearClient(access_token=creds.access_token)

    epic_readme = ctx.worktree_root / "epics" / epic_row.slug / "README.md"
    epic_doc = None
    if epic_readme.exists():
        try:
            epic_doc = load_epic_doc(epic_readme)
        except DocLoadError:
            epic_doc = None

    project_id = project or epic_row.linear_project_id
    if not project_id and epic_doc is not None:
        project_id = epic_doc.metadata.linear_project_id

    milestone_raw = (
        epic_doc.metadata.linear.milestone_id
        if epic_doc is not None and epic_doc.metadata.linear is not None
        else None
    )

    if not project_id:
        raise typer.BadParameter(
            "Missing Linear project id. Pass --project <id|slug|name>, set it in the epic doc metadata, "
            "or run `rn linear projects` to discover ids."
        )
    try:
        project_id = await _resolve_linear_project_id(client, project_id)
    except typer.BadParameter:
        raise
    except Exception as e:
        raise typer.BadParameter(str(e)) from e

    milestone: LinearMilestone | None = None
    if milestone_raw:
        milestone = await _resolve_linear_milestone(
            client, project_id=project_id, raw=milestone_raw
        )

    stats = LinearPushStats()

    epic_dir = ctx.worktree_root / "epics" / epic_row.slug
    tasks_dir = epic_dir / "tasks"
    task_readmes = sorted(tasks_dir.glob("*/README.md")) if tasks_dir.exists() else []
    if not task_readmes:
        return stats

    def _raise_linear(e: LinearApiError) -> None:
        msg = str(e)
        if (
            "Invalid scope" in msg
            or e.status_code == 403
            or (e.code or "").upper() == "FORBIDDEN"
        ):
            raise typer.BadParameter(
                "Linear token lacks write scope. Set REDESMYN_LINEAR_SCOPES='read,write' (or 'read write'), "
                "then run `rn linear logout` and `rn linear auth`."
            ) from e
        raise typer.BadParameter(f"Linear API error: {msg}") from e

    task_docs: list[tuple[Path, "TaskDoc"]] = []
    for readme in task_readmes:
        try:
            task_docs.append((readme, load_task_doc(readme)))
        except DocLoadError as e:
            raise typer.BadParameter(str(e)) from e

    def _load_metadata_dict(markdown: str) -> dict[str, Any]:
        try:
            block = extract_fenced_block_after_heading(
                markdown, heading="Metadata", allowed_langs={"yaml", "yml"}
            )
        except MarkdownSectionError:
            return {}
        return parse_yaml_block(block.content)

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            epic_db = await session.get(Epic, epic_row.id)
            if epic_db is None:
                raise typer.BadParameter("Epic not found")
            if (
                epic_db.linear_project_id is not None
                and epic_db.linear_project_id != project_id
            ):
                raise typer.BadParameter(
                    f"Epic is linked to a different Linear project ({epic_db.linear_project_id})"
                )
            if epic_db.linear_project_id is None and not dry_run:
                epic_db.linear_project_id = project_id

            label_name = epic_db.slug if milestone is None else None

            defaults = None
            if not dry_run:
                try:
                    defaults = await ensure_linear_write_defaults(
                        session,
                        epic_id=epic_db.id,
                        label_name=label_name,
                        project_id=project_id,
                        client=client,
                    )
                except LinearApiError as e:
                    _raise_linear(e)

            tasks = list(
                await session.scalars(
                    select(Task).where(Task.epic_id == epic_db.id).order_by(Task.id)
                )
            )
            tasks_by_local_path = {
                t.local_path: t for t in tasks if t.local_path is not None
            }

            state_id_cache: dict[tuple[str, str], str] = {}

            async def _state_id(*, team_id: str, state_type: str) -> str:
                key = (team_id, state_type)
                cached = state_id_cache.get(key)
                if cached is not None:
                    return cached
                try:
                    resolved = await resolve_team_state_id(
                        client, team_id=team_id, state_type=state_type
                    )
                except LinearApiError as e:
                    _raise_linear(e)
                state_id_cache[key] = resolved
                return resolved

            def _preview(value: str | None, *, limit: int = 200) -> str | None:
                if value is None:
                    return None
                if len(value) <= limit:
                    return value
                return f"{value[:limit]}…"

            if dry_run:
                from redesmyn.integrations.linear import (
                    fetch_issue_blocker_ids,
                    resolve_default_team,
                )

                planned_issue_ops: list[dict[str, Any]] = []
                planned_doc_ops: list[dict[str, Any]] = []
                planned_blocker_ops: list[dict[str, Any]] = []
                warnings: list[str] = []

                cached_defaults = await load_linear_write_defaults(
                    session, epic_id=epic_db.id
                )
                label_id: str | None = None
                if milestone is None:
                    label_id = cached_defaults.label_id if cached_defaults else None
                team_id: str | None = (
                    cached_defaults.team_id if cached_defaults else None
                )

                if team_id is None:
                    try:
                        team = await resolve_default_team(
                            client,
                            project_id=project_id,
                            preferred_team_id=(
                                cached_defaults.team_id if cached_defaults else None
                            ),
                        )
                        team_id = team.id
                    except Exception:
                        team_id = None

                if milestone is None and label_id is None:
                    try:
                        found = await fetch_label_by_name(
                            client, label_name=epic_db.slug
                        )
                        label_id = found.id if found is not None else None
                    except Exception as e:
                        warnings.append(
                            f"Could not resolve Linear label {epic_db.slug!r}: {e}"
                        )
                        label_id = None

                task_issue_by_path: dict[Path, dict[str, Any]] = {}
                # First pass: plan issue create/update + doc metadata updates.
                for readme, doc in task_docs:
                    if doc.title is None:
                        raise typer.BadParameter(
                            f"Task doc missing title (H1): {readme}"
                        )

                    rel_path = str(readme.relative_to(ctx.worktree_root))
                    task_row = tasks_by_local_path.get(rel_path)
                    if task_row is None:
                        raise typer.BadParameter(f"Task not found in DB for {rel_path}")

                    meta = doc.metadata
                    doc_issue_id = meta.linear.issue_id if meta.linear else None
                    db_issue_id = task_row.linear_issue_id
                    if doc_issue_id and db_issue_id and doc_issue_id != db_issue_id:
                        raise typer.BadParameter(
                            f"Task {rel_path} has conflicting Linear issue ids (doc={doc_issue_id}, db={db_issue_id})"
                        )
                    issue_id = doc_issue_id or db_issue_id

                    local_ref = _task_ref_from_doc(readme, meta)
                    desired_title = _linear_title_from_doc(
                        doc.title or "",
                        identifier=meta.linear.identifier if meta.linear else None,
                        local_id=local_ref,
                    )
                    brief = _extract_markdown_section(
                        doc.markdown, heading="Brief (local)"
                    )
                    task_state = task_row.state if task_row.state else TaskState.Todo
                    desired_state_type = linear_state_type_from_task_state(task_state)

                    existing: LinearIssue | None = None
                    if issue_id:
                        try:
                            existing = await fetch_issue(client, issue_id=issue_id)
                        except LinearApiError as e:
                            _raise_linear(e)

                    changes: dict[str, Any] = {}
                    ensure_scope: dict[str, Any]
                    needs_scope = False
                    if milestone is not None:
                        ensure_scope = {
                            "mode": "milestone",
                            "id": milestone.id,
                            "name": milestone.name,
                        }
                        if existing is not None:
                            current = await fetch_issue_project_milestone_id(
                                client, issue_id=existing.id
                            )
                            if current != milestone.id:
                                needs_scope = True
                                changes["project_milestone_id"] = {
                                    "from": current,
                                    "to": milestone.id,
                                }
                    else:
                        ensure_scope = {
                            "mode": "label",
                            "id": label_id,
                            "name": epic_db.slug,
                        }
                        if existing is not None and label_id is not None:
                            needs_scope = label_id not in existing.label_ids
                            if needs_scope:
                                changes["labels_add"] = [label_id]
                        elif existing is not None and label_id is None:
                            changes["labels_add"] = ["<unknown_label_id>"]

                    if existing is not None:
                        desired_description = (
                            brief if brief is not None else existing.description
                        )
                        if existing.title != desired_title:
                            changes["title"] = {
                                "from": existing.title,
                                "to": desired_title,
                            }
                        if (existing.description or "") != (desired_description or ""):
                            changes["description"] = {
                                "from_preview": _preview(existing.description),
                                "to_preview": _preview(desired_description),
                                "from_len": len(existing.description or ""),
                                "to_len": len(desired_description or ""),
                            }
                        if (
                            existing.state_type or ""
                        ).lower() != desired_state_type.lower():
                            changes["state_type"] = {
                                "from": existing.state_type,
                                "to": desired_state_type,
                            }
                        action = "update" if (changes or needs_scope) else "noop"
                        planned_issue_ops.append(
                            {
                                "kind": "issue",
                                "action": action,
                                "task": {"path": rel_path, "ref": local_ref},
                                "linear": {
                                    "issue_id": existing.id,
                                    "identifier": existing.identifier,
                                },
                                "ensure_scope": ensure_scope,
                                "changes": changes,
                            }
                        )
                        task_issue_by_path[readme] = {
                            "issue_id": existing.id,
                            "identifier": existing.identifier,
                            "ref": local_ref,
                            "planned_create": False,
                        }

                        markdown = readme.read_text(encoding="utf-8")
                        existing_meta = _load_metadata_dict(markdown)
                        linear_meta = existing_meta.get("linear")
                        if not isinstance(linear_meta, dict):
                            linear_meta = {}
                            existing_meta["linear"] = linear_meta
                        linear_meta["issue_id"] = existing.id
                        linear_meta["identifier"] = existing.identifier
                        next_markdown = upsert_metadata_yaml(
                            markdown, yaml_data=existing_meta
                        )
                        if next_markdown != markdown:
                            planned_doc_ops.append(
                                {
                                    "kind": "doc",
                                    "action": "update",
                                    "task": {"path": rel_path, "ref": local_ref},
                                    "linear": {
                                        "issue_id": existing.id,
                                        "identifier": existing.identifier,
                                    },
                                }
                            )
                    else:
                        planned_issue_ops.append(
                            {
                                "kind": "issue",
                                "action": "create",
                                "task": {"path": rel_path, "ref": local_ref},
                                "linear": {"issue_id": None, "identifier": None},
                                "ensure_scope": ensure_scope,
                                "changes": {
                                    "title": {"to": desired_title},
                                    "description": {
                                        "to_preview": _preview(brief),
                                        "to_len": len(brief or ""),
                                    },
                                    "state_type": {"to": desired_state_type},
                                    "team_id": team_id,
                                    "project_milestone_id": (
                                        milestone.id if milestone is not None else None
                                    ),
                                },
                            }
                        )
                        task_issue_by_path[readme] = {
                            "issue_id": None,
                            "identifier": None,
                            "ref": local_ref,
                            "planned_create": True,
                        }
                        planned_doc_ops.append(
                            {
                                "kind": "doc",
                                "action": "update_after_create",
                                "task": {"path": rel_path, "ref": local_ref},
                            }
                        )

                # Second pass: plan dependency edges after all issues exist.
                ref_to_issue_id: dict[str, str] = {}
                ref_to_planned_task: dict[str, str] = {}
                for readme, info in task_issue_by_path.items():
                    issue_id = info.get("issue_id")
                    identifier = info.get("identifier")
                    local_ref = info.get("ref")
                    if isinstance(issue_id, str) and issue_id:
                        for ref in [local_ref, identifier, issue_id]:
                            if isinstance(ref, str) and ref:
                                existing_issue_id = ref_to_issue_id.get(ref)
                                if (
                                    existing_issue_id is not None
                                    and existing_issue_id != issue_id
                                ):
                                    raise typer.BadParameter(
                                        f"Ambiguous task ref {ref!r}; matches multiple tasks in docs (v0)"
                                    )
                                ref_to_issue_id[ref] = issue_id
                    else:
                        if isinstance(local_ref, str) and local_ref:
                            ref_to_planned_task[local_ref] = str(readme)

                for readme, doc in task_docs:
                    rel_path = str(readme.relative_to(ctx.worktree_root))
                    info = task_issue_by_path.get(readme) or {}
                    issue_id = info.get("issue_id")
                    local_ref = info.get("ref")

                    markdown = readme.read_text(encoding="utf-8")
                    meta_dict = _load_metadata_dict(markdown)
                    try:
                        meta_parsed = TaskMetadata.model_validate(meta_dict)
                    except Exception as e:
                        raise typer.BadParameter(
                            f"Invalid task doc metadata in {readme}: {e}"
                        ) from e

                    refs: list[str] = []
                    if meta_parsed.stacked_on:
                        refs.append(meta_parsed.stacked_on)
                    refs.extend(meta_parsed.must_land_after)

                    deduped: list[str] = []
                    seen = set()
                    for ref in refs:
                        if ref in seen:
                            continue
                        seen.add(ref)
                        deduped.append(ref)

                    desired_ids: list[str] = []
                    pending_refs: list[str] = []
                    unresolved: list[str] = []
                    for ref in deduped:
                        resolved = ref_to_issue_id.get(ref)
                        if resolved is not None:
                            desired_ids.append(resolved)
                            continue
                        if ref in ref_to_planned_task:
                            pending_refs.append(ref)
                            continue
                        if _looks_like_uuid(ref):
                            desired_ids.append(ref)
                            continue
                        unresolved.append(ref)

                    if unresolved:
                        planned_blocker_ops.append(
                            {
                                "kind": "blockers",
                                "action": "skip",
                                "task": {
                                    "path": rel_path,
                                    "ref": local_ref,
                                    "issue_id": issue_id,
                                },
                                "unresolved_refs": unresolved,
                            }
                        )
                        continue

                    if not isinstance(issue_id, str) or not issue_id:
                        planned_blocker_ops.append(
                            {
                                "kind": "blockers",
                                "action": "update_after_create",
                                "task": {
                                    "path": rel_path,
                                    "ref": local_ref,
                                    "issue_id": None,
                                },
                                "desired_blocker_issue_ids": desired_ids,
                                "pending_blocker_refs": pending_refs,
                            }
                        )
                        continue

                    try:
                        existing_blockers = await fetch_issue_blocker_ids(
                            client, issue_id=issue_id
                        )
                    except LinearApiError as e:
                        _raise_linear(e)

                    desired_set = set(desired_ids)
                    existing_set = set(existing_blockers)
                    to_add = sorted(desired_set - existing_set)
                    to_remove = sorted(existing_set - desired_set)
                    action = (
                        "update" if (to_add or to_remove or pending_refs) else "noop"
                    )
                    planned_blocker_ops.append(
                        {
                            "kind": "blockers",
                            "action": action,
                            "task": {
                                "path": rel_path,
                                "ref": local_ref,
                                "issue_id": issue_id,
                            },
                            "desired_blocker_issue_ids": sorted(desired_set),
                            "existing_blocker_issue_ids": sorted(existing_set),
                            "add": to_add,
                            "remove": to_remove,
                            "pending_blocker_refs": pending_refs,
                        }
                    )

                # Emit plan
                scope = (
                    {"mode": "milestone", "id": milestone.id, "name": milestone.name}
                    if milestone is not None
                    else {"mode": "label", "name": epic_db.slug, "id": label_id}
                )
                plan = {
                    "version": 1,
                    "mode": "dry_run",
                    "target": "linear",
                    "epic": epic_db.slug,
                    "project_id": project_id,
                    "scope": scope,
                    "operations": {
                        "issues": planned_issue_ops,
                        "docs": planned_doc_ops,
                        "blockers": planned_blocker_ops,
                    },
                    "warnings": warnings,
                }

                if output_format == SyncOutputFormat.Json:
                    typer.echo(json.dumps(plan, indent=2, sort_keys=True))
                else:
                    typer.echo(
                        f"Scope: {scope['mode']} ({scope.get('name') or scope.get('id')})"
                    )
                    creates = sum(
                        1 for op in planned_issue_ops if op.get("action") == "create"
                    )
                    updates = sum(
                        1 for op in planned_issue_ops if op.get("action") == "update"
                    )
                    noops = sum(
                        1 for op in planned_issue_ops if op.get("action") == "noop"
                    )
                    blocker_updates = sum(
                        1 for op in planned_blocker_ops if op.get("action") == "update"
                    )
                    blocker_skips = sum(
                        1 for op in planned_blocker_ops if op.get("action") == "skip"
                    )
                    typer.echo(
                        f"Plan (dry-run): issues create={creates} update={updates} noop={noops}"
                    )
                    for op in planned_issue_ops:
                        action = op.get("action")
                        task = op.get("task", {})
                        linear = op.get("linear", {})
                        ident = linear.get("identifier") or "<new>"
                        ref = task.get("ref") or task.get("path")
                        changes = op.get("changes") or {}
                        changed_keys = ", ".join(
                            sorted(k for k in changes.keys() if k != "description")
                        )
                        if "description" in changes:
                            changed_keys = (changed_keys + ", description").strip(", ")
                        suffix = f" ({changed_keys})" if changed_keys else ""
                        typer.echo(f"- {action}: {ident} [{ref}]{suffix}")
                    typer.echo(
                        f"Plan (dry-run): blockers update={blocker_updates} skip={blocker_skips}"
                    )
                    for w in warnings:
                        typer.echo(f"warning: {w}", err=True)

                return stats

            assert defaults is not None

            for readme, doc in task_docs:
                if doc.title is None:
                    raise typer.BadParameter(f"Task doc missing title (H1): {readme}")

                rel_path = str(readme.relative_to(ctx.worktree_root))
                task_row = tasks_by_local_path.get(rel_path)
                if task_row is None:
                    raise typer.BadParameter(f"Task not found in DB for {rel_path}")

                meta = doc.metadata
                doc_issue_id = meta.linear.issue_id if meta.linear else None
                db_issue_id = task_row.linear_issue_id
                if doc_issue_id and db_issue_id and doc_issue_id != db_issue_id:
                    raise typer.BadParameter(
                        f"Task {rel_path} has conflicting Linear issue ids (doc={doc_issue_id}, db={db_issue_id})"
                    )
                issue_id = doc_issue_id or db_issue_id

                local_ref = _task_ref_from_doc(readme, meta)
                desired_title = _linear_title_from_doc(
                    doc.title or "",
                    identifier=meta.linear.identifier if meta.linear else None,
                    local_id=local_ref,
                )
                brief = _extract_markdown_section(doc.markdown, heading="Brief (local)")

                task_state = task_row.state if task_row.state else TaskState.Todo
                state_type = linear_state_type_from_task_state(task_state)

                if issue_id:
                    try:
                        existing = await fetch_issue(client, issue_id=issue_id)
                    except LinearApiError as e:
                        _raise_linear(e)
                    team_id = existing.team_id
                    if not team_id:
                        try:
                            team_id = await fetch_issue_team_id(
                                client, issue_id=issue_id
                            )
                        except LinearApiError as e:
                            _raise_linear(e)
                    if not team_id:
                        raise typer.BadParameter(
                            f"Linear issue {issue_id} is missing a team id"
                        )
                    description = brief if brief is not None else existing.description
                    try:
                        wrote = False
                        needs_update = (
                            existing.title != desired_title
                            or (existing.description or "") != (description or "")
                            or (existing.state_type or "").lower() != state_type.lower()
                        )
                        if milestone is not None:
                            current_mid = await fetch_issue_project_milestone_id(
                                client, issue_id=issue_id
                            )
                            if current_mid != milestone.id:
                                needs_update = True
                        issue = existing
                        if needs_update:
                            issue = await update_issue(
                                client,
                                issue_id=issue_id,
                                title=desired_title,
                                description=description,
                                project_milestone_id=(
                                    milestone.id if milestone is not None else None
                                ),
                                state_id=await _state_id(
                                    team_id=team_id, state_type=state_type
                                ),
                            )
                            wrote = True
                        if (
                            defaults.label_id is not None
                            and defaults.label_id not in existing.label_ids
                        ):
                            await ensure_issue_has_label(
                                client,
                                issue_id=issue_id,
                                label_id=defaults.label_id,
                            )
                            wrote = True
                    except LinearApiError as e:
                        _raise_linear(e)
                    if wrote:
                        stats.issues_updated += 1
                else:
                    try:
                        issue = await create_issue(
                            client,
                            team_id=defaults.team_id,
                            project_id=project_id,
                            project_milestone_id=(
                                milestone.id if milestone is not None else None
                            ),
                            title=desired_title,
                            description=brief,
                            label_ids=(
                                [defaults.label_id]
                                if defaults.label_id is not None
                                else None
                            ),
                            state_id=await _state_id(
                                team_id=defaults.team_id, state_type=state_type
                            ),
                        )
                    except LinearApiError as e:
                        _raise_linear(e)
                    stats.issues_created += 1

                if task_row.linear_issue_id != issue.id:
                    task_row.linear_issue_id = issue.id
                if task_row.linear_identifier != issue.identifier:
                    task_row.linear_identifier = issue.identifier
                task_row.linear_state_type = issue.state_type
                task_row.linear_state_name = issue.state_name
                task_row.linear_state_observed_at = datetime.now(UTC)

                markdown = readme.read_text(encoding="utf-8")
                existing_meta = _load_metadata_dict(markdown)

                linear_meta = existing_meta.get("linear")
                if not isinstance(linear_meta, dict):
                    linear_meta = {}
                    existing_meta["linear"] = linear_meta
                linear_meta["issue_id"] = issue.id
                linear_meta["identifier"] = issue.identifier

                next_markdown = upsert_metadata_yaml(markdown, yaml_data=existing_meta)
                if next_markdown != markdown:
                    readme.write_text(next_markdown, encoding="utf-8")
                    stats.docs_updated += 1

            await session.commit()

    finally:
        await engine.dispose()

    # Ensure the epic project id is persisted in the epic doc for deterministic
    # bootstrap on a new machine.
    if not dry_run:
        epic_readme = epic_dir / "README.md"
        if epic_readme.exists():
            markdown = epic_readme.read_text(encoding="utf-8")
            yaml_data = _load_metadata_dict(markdown)
            yaml_data["slug"] = epic_row.slug
            yaml_data["name"] = epic_row.name
            yaml_data["root_branch"] = epic_row.root_branch
            linear_meta = yaml_data.get("linear")
            if not isinstance(linear_meta, dict):
                linear_meta = {}
                yaml_data["linear"] = linear_meta
            linear_meta["project_id"] = project_id
            epic_readme.write_text(
                upsert_metadata_yaml(markdown, yaml_data=yaml_data), encoding="utf-8"
            )

    # Second pass: after all issues exist, push dependency edges.
    ref_to_issue_id: dict[str, str] = {}
    issue_id_by_path: dict[Path, str] = {}
    for readme, doc in task_docs:
        meta = doc.metadata
        local_ref = _task_ref_from_doc(readme, meta)
        markdown = readme.read_text(encoding="utf-8")
        meta_dict = _load_metadata_dict(markdown)
        linear_meta = meta_dict.get("linear")
        if not isinstance(linear_meta, dict):
            continue
        issue_id = linear_meta.get("issue_id")
        identifier = linear_meta.get("identifier")
        if not isinstance(issue_id, str) or not issue_id:
            continue
        issue_id_by_path[readme] = issue_id
        for ref in [local_ref, identifier, issue_id]:
            if not isinstance(ref, str) or not ref:
                continue
            existing = ref_to_issue_id.get(ref)
            if existing is not None and existing != issue_id:
                raise typer.BadParameter(
                    f"Ambiguous task ref {ref!r}; matches multiple tasks in docs (v0)"
                )
            ref_to_issue_id[ref] = issue_id

    for readme, doc in task_docs:
        markdown = readme.read_text(encoding="utf-8")
        meta_dict = _load_metadata_dict(markdown)
        try:
            meta_parsed = TaskMetadata.model_validate(meta_dict)
        except Exception as e:
            raise typer.BadParameter(
                f"Invalid task doc metadata in {readme}: {e}"
            ) from e
        issue_id = issue_id_by_path.get(readme)
        if issue_id is None:
            continue

        refs: list[str] = []
        if meta_parsed.stacked_on:
            refs.append(meta_parsed.stacked_on)
        refs.extend(meta_parsed.must_land_after)

        deduped: list[str] = []
        seen = set()
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            deduped.append(ref)

        blocker_ids: list[str] = []
        unresolved: list[str] = []
        for ref in deduped:
            resolved = ref_to_issue_id.get(ref)
            if resolved is not None:
                blocker_ids.append(resolved)
                continue
            if _looks_like_uuid(ref):
                blocker_ids.append(ref)
                continue
            unresolved.append(ref)

        if unresolved:
            stats.blockers_skipped += 1
            typer.echo(
                f"warning: skipping dependency push for {readme.parent.name}; "
                f"unresolved refs: {', '.join(unresolved)}",
                err=True,
            )
            continue

        try:
            await set_issue_blockers(
                client, issue_id=issue_id, blocker_issue_ids=blocker_ids
            )
        except LinearApiError as e:
            _raise_linear(e)
        stats.blockers_updated += 1

    return stats


def _default_worktree_path(ctx: RepoContext, *, branch: str) -> Path:
    safe_parts = []
    for part in branch.split("/"):
        if part in {"", ".", ".."}:
            continue
        safe_parts.append(part.replace(":", "_"))
    safe_rel = Path(*safe_parts) if safe_parts else Path(branch.replace(":", "_"))
    return ctx.state_dir / "worktrees" / safe_rel


def _git_cwd_from_args(base_cwd: Path, git_args: list[str]) -> Path:
    cwd = base_cwd
    i = 0
    while i < len(git_args):
        arg = git_args[i]
        if arg == "-C" and i + 1 < len(git_args):
            cwd = (cwd / git_args[i + 1]).resolve()
            i += 2
            continue
        if arg == "-c" and i + 1 < len(git_args):
            i += 2
            continue
        if arg in {"--git-dir", "--work-tree"} and i + 1 < len(git_args):
            i += 2
            continue
        if arg.startswith("-"):
            i += 1
            continue
        break
    return cwd


async def _load_block_summary(ctx: RepoContext, *, branch: str | None) -> str | None:
    try:
        block = await get_effective_block(
            ctx,
            branch=branch,
            policy=BlockPolicy.GitMutations,
        )
    except NotInitializedError:
        return None

    if block is None:
        return None

    reason = block.reason or "n/a"
    mode = block.mode.value if hasattr(block.mode, "value") else block.mode
    return f"{mode} (scope={block.scope}, reason={reason})"


@app.command()
def status(
    cwd: Path | None = typer.Option(None, help="Run from this directory."),
) -> None:
    """Show current repo orchestration status."""
    try:
        ctx = get_repo_context(cwd=cwd)
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    repo = asyncio.run(_load_repository_row(ctx))
    branch = current_branch(cwd=cwd or Path.cwd())
    block_summary = asyncio.run(_load_block_summary(ctx, branch=branch))
    typer.echo(f"Worktree: {ctx.worktree_root}")
    typer.echo(f"Repo: {ctx.repo_root}")
    typer.echo(f"State: {ctx.state_dir}")
    typer.echo(f"DB: {ctx.db_path}")
    if repo is None:
        typer.echo("Initialized: no")
    else:
        typer.echo("Initialized: yes")
        typer.echo(f"Default branch: {repo.default_branch}")
    typer.echo(f"Block (git): {block_summary or 'none'}")


def _with_prepend_pythonpath(env: dict[str, str], path: Path) -> dict[str, str]:
    value = str(path)
    if old := env.get("PYTHONPATH"):
        env = env.copy()
        env["PYTHONPATH"] = f"{value}{os.pathsep}{old}"
        return env
    env = env.copy()
    env["PYTHONPATH"] = value
    return env


def _terminate_process(
    proc: subprocess.Popen[str] | subprocess.Popen[bytes], sig: signal.Signals
) -> None:
    if proc.poll() is not None:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, sig)
            return
        except ProcessLookupError:
            return
        except Exception:
            pass
    try:
        proc.send_signal(sig)
    except ProcessLookupError:
        return


@app.command()
def dev(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(9234, help="Dashboard dev server port."),
    api_port: int = typer.Option(9235, help="Daemon bind port."),
    reload: bool = typer.Option(True, help="Auto-reload on code changes."),
) -> None:
    """Run dashboard HMR + daemon reload on a single origin."""
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if not shutil.which("npm"):
        typer.echo(
            "error: npm not found. Install Node.js and npm, then run "
            "`cd dashboard && npm install`.",
            err=True,
        )
        raise typer.Exit(2)

    if port == api_port:
        typer.echo("error: --port and --api-port must be different", err=True)
        raise typer.Exit(2)

    # Migrate the local DB before starting the daemon, since `redesmyn.api` does not auto-upgrade.
    try:
        asyncio.run(init_repo(ctx, migrate=True))
    except Exception as e:
        typer.echo(f"error: failed to migrate DB ({e})", err=True)
        raise typer.Exit(2)

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "redesmyn.api:app",
        "--host",
        host,
        "--port",
        str(api_port),
    ]
    if reload:
        backend_cmd.extend(
            [
                "--reload",
                "--reload-dir",
                str(ctx.worktree_root / "redesmyn"),
            ]
        )

    dashboard_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        host,
        "--port",
        str(port),
        "--strictPort",
    ]

    backend_env = _with_prepend_pythonpath(os.environ.copy(), ctx.worktree_root)
    backend_env.setdefault("REDESMYN_REPO_ROOT", str(ctx.repo_root))
    backend_env.setdefault("REDESMYN_WORKTREE_ROOT", str(ctx.worktree_root))
    backend_env.setdefault("REDESMYN_DB_PATH", str(ctx.db_path))
    dashboard_env = os.environ.copy()
    dashboard_env.setdefault("REDESMYN_DAEMON_ORIGIN", f"http://{host}:{api_port}")
    daemon_origin = dashboard_env["REDESMYN_DAEMON_ORIGIN"]

    typer.echo(f"Dashboard: http://{host}:{port}/")
    typer.echo(f"API:       http://{host}:{port}/v1/ (proxied to {daemon_origin})")

    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(ctx.worktree_root),
        env=backend_env,
        start_new_session=True,
    )
    dashboard_proc = subprocess.Popen(
        dashboard_cmd,
        cwd=str(ctx.worktree_root / "dashboard"),
        env=dashboard_env,
        start_new_session=True,
    )

    procs: list[tuple[str, subprocess.Popen[str] | subprocess.Popen[bytes]]] = [
        ("daemon", backend_proc),
        ("dashboard", dashboard_proc),
    ]

    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is None:
                    continue
                typer.echo(f"{name} exited ({code}); stopping…", err=True)
                for _, other in procs:
                    _terminate_process(other, signal.SIGTERM)
                for _, other in procs:
                    try:
                        other.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        _terminate_process(other, kill_signal)
                raise typer.Exit(code)
            time.sleep(0.2)
    except KeyboardInterrupt:
        typer.echo("Stopping…", err=True)
        for _, proc in procs:
            _terminate_process(proc, signal.SIGTERM)
        for _, proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate_process(proc, kill_signal)
        raise typer.Exit(130) from None


@daemon_app.command("run")
def daemon_run(
    control_plane: str | None = typer.Option(
        None, "--control-plane", help="Control plane origin (http(s)://...)."
    ),
    token: str | None = typer.Option(
        None, "--token", help="Daemon auth token (defaults from settings)."
    ),
    workspace_id: str = typer.Option(
        "default", "--workspace-id", help="Workspace id for telemetry attribution."
    ),
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        help="Repo id for telemetry attribution (defaults from repo).",
    ),
    poll_interval: float = typer.Option(
        1.0, "--poll-interval", help="Telemetry poll interval (seconds)."
    ),
    heartbeat_interval: float = typer.Option(
        5.0, "--heartbeat-interval", help="Heartbeat interval (seconds)."
    ),
) -> None:
    """Run the daemon in the foreground."""
    try:
        _ = get_repo_context()
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    from redesmyn.daemon_runtime import DaemonRuntimeConfig, run_daemon
    from redesmyn.settings import load_settings

    ctx = get_repo_context()
    settings = load_settings(repo_root=ctx.repo_root)

    inferred_workspace_id = workspace_id
    inferred_repo_id = repo_id
    if inferred_repo_id is None:
        inferred_repo_id = compute_repo_id(ctx.repo_root)
        try:

            async def _read_repo_key() -> tuple[str, str] | None:
                engine = create_engine(ctx.db_path)
                try:
                    sessionmaker = create_sessionmaker(engine)
                    async with sessionmaker() as session:
                        row = await session.scalar(
                            select(Repository).where(
                                Repository.repo_root == str(ctx.repo_root)
                            )
                        )
                        if row is None:
                            return None
                        return row.workspace_id, row.repo_id
                finally:
                    await engine.dispose()

            key = asyncio.run(_read_repo_key())
        except Exception:
            key = None
        if key is not None:
            db_workspace_id, db_repo_id = key
            inferred_repo_id = db_repo_id
            if workspace_id == "default":
                inferred_workspace_id = db_workspace_id
    cfg = DaemonRuntimeConfig(
        control_plane_url=(
            control_plane
            or os.environ.get("REDESMYN_CONTROL_PLANE_ORIGIN")
            or f"http://{settings.api_host}:{settings.api_port}"
        ),
        token=token or settings.daemon_auth_token,
        workspace_id=inferred_workspace_id,
        repo_id=inferred_repo_id,
        poll_interval_s=poll_interval,
        heartbeat_interval_s=heartbeat_interval,
    )

    try:
        asyncio.run(run_daemon(ctx, settings, cfg))
    except KeyboardInterrupt:
        raise typer.Exit(130) from None


@server_app.command("run")
def server_run(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(9234, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes."),
    observer: bool = typer.Option(
        True,
        "--observer/--no-observer",
        help="Run the repo observer in the server process (debug only).",
    ),
) -> None:
    """Run the control plane server in the foreground."""
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    try:
        import uvicorn
    except Exception as e:  # pragma: no cover
        typer.echo(f"error: uvicorn not available ({e})", err=True)
        raise typer.Exit(2)

    # Migrate the local DB before starting the daemon, since `redesmyn.api` does not auto-upgrade.
    try:
        asyncio.run(init_repo(ctx, migrate=True))
    except Exception as e:
        typer.echo(f"error: failed to migrate DB ({e})", err=True)
        raise typer.Exit(2)

    os.environ.setdefault("REDESMYN_REPO_ROOT", str(ctx.repo_root))
    os.environ.setdefault("REDESMYN_WORKTREE_ROOT", str(ctx.worktree_root))
    os.environ.setdefault("REDESMYN_DB_PATH", str(ctx.db_path))

    if observer:
        os.environ.pop("REDESMYN_NO_OBSERVER", None)
    else:
        os.environ["REDESMYN_NO_OBSERVER"] = "1"

    uvicorn.run("redesmyn.api:app", host=host, port=port, reload=reload)


@daemon_app.command("start")
def daemon_start() -> None:
    typer.echo("Not implemented yet. Use `rn daemon run` for now.", err=True)
    raise typer.Exit(2)


@daemon_app.command("up")
def daemon_up() -> None:
    daemon_start()


@daemon_app.command("stop")
def daemon_stop() -> None:
    typer.echo("Not implemented yet.", err=True)
    raise typer.Exit(2)


@daemon_app.command("down")
def daemon_down() -> None:
    daemon_stop()


@daemon_app.command("status")
def daemon_status() -> None:
    typer.echo("Not implemented yet. Use `rn status` for now.", err=True)
    raise typer.Exit(2)


app.add_typer(daemon_app, name="daemon")
app.add_typer(server_app, name="server")
app.add_typer(config_app, name="config")
app.add_typer(sandbox_app, name="sandbox")


@observer_app.command("run")
def observer_run(
    interval: float = typer.Option(1.0, "--interval", help="Poll interval (seconds)."),
    emit_baseline: bool = typer.Option(
        False,
        "--emit-baseline/--no-emit-baseline",
        help="Emit events on first observation (defaults to no).",
    ),
    once: bool = typer.Option(False, "--once", help="Run one poll and exit."),
) -> None:
    """Run the host-local repo observer in the foreground."""
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if interval <= 0:
        raise typer.BadParameter("--interval must be > 0")

    try:
        asyncio.run(
            run_repo_observer(
                ctx,
                interval_s=interval,
                emit_baseline=emit_baseline,
                once=once,
            )
        )
    except KeyboardInterrupt:
        raise typer.Exit(130) from None


app.add_typer(observer_app, name="observer")


@epic_app.command("create")
def epic_create(
    name: str = typer.Option(..., help="Epic name."),
    slug: str | None = typer.Option(None, help="Epic slug (defaults from name)."),
    root_branch: str | None = typer.Option(
        None, help="Epic root branch (defaults to repo default)."
    ),
    linear_project_id: str | None = typer.Option(
        None, help="Linear project id (optional)."
    ),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    slug_value = slug or slugify(name, fallback="epic")

    async def _run() -> Epic:
        repo = await _require_repo_row(ctx)

        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                existing = await session.scalar(
                    select(Epic).where(
                        Epic.repository_id == repo.id,
                        Epic.slug == slug_value,
                    )
                )
                if existing is not None:
                    return existing

                epic_row = Epic(
                    repository_id=repo.id,
                    name=name,
                    slug=slug_value,
                    root_branch=root_branch or repo.default_branch,
                    linear_project_id=linear_project_id,
                )
                session.add(epic_row)
                await session.commit()
                await session.refresh(epic_row)
                return epic_row
        finally:
            await engine.dispose()

    epic_row = asyncio.run(_run())
    typer.echo(f"Epic: {epic_row.id} {epic_row.slug} (root={epic_row.root_branch})")


@epic_app.command("list")
def epic_list() -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> list[Epic]:
        repo = await _require_repo_row(ctx)

        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                rows = await session.scalars(
                    select(Epic).where(Epic.repository_id == repo.id).order_by(Epic.id)
                )
                return list(rows)
        finally:
            await engine.dispose()

    epics = asyncio.run(_run())
    if not epics:
        typer.echo("No epics.")
        return
    for e in epics:
        typer.echo(f"{e.id}: {e.slug} name={e.name} root={e.root_branch}")


app.add_typer(epic_app, name="epic")


@task_app.command("add")
def task_add(
    title: str = typer.Option(..., help="Task title."),
    body: str | None = typer.Option(None, help="Task body/description."),
    epic: str | None = typer.Option(
        None, help="Epic slug or id (defaults if only one epic)."
    ),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> Task:
        epic_row = await _resolve_epic(ctx, epic=epic)
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                task = Task(epic_id=epic_row.id, title=title, body=body)
                session.add(task)
                await session.commit()
                await session.refresh(task)
                return task
        finally:
            await engine.dispose()

    task = asyncio.run(_run())
    typer.echo(f"Task: {task.id} {task.title}")


@task_app.command("list")
def task_list(
    epic: str | None = typer.Option(
        None, help="Epic slug or id (defaults if only one epic)."
    ),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> list[Task]:
        epic_row = await _resolve_epic(ctx, epic=epic)
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                rows = await session.scalars(
                    select(Task).where(Task.epic_id == epic_row.id).order_by(Task.id)
                )
                return list(rows)
        finally:
            await engine.dispose()

    tasks = asyncio.run(_run())
    if not tasks:
        typer.echo("No tasks.")
        return
    for t in tasks:
        branch = f" branch={t.branch_name}" if t.branch_name else ""
        parent = f" parent={t.parent_task_id}" if t.parent_task_id else ""
        typer.echo(f"{t.id}:{branch}{parent} {t.title}")


app.add_typer(task_app, name="task")


@agent_app.command("start")
def agent_start(
    task_id: int = typer.Option(..., "--task", help="Task id."),
    harness: str | None = typer.Option(
        None,
        "--harness",
        help="Harness command (shell-like). Defaults from config.harness.command.",
    ),
    agent_kind: AgentKindSelection | None = typer.Option(
        None,
        "--agent-kind",
        help=(
            "Agent kind selection (auto/generic/codex/claude_code). "
            "Defaults from config.harness.agent_kind."
        ),
    ),
    prelude: str | None = typer.Option(
        None,
        "--prelude",
        help="One-time prelude message to send for this start only (supports placeholders like {task_id}).",
    ),
    detach: bool | None = typer.Option(
        None,
        "--detach/--no-detach",
        help="Run in a detached session (tmux if available). Defaults from config.harness.detach.",
    ),
) -> None:
    """Start the per-task agent (tmux-first)."""
    try:
        repo_ctx = get_repo_context()
        _ensure_initialized(repo_ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    defaults = load_orchestration_defaults(repo_ctx)
    effective_harness = harness or defaults.harness.command
    if effective_harness is None:
        typer.echo(
            "error: Missing --harness (or set config.harness.command via `rn config set harness.command …`)",
            err=True,
        )
        raise typer.Exit(2)
    effective_detach = detach if detach is not None else defaults.harness.detach
    effective_agent_kind = (
        agent_kind if agent_kind is not None else defaults.harness.agent_kind
    )

    try:
        result = asyncio.run(
            start_task_agent(
                repo_ctx,
                task_id=task_id,
                harness_command=effective_harness,
                detach=effective_detach,
                prelude_override=prelude,
                agent_kind_selection=effective_agent_kind,
            )
        )
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    verb = "Started" if result.started else "Already running"
    typer.echo(f"{verb}: agent a-{task_id} (attach={result.attach.type})")
    for warning in result.warnings:
        typer.echo(f"warning: {warning}", err=True)

    typer.echo(f"Attach:  rn agent attach --task {task_id}")
    typer.echo(f"Logs:    rn agent logs --task {task_id}")
    typer.echo(f"Worktree: rn shell --task-id {task_id}")


@agent_app.command("restart")
def agent_restart(
    task_id: int = typer.Option(..., "--task", help="Task id."),
    harness: str | None = typer.Option(
        None,
        "--harness",
        help="Harness command (shell-like). Defaults to the last known command for this task.",
    ),
    agent_kind: AgentKindSelection | None = typer.Option(
        None,
        "--agent-kind",
        help=(
            "Agent kind selection (auto/generic/codex/claude_code). "
            "Defaults to the last known selection for this task."
        ),
    ),
    prelude: str | None = typer.Option(
        None,
        "--prelude",
        help="One-time prelude message to send for this restart only (supports placeholders like {task_id}).",
    ),
    detach: bool | None = typer.Option(
        None,
        "--detach/--no-detach",
        help="Run in a detached session (tmux if available). Defaults from config.harness.detach.",
    ),
) -> None:
    """Restart the per-task agent."""
    try:
        repo_ctx = get_repo_context()
        _ensure_initialized(repo_ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    defaults = load_orchestration_defaults(repo_ctx)
    effective_detach = detach if detach is not None else defaults.harness.detach
    effective_harness = harness if harness is not None else defaults.harness.command

    try:
        result = asyncio.run(
            restart_task_agent(
                repo_ctx,
                task_id=task_id,
                harness_command=effective_harness,
                detach=effective_detach,
                prelude_override=prelude,
                agent_kind_selection_override=agent_kind,
                default_agent_kind_selection=defaults.harness.agent_kind,
            )
        )
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    typer.echo(f"Restarted: agent a-{task_id} (attach={result.attach.type})")
    for warning in result.warnings:
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(f"Attach:  rn agent attach --task {task_id}")
    typer.echo(f"Logs:    rn agent logs --task {task_id}")
    typer.echo(f"Worktree: rn shell --task-id {task_id}")


def _attach_agent_for_task_id(repo_ctx: RepoContext, *, task_id: int) -> None:
    try:
        agent_session_row = asyncio.run(
            load_task_agent_session(repo_ctx, task_id=task_id)
        )
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if agent_session_row is None:
        typer.echo("No running agent for this task.", err=True)
        raise typer.Exit(1)

    try:
        code = attach_agent_session(agent_session_row=agent_session_row)
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        log_path = agent_log_path_for_session_row(
            repo_ctx, agent_session_row=agent_session_row
        )
        typer.echo(f"Logs: rn agent logs --task {task_id}  (path: {log_path})")
        raise typer.Exit(2)
    raise typer.Exit(code)


@agent_app.command("attach")
def agent_attach(
    task_id: int | None = typer.Option(
        None,
        "--task",
        help=(
            "Task id (DB primary key). Defaults from RN_TASK_ID (rn shell) or the "
            "current git branch (when run from a task worktree)."
        ),
    ),
) -> None:
    """Attach to a detached agent session (tmux)."""
    try:
        repo_ctx = get_repo_context()
        _ensure_initialized(repo_ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    engine = create_engine(repo_ctx.db_path)
    sessionmaker = create_sessionmaker(engine)
    try:
        resolved_task_id = asyncio.run(
            _resolve_task_id_for_current_context(
                sessionmaker=sessionmaker,
                explicit_task_id=task_id,
            )
        )
    except MergePlanError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    finally:
        asyncio.run(engine.dispose())

    _attach_agent_for_task_id(repo_ctx, task_id=resolved_task_id)


@agent_app.command("stop")
def agent_stop(
    task_id: int = typer.Option(..., "--task", help="Task id."),
) -> None:
    """Stop a runner-owned agent session for this task."""
    try:
        repo_ctx = get_repo_context()
        _ensure_initialized(repo_ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    try:
        stopped = asyncio.run(stop_task_agent(repo_ctx, task_id=task_id))
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if not stopped:
        typer.echo("No running agent for this task.")
        return
    typer.echo(f"Stopped: agent a-{task_id}")


@agent_app.command("logs")
def agent_logs(
    task_id: int = typer.Option(..., "--task", help="Task id."),
    lines: int = typer.Option(200, "--lines", help="Lines of history to show."),
    follow: bool = typer.Option(
        True,
        "--follow/--no-follow",
        help="Follow log output (tail -f).",
    ),
) -> None:
    """Tail agent logs (tmux pipe-pane or runner log file)."""
    try:
        repo_ctx = get_repo_context()
        _ensure_initialized(repo_ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    try:
        agent_session_row = asyncio.run(
            load_task_agent_session(repo_ctx, task_id=task_id, active_only=False)
        )
    except RuntimeError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if agent_session_row is None:
        typer.echo("No agent found for this task.", err=True)
        raise typer.Exit(1)

    log_path = agent_log_path_for_session_row(
        repo_ctx, agent_session_row=agent_session_row
    )
    if not log_path.exists():
        typer.echo(f"No log file found: {log_path}", err=True)
        raise typer.Exit(1)

    cmd = ["tail", "-n", str(lines)]
    if follow:
        cmd.append("-f")
    cmd.append(str(log_path))
    proc = subprocess.run(cmd)
    raise typer.Exit(proc.returncode)


app.add_typer(agent_app, name="agent")


async def _load_linear_auth(ctx: RepoContext) -> LinearAuth | None:
    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            return await session.scalar(
                select(LinearAuth).order_by(desc(LinearAuth.id)).limit(1)
            )
    finally:
        await engine.dispose()


async def _maybe_migrate_repo_linear_auth_to_keychain(
    ctx: RepoContext,
) -> bool:
    store = default_linear_credential_store()
    if store.get() is not None:
        return False

    engine = create_engine(ctx.db_path)
    try:
        sessionmaker = create_sessionmaker(engine)
        async with sessionmaker() as session:
            auth = await session.scalar(
                select(LinearAuth).order_by(desc(LinearAuth.id)).limit(1)
            )
            if auth is None:
                return False

            store.set(
                LinearCredentials(
                    access_token=auth.access_token,
                    refresh_token=auth.refresh_token,
                    token_type=auth.token_type,
                    scope=auth.scope,
                    expires_at=auth.expires_at,
                    connected_at=auth.created_at,
                )
            )

            # One-way migration: once creds are in the machine store, remove the
            # repo-scoped record so `rn linear logout` actually disconnects.
            await session.execute(delete(LinearAuth))
            await session.commit()
            return True
    finally:
        await engine.dispose()


def _load_settings_for_linear(ctx: RepoContext | None) -> RedesmynSettings:
    from redesmyn.settings import load_settings

    return load_settings(repo_root=ctx.repo_root if ctx else None)


async def _require_fresh_linear_credentials(
    *,
    ctx: RepoContext | None,
    skew: timedelta = timedelta(minutes=5),
) -> LinearCredentials:
    store = default_linear_credential_store()

    if store.get() is None and ctx is not None and ctx.db_path.exists():
        await _maybe_migrate_repo_linear_auth_to_keychain(ctx)

    creds = store.get()
    if creds is None:
        raise typer.BadParameter("Linear is not connected. Run `rn linear auth`.")

    settings = _load_settings_for_linear(ctx)
    if is_expiring_soon(creds, skew=skew):
        if not creds.refresh_token:
            raise typer.BadParameter(
                "Linear access token is expired/expiring and no refresh token is available. Run `rn linear auth`."
            )
        try:
            token = await refresh_access_token(
                settings, refresh_token=creds.refresh_token
            )
        except Exception as e:
            raise typer.BadParameter(
                f"Could not refresh Linear access token: {e}"
            ) from e
        next_creds = LinearCredentials(
            access_token=token.access_token,
            refresh_token=token.refresh_token or creds.refresh_token,
            token_type=token.token_type,
            scope=token.scope or creds.scope,
            expires_at=token.expires_at,
            connected_at=creds.connected_at,
        )
        store.set(next_creds)
        return next_creds

    return creds


@linear_app.command("status")
def linear_status(
    whoami: bool = typer.Option(
        False, "--whoami", help="Fetch and print account info."
    ),
) -> None:
    ctx: RepoContext | None
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError:
        ctx = None

    if ctx is not None and not ctx.db_path.exists():
        ctx = None

    try:
        creds = asyncio.run(_require_fresh_linear_credentials(ctx=ctx))
    except typer.BadParameter:
        typer.echo("Linear: not connected")
        raise typer.Exit(1)

    typer.echo("Linear: connected")
    typer.echo(f"Connected at: {creds.connected_at.isoformat()}")
    if creds.scope:
        typer.echo(f"Scopes: {creds.scope}")

    if whoami:

        async def _run() -> None:
            client = LinearClient(access_token=creds.access_token)
            data = await client.graphql("query { viewer { id name email } }")
            typer.echo(str(data.get("viewer")))

        asyncio.run(_run())


@linear_app.command("auth")
def linear_auth(
    timeout_seconds: int = typer.Option(180, help="Max time to wait for auth."),
) -> None:
    ctx: RepoContext | None
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError:
        ctx = None

    if ctx is not None and not ctx.db_path.exists():
        ctx = None

    settings = _load_settings_for_linear(ctx)
    if not settings.linear_client_id:
        redirect_uri = linear_redirect_uri(settings)
        typer.echo(
            "error: Linear OAuth is not configured. Create a Linear OAuth app and set:\n"
            "  REDESMYN_LINEAR_CLIENT_ID\n"
            "in `.env` (see `.env.example`).\n"
            f"Redirect URL: {redirect_uri}",
            err=True,
        )
        raise typer.Exit(2)

    redirect_uri = linear_redirect_uri(settings)
    state = new_oauth_state()
    code_verifier = new_pkce_verifier()
    code_challenge = pkce_code_challenge(code_verifier)
    authorize_url = linear_authorize_url(
        settings, state=state, redirect_uri=redirect_uri, code_challenge=code_challenge
    )

    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse
    import threading

    callback: dict[str, str | None] = {"code": None, "state": None, "error": None}
    got_callback = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/v1/linear/oauth/callback":
                self.send_response(404)
                self.end_headers()
                return

            qs = parse_qs(parsed.query)
            callback["code"] = (qs.get("code") or [None])[0]
            callback["state"] = (qs.get("state") or [None])[0]
            callback["error"] = (qs.get("error") or [None])[0]
            got_callback.set()

            if callback["error"]:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(
                    b"<h1>Linear auth failed</h1><p>Return to the terminal for details.</p>"
                )
                return

            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"<h1>Linear connected</h1><p>You can close this tab and return to Redesmyn.</p>"
            )

    host = settings.api_host
    port = settings.api_port
    try:
        httpd = HTTPServer((host, port), _Handler)
    except OSError as listen_error:
        listen_error_msg = str(listen_error)
        base = f"http://{host}:{port}"
        start_url = f"{base}/v1/linear/oauth/start"
        status_url = f"{base}/v1/linear/status"

        import httpx

        async def _poll() -> None:
            loop = asyncio.get_running_loop()
            async with httpx.AsyncClient(timeout=5.0) as client:
                start = loop.time()
                while True:
                    if loop.time() - start > timeout_seconds:
                        raise typer.BadParameter(
                            "Timed out waiting for Linear authorization"
                        )

                    try:
                        resp = await client.get(status_url)
                    except httpx.RequestError:
                        raise typer.BadParameter(
                            f"Could not listen on {host}:{port} for OAuth callback ({listen_error_msg}). "
                            "Stop the process using that port, or run the Redesmyn API on that port, or change "
                            "REDESMYN_API_HOST/REDESMYN_API_PORT and update your Linear OAuth redirect URL."
                        ) from None

                    if resp.status_code == 404:
                        raise typer.BadParameter(
                            f"Could not listen on {host}:{port} for OAuth callback ({listen_error_msg}). "
                            f"Also, {status_url} returned 404 (this does not look like the Redesmyn API). "
                            "Stop the process using that port or change REDESMYN_API_HOST/REDESMYN_API_PORT "
                            "and update your Linear OAuth redirect URL."
                        ) from None

                    if resp.status_code != 200:
                        await asyncio.sleep(1.0)
                        continue

                    try:
                        payload = resp.json()
                    except ValueError:
                        raise typer.BadParameter(
                            f"Could not listen on {host}:{port} for OAuth callback ({listen_error_msg}). "
                            f"Also, {status_url} did not return JSON (this does not look like the Redesmyn API). "
                            "Stop the process using that port or change REDESMYN_API_HOST/REDESMYN_API_PORT "
                            "and update your Linear OAuth redirect URL."
                        ) from None

                    if not isinstance(payload, dict) or "connected" not in payload:
                        raise typer.BadParameter(
                            f"Could not listen on {host}:{port} for OAuth callback ({listen_error_msg}). "
                            f"Also, {status_url} returned an unexpected payload (this does not look like the Redesmyn API). "
                            "Stop the process using that port or change REDESMYN_API_HOST/REDESMYN_API_PORT "
                            "and update your Linear OAuth redirect URL."
                        ) from None

                    if payload.get("connected"):
                        return
                    await asyncio.sleep(1.0)

        if not webbrowser.open(start_url):
            typer.echo(start_url)

        try:
            asyncio.run(_poll())
        except typer.BadParameter as poll_error:
            typer.echo(f"error: {poll_error}", err=True)
            raise typer.Exit(2)

        typer.echo("Linear: connected")
        return

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        if not webbrowser.open(authorize_url):
            typer.echo(authorize_url)

        if not got_callback.wait(timeout_seconds):
            raise typer.BadParameter("Timed out waiting for Linear authorization")

        if callback["error"]:
            raise typer.BadParameter(
                f"Linear authorization failed: {callback['error']}"
            )

        code = callback["code"]
        returned_state = callback["state"]
        if not code or not returned_state:
            raise typer.BadParameter("Missing code/state in Linear callback")
        if returned_state != state:
            raise typer.BadParameter("State mismatch in Linear callback")

        try:
            token = asyncio.run(
                exchange_code_for_token(
                    settings,
                    code=code,
                    redirect_uri=redirect_uri,
                    code_verifier=code_verifier,
                )
            )
        except Exception as e:
            raise typer.BadParameter(
                f"Could not complete Linear OAuth token exchange: {e}"
            ) from e
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    finally:
        httpd.shutdown()
        httpd.server_close()

    store = default_linear_credential_store()
    store.set(
        LinearCredentials(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            token_type=token.token_type,
            scope=token.scope,
            expires_at=token.expires_at,
            connected_at=datetime.now(UTC),
        )
    )

    typer.echo("Linear: connected")


@linear_app.command("logout")
def linear_logout() -> None:
    store = default_linear_credential_store()
    store.clear()

    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError:
        ctx = None

    if ctx is not None and ctx.db_path.exists():

        async def _purge_legacy() -> None:
            engine = create_engine(ctx.db_path)
            try:
                sessionmaker = create_sessionmaker(engine)
                async with sessionmaker() as session:
                    await session.execute(delete(LinearAuth))
                    await session.commit()
            finally:
                await engine.dispose()

        asyncio.run(_purge_legacy())

    typer.echo("Linear: logged out")


@linear_app.command("whoami")
def linear_whoami() -> None:
    ctx: RepoContext | None
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError:
        ctx = None

    if ctx is not None and not ctx.db_path.exists():
        ctx = None

    try:
        creds = asyncio.run(_require_fresh_linear_credentials(ctx=ctx))
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> None:
        client = LinearClient(access_token=creds.access_token)
        data = await client.graphql("query { viewer { id name email } }")
        typer.echo(str(data.get("viewer")))

    asyncio.run(_run())


@linear_app.command("projects")
def linear_projects() -> None:
    """List accessible Linear projects (id, slug, name, teams)."""
    ctx: RepoContext | None
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError:
        ctx = None

    if ctx is not None and not ctx.db_path.exists():
        ctx = None

    try:
        creds = asyncio.run(_require_fresh_linear_credentials(ctx=ctx))
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> None:
        client = LinearClient(access_token=creds.access_token)
        try:
            projects = await fetch_projects(client)
        except Exception as e:
            raise typer.BadParameter(f"Could not list Linear projects: {e}") from e

        typer.echo("id\tslug\tname\tteams")
        for project in sorted(
            projects,
            key=lambda p: ((p.slug or "").lower(), p.name.lower(), p.id),
        ):
            typer.echo(_format_linear_project(project))

    try:
        asyncio.run(_run())
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)


@linear_app.command("milestones")
def linear_milestones(
    project: str = typer.Option(
        ..., "--project", help="Linear project id, slug, or name."
    ),
) -> None:
    """List milestones for a Linear project (id, name)."""
    ctx: RepoContext | None
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError:
        ctx = None

    if ctx is not None and not ctx.db_path.exists():
        ctx = None

    try:
        creds = asyncio.run(_require_fresh_linear_credentials(ctx=ctx))
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> None:
        client = LinearClient(access_token=creds.access_token)
        project_id = await _resolve_linear_project_id(client, project)
        milestones = await fetch_project_milestones(client, project_id=project_id)
        typer.echo("id\tname")
        for milestone in sorted(milestones, key=lambda m: (m.name.lower(), m.id)):
            typer.echo(_format_linear_milestone(milestone))

    try:
        asyncio.run(_run())
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)


def _task_state_from_linear(state_type: str | None) -> TaskState:
    # Back-compat wrapper for older call sites; prefer
    # task_state_from_linear_state_type in new code.
    return task_state_from_linear_state_type(state_type)


@linear_app.command("import")
def linear_import(
    project: str = typer.Option(
        ..., "--project", help="Linear project id, slug, or name."
    ),
    epic: str | None = typer.Option(
        None, help="Epic slug or id (defaults if only one epic)."
    ),
    create_branches: bool = typer.Option(
        True,
        "--create-branches/--no-create-branches",
        help="Create branches/worktrees for imported tasks.",
    ),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    try:
        creds = asyncio.run(_require_fresh_linear_credentials(ctx=ctx))
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    epic_row = asyncio.run(_resolve_epic(ctx, epic=epic))
    client = LinearClient(access_token=creds.access_token)
    try:
        project_id = asyncio.run(_resolve_linear_project_id(client, project))
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _fetch() -> tuple[list[LinearIssue], list[LinearIssueRelation]]:
        issues = await fetch_project_issues(client, project_id=project_id)
        try:
            relations = await fetch_project_issue_relations(
                client, project_id=project_id
            )
        except Exception as e:
            typer.echo(
                f"warning: could not fetch issue relations; parent inference disabled ({e})",
                err=True,
            )
            relations: list[LinearIssueRelation] = []
        return issues, relations

    issues, relations = asyncio.run(_fetch())
    issue_ids = {i.id for i in issues}

    blockers_by_issue: dict[str, list[str]] = {i.id: [] for i in issues}
    for rel in relations:
        rel_type = rel.type.lower()
        if rel_type == "blocks":
            if rel.related_issue_id in blockers_by_issue:
                blockers_by_issue[rel.related_issue_id].append(rel.issue_id)
            continue
        if rel_type in {"blocked_by", "blockedby"}:
            if rel.issue_id in blockers_by_issue:
                blockers_by_issue[rel.issue_id].append(rel.related_issue_id)

    parent_by_issue: dict[str, str | None] = {}
    for issue_id, blockers in blockers_by_issue.items():
        if not blockers:
            parent_by_issue[issue_id] = None
            continue
        if len(blockers) > 1:
            raise typer.BadParameter(
                f"Linear issue {issue_id} has multiple blockers; choose a parent explicitly (v0)"
            )
        parent = blockers[0]
        if parent not in issue_ids:
            raise typer.BadParameter(
                f"Linear issue {issue_id} blocker {parent} is outside the imported project (v0)"
            )
        parent_by_issue[issue_id] = parent

    depth_cache: dict[str, int] = {}

    def _depth(issue_id: str, *, stack: set[str]) -> int:
        if issue_id in depth_cache:
            return depth_cache[issue_id]
        if issue_id in stack:
            raise typer.BadParameter("Cycle detected in Linear blocked-by graph (v0)")
        stack.add(issue_id)
        parent = parent_by_issue.get(issue_id)
        depth = 0 if parent is None else _depth(parent, stack=stack) + 1
        stack.remove(issue_id)
        depth_cache[issue_id] = depth
        return depth

    issues_sorted = sorted(issues, key=lambda issue: _depth(issue.id, stack=set()))

    def _branch_name(identifier: str, title: str) -> str:
        slug = slugify(title, fallback="task")[:60].strip("-") or "task"
        return f"rn/{epic_row.slug}/{identifier}-{slug}"

    async def _apply() -> None:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                epic_db = await session.get(Epic, epic_row.id)
                if epic_db is None:
                    raise typer.BadParameter("Epic not found")
                if epic_db.linear_project_id is None:
                    epic_db.linear_project_id = project
                elif epic_db.linear_project_id != project:
                    raise typer.BadParameter(
                        f"Epic is linked to a different Linear project ({epic_db.linear_project_id})"
                    )

                task_by_issue_id: dict[str, Task] = {}
                observed_at = datetime.now(UTC)
                for issue in issues_sorted:
                    task = await session.scalar(
                        select(Task).where(
                            Task.epic_id == epic_db.id,
                            Task.linear_issue_id == issue.id,
                        )
                    )
                    title = f"{issue.identifier} {issue.title}"
                    if task is None:
                        task = Task(
                            epic_id=epic_db.id,
                            title=title,
                            body=issue.description,
                            source=TaskSource.Linear,
                            authority=TaskAuthority.Linear,
                            state=_task_state_from_linear(issue.state_type),
                            linear_issue_id=issue.id,
                        )
                        session.add(task)
                        await session.flush()
                    else:
                        task.title = title
                        task.body = issue.description
                        task.state = _task_state_from_linear(issue.state_type)

                    task.linear_state_type = issue.state_type
                    task.linear_state_name = issue.state_name
                    task.linear_state_observed_at = observed_at

                    task_by_issue_id[issue.id] = task

                    if not create_branches:
                        continue

                    branch = _branch_name(issue.identifier, issue.title)
                    existing = await session.scalar(
                        select(Task).where(
                            Task.epic_id == epic_db.id,
                            Task.branch_name == branch,
                            Task.id != task.id,
                        )
                    )
                    if existing is not None:
                        raise typer.BadParameter(
                            f"Ambiguous branch {branch!r}; assigned to multiple tasks ({existing.id} and {task.id})"
                        )
                    task.branch_name = branch

                if not create_branches:
                    await session.commit()
                    return

                for issue in issues_sorted:
                    parent_issue_id = parent_by_issue.get(issue.id)
                    task = task_by_issue_id[issue.id]
                    parent_task = (
                        task_by_issue_id[parent_issue_id]
                        if parent_issue_id is not None
                        else None
                    )
                    task.parent_task_id = parent_task.id if parent_task else None

                    base_ref = (
                        parent_task.branch_name
                        if parent_task is not None
                        and parent_task.branch_name is not None
                        else epic_db.root_branch
                    )
                    if task.worktree_path is None:
                        branch = task.branch_name
                        if branch is None:
                            continue
                        worktree_path = _default_worktree_path(ctx, branch=branch)
                        if worktree_path.exists():
                            raise typer.BadParameter(
                                f"Worktree path already exists: {worktree_path}"
                            )
                        try:
                            git_worktree_add(
                                ctx.repo_root,
                                worktree_path=worktree_path,
                                branch_name=branch,
                                base_ref=base_ref,
                            )
                        except GitCommandError as e:
                            raise typer.BadParameter(str(e)) from e
                        task.worktree_path = str(worktree_path)

                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_apply())
    typer.echo(f"Imported {len(issues)} issues into epic {epic_row.slug}")


app.add_typer(linear_app, name="linear")


@block_app.command("lax")
def block_lax(
    scope: str = typer.Option("repo", help="Block scope (v0: use 'repo')."),
    reason: str | None = typer.Option(None, help="Human-readable reason."),
) -> None:
    try:
        repo_ctx = get_repo_context()
        block = asyncio.run(
            set_manual_block(
                repo_ctx,
                scope=scope,
                policy=BlockPolicy.GitMutations,
                mode="lax",
                reason=reason,
            )
        )
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    typer.echo(f"Blocked: {block.mode.value} (scope={block.scope})")


@block_app.command("strict")
def block_strict(
    scope: str = typer.Option("repo", help="Block scope (v0: use 'repo')."),
    reason: str | None = typer.Option(None, help="Human-readable reason."),
) -> None:
    try:
        repo_ctx = get_repo_context()
        block = asyncio.run(
            set_manual_block(
                repo_ctx,
                scope=scope,
                policy=BlockPolicy.GitMutations,
                mode="strict",
                reason=reason,
            )
        )
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    typer.echo(f"Blocked: {block.mode.value} (scope={block.scope})")


@block_app.command("clear")
def block_clear(
    scope: str = typer.Option("repo", help="Block scope (v0: use 'repo')."),
    reason: str | None = typer.Option(None, help="Reason for clearing."),
) -> None:
    try:
        repo_ctx = get_repo_context()
        block = asyncio.run(
            clear_block(
                repo_ctx,
                scope=scope,
                policy=BlockPolicy.GitMutations,
                reason=reason,
            )
        )
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if block is None:
        typer.echo("No active block.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Block cleared: {block.mode.value} (scope={block.scope})")


@block_app.command("list")
def block_list(
    scope: str = typer.Option("repo", help="Block scope (v0: use 'repo')."),
) -> None:
    try:
        repo_ctx = get_repo_context()
        blocks = asyncio.run(
            list_blocks(repo_ctx, scope=scope, policy=BlockPolicy.GitMutations)
        )
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if not blocks:
        typer.echo("No blocks.")
        return

    for b in blocks:
        cleared = "active" if b.cleared_at is None else "cleared"
        typer.echo(
            f"{b.id}: {b.mode.value} {cleared} scope={b.scope} reason={b.reason or 'n/a'}"
        )


app.add_typer(block_app, name="block")


@app.command(
    "git",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def git_proxy(ctx: typer.Context) -> None:
    """Proxy git, enforcing Redesmyn invariants when possible."""
    git_args = list(ctx.args)
    if not git_args:
        git_args = ["--help"]

    base_cwd = Path.cwd()
    git_cwd = _git_cwd_from_args(base_cwd, git_args)

    try:
        repo_ctx = get_repo_context(cwd=git_cwd)
        branch = current_branch(cwd=git_cwd)
        block = asyncio.run(
            get_effective_block(
                repo_ctx,
                branch=branch,
                policy=BlockPolicy.GitMutations,
            )
        )
    except NotInitializedError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if block is not None:
        decision = does_block_git(git_args, mode=block.mode)
        if not decision.allowed:
            reason = block.reason or "n/a"
            typer.echo(
                f"blocked: {decision.reason} (scope={block.scope}, reason={reason})",
                err=True,
            )
            raise typer.Exit(3)

    proc = subprocess.run(["git", *git_args])
    subcommand = detect_git_subcommand(git_args)
    if proc.returncode == 0 and subcommand and subcommand not in READ_ONLY_SUBCOMMANDS:
        try:
            asyncio.run(update_git_projections(repo_ctx))
        except Exception:
            pass
    raise typer.Exit(proc.returncode)


def main() -> None:
    try:
        ctx = get_repo_context()
    except NotAGitRepositoryError:
        ctx = None
    if ctx is not None:
        configure_logging(state_dir=ctx.state_dir)

    try:
        result = typer.main.get_command(app).main(
            args=sys.argv[1:],
            prog_name="rn",
            standalone_mode=False,
        )
        raise SystemExit(result if isinstance(result, int) else 0)
    except BrokenPipeError:
        raise SystemExit(141) from None
    except click.Abort:
        typer.echo("", err=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        typer.echo("", err=True)
        raise SystemExit(130) from None
    except (typer.Exit, click.exceptions.Exit) as e:
        raise SystemExit(e.exit_code) from None
    except click.ClickException as e:
        e.show()
        raise SystemExit(e.exit_code) from None
    except Exception as e:
        if _DEBUG or os.environ.get("REDESMYN_DEBUG") in {"1", "true", "TRUE"}:
            raise

        def _fmt_one(exc: BaseException) -> str:
            message = str(exc).strip()
            return message or exc.__class__.__name__

        lines = [f"error: {_fmt_one(e)}"]
        cause = e.__cause__ or (
            None if getattr(e, "__suppress_context__", False) else e.__context__
        )
        if cause is not None:
            lines.append(f"caused by: {_fmt_one(cause)}")
        lines.append("hint: re-run with --debug for a full traceback")
        for line in lines:
            typer.echo(line, err=True)
        raise SystemExit(1) from None
