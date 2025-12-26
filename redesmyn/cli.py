from __future__ import annotations

import asyncio
import re
import subprocess
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import typer
from sqlalchemy import desc, select

from redesmyn import __version__
from redesmyn.blocks import (
    NotInitializedError,
    clear_block,
    get_effective_block,
    list_blocks,
    set_manual_block,
)
from redesmyn.context import RepoContext, get_repo_context
from redesmyn.db import (
    Agent,
    Epic,
    LinearAuth,
    Node,
    Repository,
    Task,
    create_engine,
    create_sessionmaker,
)
from redesmyn.docs.loader import DocLoadError, load_epic_doc, load_task_doc
from redesmyn.docs.writer import upsert_metadata_yaml, upsert_synced_section
from redesmyn.domain.enums import BlockPolicy, TaskAuthority, TaskSource, TaskState
from redesmyn.integrations.linear import (
    LinearClient,
    LinearIssue,
    LinearIssueRelation,
    fetch_project_issue_relations,
    fetch_project_issues,
)
from redesmyn.git_proxy import does_block_git
from redesmyn.orchestrator import init_repo
from redesmyn.repo import (
    GitCommandError,
    NotAGitRepositoryError,
    current_branch,
    git_worktree_add,
)
from redesmyn.strings import slugify

app = typer.Typer(add_completion=False, help="Redesmyn CLI (`rn`).")
daemon_app = typer.Typer(add_completion=False, help="Daemon management.")
block_app = typer.Typer(
    add_completion=False,
    help="Block controls (use `rn pause` as an alias for v0).",
)
epic_app = typer.Typer(add_completion=False, help="Epic management.")
task_app = typer.Typer(add_completion=False, help="Task management.")
node_app = typer.Typer(add_completion=False, help="Node/branch graph management.")
agent_app = typer.Typer(add_completion=False, help="Agent management.")
linear_app = typer.Typer(add_completion=False, help="Linear integration.")


@dataclass(slots=True)
class SyncStats:
    epics_created: int = 0
    epics_updated: int = 0
    tasks_created: int = 0
    tasks_updated: int = 0
    nodes_created: int = 0
    nodes_updated: int = 0


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
        None, "--project", help="Linear project id (when syncing from/to Linear)."
    ),
    create_nodes: bool = typer.Option(
        True,
        "--create-nodes/--no-create-nodes",
        help="Create/update nodes in the branch graph.",
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
        src = from_.lower()
        if src == "local":
            stats = asyncio.run(
                _sync_from_local(ctx, epic=epic, create_nodes=create_nodes)
            )
            typer.echo(
                "Synced from local: "
                f"epics +{stats.epics_created}/~{stats.epics_updated}, "
                f"tasks +{stats.tasks_created}/~{stats.tasks_updated}, "
                f"nodes +{stats.nodes_created}/~{stats.nodes_updated}"
            )
            return
        if src == "linear":
            stats = asyncio.run(
                _sync_from_linear(
                    ctx,
                    epic=epic,
                    project=project,
                    create_nodes=create_nodes,
                )
            )
            typer.echo(
                "Synced from Linear: "
                f"epics +{stats.epics_created}/~{stats.epics_updated}, "
                f"tasks +{stats.tasks_created}/~{stats.tasks_updated}, "
                f"nodes +{stats.nodes_created}/~{stats.nodes_updated}"
            )
            return
        typer.echo("error: --from must be one of: local, linear", err=True)
        raise typer.Exit(2)

    dst = (to or "").lower()
    if dst == "linear":
        typer.echo("error: sync --to linear is not implemented yet", err=True)
        raise typer.Exit(2)
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

    asyncio.run(init_repo(ctx))
    typer.echo(f"Initialized: {ctx.state_dir}")
    typer.echo(f"DB: {ctx.db_path}")


def _ensure_initialized(ctx: RepoContext) -> None:
    if not ctx.db_path.exists():
        raise NotInitializedError(
            "Redesmyn is not initialized in this repo. Run `rn init`."
        )


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


_LOCAL_TASK_ID_RE = re.compile(r"^T-(?P<id>\d+)$")


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
    create_nodes: bool,
) -> SyncStats:
    epic_fs = _infer_single_epic_slug_from_fs(ctx.repo_root)
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

            epic_readme = ctx.repo_root / "epics" / epic_slug / "README.md"
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

            tasks_dir = ctx.repo_root / "epics" / epic_row.slug / "tasks"
            task_readmes = (
                sorted(tasks_dir.glob("*/README.md")) if tasks_dir.exists() else []
            )

            task_docs = []
            for readme in task_readmes:
                try:
                    task_docs.append(load_task_doc(readme))
                except DocLoadError as e:
                    raise typer.BadParameter(str(e)) from e

            node_by_ref: dict[str, Node] = {}
            node_by_path: dict[Path, Node] = {}

            for doc in task_docs:
                if doc.title is None:
                    raise typer.BadParameter(f"Task doc missing title (H1): {doc.path}")

                meta = doc.metadata
                linear_issue_id = meta.linear.issue_id if meta.linear else None
                linear_identifier = meta.linear.identifier if meta.linear else None
                rel_path = str(doc.path.relative_to(ctx.repo_root))

                task: Task | None = None
                if linear_issue_id:
                    task = await session.scalar(
                        select(Task).where(
                            Task.epic_id == epic_row.id,
                            Task.linear_issue_id == linear_issue_id,
                        )
                    )

                if task is None and meta.id:
                    m = _LOCAL_TASK_ID_RE.match(meta.id.strip())
                    if m:
                        candidate = await session.get(Task, int(m.group("id")))
                        if candidate is not None and candidate.epic_id == epic_row.id:
                            task = candidate

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
                        local_path=rel_path,
                    )
                    session.add(task)
                    await session.flush()
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
                    if updated:
                        stats.tasks_updated += 1

                if not create_nodes:
                    continue

                branch = meta.node.branch if meta.node and meta.node.branch else None
                if not branch:
                    identifier = linear_identifier or meta.id or f"task-{task.id}"
                    short = (
                        slugify(doc.title, fallback="task")[:60].strip("-") or "task"
                    )
                    branch = f"rn/{epic_row.slug}/{identifier}-{short}"

                node = await session.scalar(
                    select(Node).where(
                        Node.epic_id == epic_row.id,
                        Node.branch_name == branch,
                    )
                )
                if node is None:
                    node = Node(
                        epic_id=epic_row.id,
                        branch_name=branch,
                        linear_issue_id=linear_issue_id,
                    )
                    session.add(node)
                    await session.flush()
                    stats.nodes_created += 1
                else:
                    updated = False
                    if linear_issue_id and node.linear_issue_id != linear_issue_id:
                        node.linear_issue_id = linear_issue_id
                        updated = True
                    if updated:
                        stats.nodes_updated += 1

                task.node_id = node.id
                node.primary_task_id = task.id
                node_by_path[doc.path] = node

                for ref in (meta.id, linear_identifier, linear_issue_id):
                    if not ref:
                        continue
                    existing = node_by_ref.get(ref)
                    if existing is not None and existing.id != node.id:
                        raise typer.BadParameter(
                            f"Ambiguous task ref {ref!r}; matches multiple tasks in docs (v0)"
                        )
                    node_by_ref[ref] = node

            if create_nodes:
                for doc in task_docs:
                    node = node_by_path.get(doc.path)
                    if node is None:
                        continue
                    parent_ref = doc.metadata.stacked_on
                    if not parent_ref:
                        node.parent_node_id = None
                        continue
                    parent_node = node_by_ref.get(parent_ref)
                    if parent_node is None:
                        raise typer.BadParameter(
                            f"Unknown stacked_on ref {parent_ref!r} in {doc.path}"
                        )
                    node.parent_node_id = parent_node.id

            await session.commit()
            return stats
    finally:
        await engine.dispose()


async def _sync_from_linear(
    ctx: RepoContext,
    *,
    epic: str | None,
    project: str | None,
    create_nodes: bool,
) -> SyncStats:
    auth = await _load_linear_auth(ctx)
    if auth is None:
        raise typer.BadParameter("Linear is not connected. Run `rn linear auth`.")

    epic_row = await _resolve_epic(ctx, epic=epic)

    project_id = project or epic_row.linear_project_id
    if not project_id:
        epic_readme = ctx.repo_root / "epics" / epic_row.slug / "README.md"
        if epic_readme.exists():
            try:
                epic_doc = load_epic_doc(epic_readme)
                project_id = epic_doc.metadata.linear_project_id
            except DocLoadError:
                project_id = None

    if not project_id:
        raise typer.BadParameter(
            "Missing Linear project id. Pass --project <id> or set it in the epic doc metadata."
        )

    client = LinearClient(access_token=auth.access_token)
    issues = await fetch_project_issues(client, project_id=project_id)
    try:
        relations = await fetch_project_issue_relations(client, project_id=project_id)
    except Exception as e:
        typer.echo(
            f"warning: could not fetch issue relations; parent inference disabled ({e})",
            err=True,
        )
        relations = []

    issue_by_id = {i.id: i for i in issues}
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

    parent_issue_by_issue: dict[str, str | None] = {}
    for issue_id, blockers in blockers_by_issue.items():
        if not blockers:
            parent_issue_by_issue[issue_id] = None
            continue
        if len(blockers) > 1:
            raise typer.BadParameter(
                f"Linear issue {issue_id} has multiple blockers; choose a parent explicitly (v0)"
            )
        parent = blockers[0]
        if parent not in issue_by_id:
            raise typer.BadParameter(
                f"Linear issue {issue_id} blocker {parent} is outside the imported project (v0)"
            )
        parent_issue_by_issue[issue_id] = parent

    epic_dir = ctx.repo_root / "epics" / epic_row.slug
    tasks_dir = epic_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    epic_readme = epic_dir / "README.md"
    if epic_readme.exists():
        markdown = epic_readme.read_text(encoding="utf-8")
        yaml_data = {
            "slug": epic_row.slug,
            "name": epic_row.name,
            "root_branch": epic_row.root_branch,
            "linear": {"project_id": project_id},
        }
        epic_readme.write_text(
            upsert_metadata_yaml(markdown, yaml_data=yaml_data), encoding="utf-8"
        )

    def _branch_name(identifier: str, title: str) -> str:
        short = slugify(title, fallback="task")[:60].strip("-") or "task"
        return f"rn/{epic_row.slug}/{identifier}-{short}"

    for issue in issues:
        parent_issue_id = parent_issue_by_issue.get(issue.id)
        parent_identifier = (
            issue_by_id[parent_issue_id].identifier if parent_issue_id else None
        )

        readme = tasks_dir / issue.identifier / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)

        if readme.exists():
            markdown = readme.read_text(encoding="utf-8")
        else:
            markdown = f"# {issue.identifier} {issue.title}\n\n## Brief (local)\n\n"

        yaml_data = {
            "id": None,
            "group_under": None,
            "stacked_on": parent_identifier,
            "must_land_after": [],
            "linear": {"issue_id": issue.id, "identifier": issue.identifier},
            "node": {"branch": _branch_name(issue.identifier, issue.title)},
        }
        markdown = upsert_metadata_yaml(markdown, yaml_data=yaml_data)

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
        readme.write_text(markdown, encoding="utf-8")

    return await _sync_from_local(ctx, epic=epic_row.slug, create_nodes=create_nodes)


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
    typer.echo(f"Repo: {ctx.repo_root}")
    typer.echo(f"State: {ctx.state_dir}")
    typer.echo(f"DB: {ctx.db_path}")
    if repo is None:
        typer.echo("Initialized: no")
    else:
        typer.echo("Initialized: yes")
        typer.echo(f"Default branch: {repo.default_branch}")
    typer.echo(f"Block (git): {block_summary or 'none'}")


@daemon_app.command("run")
def daemon_run(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(9234, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes."),
) -> None:
    """Run the daemon in the foreground."""
    try:
        _ = get_repo_context()
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    try:
        import uvicorn
    except Exception as e:  # pragma: no cover
        typer.echo(f"error: uvicorn not available ({e})", err=True)
        raise typer.Exit(2)

    uvicorn.run("redesmyn.api:app", host=host, port=port, reload=reload)


@daemon_app.command("start")
def daemon_start() -> None:
    typer.echo("Not implemented yet. Use `rn daemon run` for now.", err=True)
    raise typer.Exit(2)


@daemon_app.command("stop")
def daemon_stop() -> None:
    typer.echo("Not implemented yet.", err=True)
    raise typer.Exit(2)


@daemon_app.command("status")
def daemon_status() -> None:
    typer.echo("Not implemented yet. Use `rn status` for now.", err=True)
    raise typer.Exit(2)


app.add_typer(daemon_app, name="daemon")


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
        node = f" node={t.node_id}" if t.node_id is not None else ""
        typer.echo(f"{t.id}:{node} {t.title}")


@task_app.command("link")
def task_link(
    task_id: int = typer.Argument(..., help="Task id."),
    node_id: int = typer.Option(..., "--node", help="Node id."),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> tuple[Task, Node]:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                task = await session.get(Task, task_id)
                if task is None:
                    raise typer.BadParameter(f"Unknown task id: {task_id}")
                node = await session.get(Node, node_id)
                if node is None:
                    raise typer.BadParameter(f"Unknown node id: {node_id}")
                task.node_id = node.id
                node.primary_task_id = task.id
                await session.commit()
                await session.refresh(task)
                await session.refresh(node)
                return task, node
        finally:
            await engine.dispose()

    task, node = asyncio.run(_run())
    typer.echo(f"Linked task {task.id} -> node {node.id} ({node.branch_name})")


app.add_typer(task_app, name="task")


@node_app.command("create")
def node_create(
    branch: str = typer.Option(..., "--branch", help="Branch name for this node."),
    parent: int | None = typer.Option(
        None, "--parent", help="Parent node id (defaults to epic root)."
    ),
    epic: str | None = typer.Option(
        None, help="Epic slug or id (defaults if only one epic)."
    ),
    worktree: Path | None = typer.Option(
        None, help="Worktree path (defaults under .redesmyn/worktrees)."
    ),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> Node:
        epic_row = await _resolve_epic(ctx, epic=epic)

        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                parent_node: Node | None = None
                if parent is not None:
                    parent_node = await session.get(Node, parent)
                    if parent_node is None:
                        raise typer.BadParameter(f"Unknown parent node id: {parent}")

                existing = await session.scalar(
                    select(Node).where(
                        Node.epic_id == epic_row.id,
                        Node.branch_name == branch,
                    )
                )
                if existing is not None:
                    return existing

                base_ref = (
                    parent_node.branch_name
                    if parent_node is not None
                    else epic_row.root_branch
                )
                worktree_path = worktree or _default_worktree_path(ctx, branch=branch)

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

                node = Node(
                    epic_id=epic_row.id,
                    branch_name=branch,
                    parent_node_id=parent_node.id if parent_node else None,
                    worktree_path=str(worktree_path),
                )
                session.add(node)
                await session.commit()
                await session.refresh(node)
                return node
        finally:
            await engine.dispose()

    node = asyncio.run(_run())
    typer.echo(
        f"Node: {node.id} branch={node.branch_name} worktree={node.worktree_path}"
    )


@node_app.command("list")
def node_list(
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

    async def _run() -> list[Node]:
        epic_row = await _resolve_epic(ctx, epic=epic)
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                rows = await session.scalars(
                    select(Node).where(Node.epic_id == epic_row.id).order_by(Node.id)
                )
                return list(rows)
        finally:
            await engine.dispose()

    nodes = asyncio.run(_run())
    if not nodes:
        typer.echo("No nodes.")
        return
    for n in nodes:
        parent_id = n.parent_node_id or "-"
        agent_id = n.agent_id or "-"
        wt = n.worktree_path or "-"
        typer.echo(
            f"{n.id}: branch={n.branch_name} parent={parent_id} agent={agent_id} wt={wt}"
        )


app.add_typer(node_app, name="node")


@agent_app.command("register")
def agent_register(
    name: str = typer.Option(..., "--name", help="Agent display name."),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> Agent:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                agent = Agent(display_name=name)
                session.add(agent)
                await session.commit()
                await session.refresh(agent)
                return agent
        finally:
            await engine.dispose()

    agent = asyncio.run(_run())
    typer.echo(f"Agent: {agent.id} {agent.display_name}")


@agent_app.command("list")
def agent_list() -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> list[Agent]:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                rows = await session.scalars(select(Agent).order_by(Agent.id))
                return list(rows)
        finally:
            await engine.dispose()

    agents = asyncio.run(_run())
    if not agents:
        typer.echo("No agents.")
        return
    for a in agents:
        typer.echo(f"{a.id}: {a.display_name} status={a.status.value}")


@agent_app.command("assign")
def agent_assign(
    agent_id: int = typer.Argument(..., help="Agent id."),
    node_id: int = typer.Option(..., "--node", help="Node id."),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    async def _run() -> tuple[Agent, Node]:
        engine = create_engine(ctx.db_path)
        try:
            sessionmaker = create_sessionmaker(engine)
            async with sessionmaker() as session:
                agent = await session.get(Agent, agent_id)
                if agent is None:
                    raise typer.BadParameter(f"Unknown agent id: {agent_id}")
                node = await session.get(Node, node_id)
                if node is None:
                    raise typer.BadParameter(f"Unknown node id: {node_id}")
                node.agent_id = agent.id
                await session.commit()
                await session.refresh(node)
                return agent, node
        finally:
            await engine.dispose()

    agent, node = asyncio.run(_run())
    typer.echo(f"Assigned agent {agent.id} -> node {node.id} ({node.branch_name})")


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


@linear_app.command("status")
def linear_status() -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    auth = asyncio.run(_load_linear_auth(ctx))
    if auth is None:
        typer.echo("Linear: not connected")
        raise typer.Exit(1)

    typer.echo("Linear: connected")
    typer.echo(f"Connected at: {auth.created_at.isoformat()}")


@linear_app.command("auth")
def linear_auth(
    wait: bool = typer.Option(True, help="Wait for authorization to complete."),
    timeout_seconds: int = typer.Option(180, help="Max time to wait for auth."),
) -> None:
    try:
        _ensure_initialized(get_repo_context())
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    from redesmyn.settings import load_settings

    settings = load_settings(repo_root=get_repo_context().repo_root)
    if not settings.linear_client_id or not settings.linear_client_secret:
        redirect_uri = (
            f"http://{settings.api_host}:{settings.api_port}/v1/linear/oauth/callback"
        )
        typer.echo(
            "error: Linear OAuth is not configured. Create a Linear OAuth app and set:\n"
            "  REDESMYN_LINEAR_CLIENT_ID\n"
            "  REDESMYN_LINEAR_CLIENT_SECRET\n"
            "in `.env` (see `.env.example`).\n"
            f"Redirect URL: {redirect_uri}",
            err=True,
        )
        raise typer.Exit(2)
    base = f"http://{settings.api_host}:{settings.api_port}"
    start_url = f"{base}/v1/linear/oauth/start"
    status_url = f"{base}/v1/linear/status"

    if not webbrowser.open(start_url):
        typer.echo(start_url)

    if not wait:
        return

    import httpx

    async def _poll() -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = asyncio.get_event_loop().time()
            while True:
                if asyncio.get_event_loop().time() - start > timeout_seconds:
                    raise typer.BadParameter(
                        "Timed out waiting for Linear authorization"
                    )

                try:
                    resp = await client.get(status_url)
                except httpx.RequestError:
                    raise typer.BadParameter(
                        f"Daemon not reachable at {base}. Run `rn daemon run`."
                    ) from None

                if resp.status_code != 200:
                    await asyncio.sleep(1.0)
                    continue

                payload = resp.json()
                if payload.get("connected"):
                    return
                await asyncio.sleep(1.0)

    try:
        asyncio.run(_poll())
    except typer.BadParameter as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    typer.echo("Linear: connected")


@linear_app.command("whoami")
def linear_whoami() -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    auth = asyncio.run(_load_linear_auth(ctx))
    if auth is None:
        typer.echo("error: Linear is not connected. Run `rn linear auth`.", err=True)
        raise typer.Exit(2)

    async def _run() -> None:
        client = LinearClient(access_token=auth.access_token)
        data = await client.graphql("query { viewer { id name email } }")
        typer.echo(str(data.get("viewer")))

    asyncio.run(_run())


def _task_state_from_linear(state_type: str | None) -> TaskState:
    if state_type is None:
        return TaskState.Todo
    normalized = state_type.lower()
    if normalized in {"started", "in_progress"}:
        return TaskState.InProgress
    if normalized in {"completed", "canceled"}:
        return TaskState.Done
    if normalized == "blocked":
        return TaskState.Blocked
    return TaskState.Todo


@linear_app.command("import")
def linear_import(
    project: str = typer.Option(..., "--project", help="Linear project id."),
    epic: str | None = typer.Option(
        None, help="Epic slug or id (defaults if only one epic)."
    ),
    create_nodes: bool = typer.Option(
        True,
        "--create-nodes/--no-create-nodes",
        help="Create nodes/branches by default.",
    ),
) -> None:
    try:
        ctx = get_repo_context()
        _ensure_initialized(ctx)
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    auth = asyncio.run(_load_linear_auth(ctx))
    if auth is None:
        typer.echo("error: Linear is not connected. Run `rn linear auth`.", err=True)
        raise typer.Exit(2)

    epic_row = asyncio.run(_resolve_epic(ctx, epic=epic))
    client = LinearClient(access_token=auth.access_token)

    async def _fetch() -> tuple[list[LinearIssue], list[LinearIssueRelation]]:
        issues = await fetch_project_issues(client, project_id=project)
        try:
            relations = await fetch_project_issue_relations(client, project_id=project)
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

                node_by_issue_id: dict[str, Node] = {}
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

                    if not create_nodes:
                        continue

                    branch = _branch_name(issue.identifier, issue.title)
                    node = await session.scalar(
                        select(Node).where(
                            Node.epic_id == epic_db.id,
                            Node.branch_name == branch,
                        )
                    )
                    if node is None:
                        node = Node(
                            epic_id=epic_db.id,
                            branch_name=branch,
                            linear_issue_id=issue.id,
                        )
                        session.add(node)
                        await session.flush()
                    else:
                        node.linear_issue_id = issue.id

                    node_by_issue_id[issue.id] = node

                if not create_nodes:
                    await session.commit()
                    return

                for issue in issues_sorted:
                    node = node_by_issue_id[issue.id]
                    parent_issue_id = parent_by_issue.get(issue.id)
                    parent_node = (
                        node_by_issue_id[parent_issue_id]
                        if parent_issue_id is not None
                        else None
                    )
                    node.parent_node_id = parent_node.id if parent_node else None

                    base_ref = (
                        parent_node.branch_name if parent_node else epic_db.root_branch
                    )
                    if node.worktree_path is None:
                        branch = node.branch_name
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
                        node.worktree_path = str(worktree_path)

                    task = await session.scalar(
                        select(Task).where(
                            Task.epic_id == epic_db.id,
                            Task.linear_issue_id == issue.id,
                        )
                    )
                    if task is not None:
                        task.node_id = node.id
                        node.primary_task_id = task.id

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
app.add_typer(block_app, name="pause")


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
    raise typer.Exit(proc.returncode)


def main() -> None:
    try:
        app()
    except BrokenPipeError:
        raise typer.Exit(141) from None
    except KeyboardInterrupt:
        typer.echo("", err=True)
        raise typer.Exit(130) from None
