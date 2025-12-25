from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import typer
from sqlalchemy import select

from redesmyn import __version__
from redesmyn.blocks import (
    NotInitializedError,
    clear_block,
    get_effective_block,
    list_blocks,
    set_manual_block,
)
from redesmyn.context import RepoContext, get_repo_context
from redesmyn.db import Repository, create_engine, create_sessionmaker
from redesmyn.domain.enums import BlockPolicy
from redesmyn.git_proxy import block_blocks_git
from redesmyn.orchestrator import init_repo
from redesmyn.repo import NotAGitRepositoryError, current_branch

app = typer.Typer(add_completion=False, help="Redesmyn CLI (`rn`).")
daemon_app = typer.Typer(add_completion=False, help="Daemon management.")
block_app = typer.Typer(
    add_completion=False,
    help="Block controls (use `rn pause` as an alias for v0).",
)


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
        decision = block_blocks_git(git_args, mode=block.mode)
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
