from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from sqlalchemy import select

from redesmyn import __version__
from redesmyn.context import RepoContext, get_repo_context
from redesmyn.db import Repository, create_engine, create_sessionmaker
from redesmyn.orchestrator import init_repo
from redesmyn.repo import NotAGitRepositoryError

app = typer.Typer(add_completion=False, help="Redesmyn CLI (`rn`).")
daemon_app = typer.Typer(add_completion=False, help="Daemon management.")


@app.command()
def version() -> None:
    print(__version__)


@app.command()
def init(cwd: Path | None = typer.Option(None, help="Run from this directory.")) -> None:
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
            return await session.scalar(select(Repository).where(Repository.repo_root == str(ctx.repo_root)))
    finally:
        await engine.dispose()


@app.command()
def status(cwd: Path | None = typer.Option(None, help="Run from this directory.")) -> None:
    """Show current repo orchestration status."""
    try:
        ctx = get_repo_context(cwd=cwd)
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    repo = asyncio.run(_load_repository_row(ctx))
    typer.echo(f"Repo: {ctx.repo_root}")
    typer.echo(f"State: {ctx.state_dir}")
    typer.echo(f"DB: {ctx.db_path}")
    if repo is None:
        typer.echo("Initialized: no")
    else:
        typer.echo("Initialized: yes")
        typer.echo(f"Default branch: {repo.default_branch}")


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


def main() -> None:
    try:
        app()
    except BrokenPipeError:
        raise typer.Exit(141) from None
    except KeyboardInterrupt:
        typer.echo("", err=True)
        raise typer.Exit(130) from None
