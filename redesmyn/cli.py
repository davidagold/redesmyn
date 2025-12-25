from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import typer
from sqlalchemy import select

from redesmyn import __version__
from redesmyn.context import RepoContext, get_repo_context
from redesmyn.db import Repository, create_engine, create_sessionmaker
from redesmyn.git_proxy import pause_blocks_git
from redesmyn.orchestrator import init_repo
from redesmyn.pause import NotInitializedError, clear_pause, get_effective_pause, list_pauses, set_pause
from redesmyn.repo import NotAGitRepositoryError, current_branch

app = typer.Typer(add_completion=False, help="Redesmyn CLI (`rn`).")
daemon_app = typer.Typer(add_completion=False, help="Daemon management.")
pause_app = typer.Typer(add_completion=False, help="Pause controls.")


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


async def _load_pause_summary(ctx: RepoContext, *, branch: str | None) -> str | None:
    try:
        pause = await get_effective_pause(ctx, branch=branch)
    except NotInitializedError:
        return None

    if pause is None:
        return None

    reason = pause.reason or "n/a"
    return f"{pause.mode} (scope={pause.scope}, reason={reason})"


@app.command()
def status(cwd: Path | None = typer.Option(None, help="Run from this directory.")) -> None:
    """Show current repo orchestration status."""
    try:
        ctx = get_repo_context(cwd=cwd)
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    repo = asyncio.run(_load_repository_row(ctx))
    branch = current_branch(cwd=cwd or Path.cwd())
    pause_summary = asyncio.run(_load_pause_summary(ctx, branch=branch))
    typer.echo(f"Repo: {ctx.repo_root}")
    typer.echo(f"State: {ctx.state_dir}")
    typer.echo(f"DB: {ctx.db_path}")
    if repo is None:
        typer.echo("Initialized: no")
    else:
        typer.echo("Initialized: yes")
        typer.echo(f"Default branch: {repo.default_branch}")
    typer.echo(f"Pause: {pause_summary or 'none'}")


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


@pause_app.command("lax")
def pause_lax(
    scope: str = typer.Option("repo", help="Pause scope (v0: use 'repo')."),
    reason: str | None = typer.Option(None, help="Human-readable reason."),
) -> None:
    try:
        repo_ctx = get_repo_context()
        pause = asyncio.run(set_pause(repo_ctx, scope=scope, mode="lax", reason=reason))
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    typer.echo(f"Paused: {pause.mode} (scope={pause.scope})")


@pause_app.command("strict")
def pause_strict(
    scope: str = typer.Option("repo", help="Pause scope (v0: use 'repo')."),
    reason: str | None = typer.Option(None, help="Human-readable reason."),
) -> None:
    try:
        repo_ctx = get_repo_context()
        pause = asyncio.run(set_pause(repo_ctx, scope=scope, mode="strict", reason=reason))
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    typer.echo(f"Paused: {pause.mode} (scope={pause.scope})")


@pause_app.command("clear")
def pause_clear(
    scope: str = typer.Option("repo", help="Pause scope (v0: use 'repo')."),
    reason: str | None = typer.Option(None, help="Reason for clearing."),
) -> None:
    try:
        repo_ctx = get_repo_context()
        pause = asyncio.run(clear_pause(repo_ctx, scope=scope, reason=reason))
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if pause is None:
        typer.echo("No active pause.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Pause cleared: {pause.mode} (scope={pause.scope})")


@pause_app.command("list")
def pause_list(
    scope: str = typer.Option("repo", help="Pause scope (v0: use 'repo')."),
) -> None:
    try:
        repo_ctx = get_repo_context()
        pauses = asyncio.run(list_pauses(repo_ctx, scope=scope))
    except (NotAGitRepositoryError, NotInitializedError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if not pauses:
        typer.echo("No pauses.")
        return

    for p in pauses:
        cleared = "active" if p.cleared_at is None else "cleared"
        typer.echo(f"{p.id}: {p.mode} {cleared} scope={p.scope} reason={p.reason or 'n/a'}")


app.add_typer(pause_app, name="pause")


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
        pause = asyncio.run(get_effective_pause(repo_ctx, branch=branch))
    except NotInitializedError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except NotAGitRepositoryError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)

    if pause is not None:
        decision = pause_blocks_git(git_args, mode=pause.mode)
        if not decision.allowed:
            reason = pause.reason or "n/a"
            typer.echo(
                f"blocked: {decision.reason} (scope={pause.scope}, reason={reason})",
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
